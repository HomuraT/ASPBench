#!/usr/bin/env python3
"""
简化版ASC（Answer Set Generation）任务最困难样本提取工具

用法:
    python 00_extract_hardest_asc_samples.py --dataset_path dataset.jsonl --results_dir experiments/results --output_path output.jsonl --num_hardest 100 --hardest_percentage 70.0
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import glob

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_jsonl(file_path: Path) -> List[Dict]:
    """读取JSONL文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析失败: {e}")
    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
        raise
    return data


def write_jsonl(data: List[Dict], file_path: Path) -> None:
    """写入JSONL文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"成功写入 {len(data)} 个样本到 {file_path}")
    except Exception as e:
        logger.error(f"写入文件失败 {file_path}: {e}")
        raise


def normalize_atom(atom: str) -> str:
    """标准化原子字符串用于比较"""
    if not isinstance(atom, str):
        atom = str(atom)
    return atom.strip().replace(' ', '').replace('.', '').lower()


def is_exact_match(predicted_atoms: List[str], golden_sets: List[List[str]]) -> bool:
    """
    检查预测的原子集是否与任何一个正确答案集完全匹配
    
    Args:
        predicted_atoms: 预测的原子列表
        golden_sets: 正确答案集的列表（可能有多个正确答案）
    
    Returns:
        bool: 是否完全匹配
    """
    if not predicted_atoms:
        predicted_atoms = []
    
    if not golden_sets:
        return len(predicted_atoms) == 0
    
    # 标准化预测结果
    predicted_set = set(normalize_atom(atom) for atom in predicted_atoms)
    
    # 检查是否与任何一个正确答案集匹配
    for golden_atoms in golden_sets:
        if not isinstance(golden_atoms, list):
            continue
        golden_set = set(normalize_atom(atom) for atom in golden_atoms)
        if predicted_set == golden_set:
            return True
    
    return False


def find_prediction_files(results_dir: Path) -> List[Path]:
    """在结果目录中找到所有预测文件"""
    pattern = str(results_dir / "*_generated.jsonl")
    files = [Path(f) for f in glob.glob(pattern)]
    logger.info(f"在 {results_dir} 中找到 {len(files)} 个预测文件")
    return files


def extract_model_name(file_path: Path) -> str:
    """从文件路径提取模型名称"""
    # 文件名格式: openkg_ASC_xxx_yyy_generated.jsonl
    # 提取中间部分作为模型名
    name = file_path.stem
    if name.endswith('_generated'):
        name = name[:-10]  # 移除 '_generated'
    # 移除前缀 'openkg_ASC_' 如果存在
    if name.startswith('openkg_ASC_'):
        name = name[11:]
    return name


def evaluate_sample_performance(
    dataset: List[Dict], 
    prediction_files: List[Path]
) -> Dict[str, Dict[str, bool]]:
    """
    评估每个样本在各个模型上的表现
    
    Args:
        dataset: 原始数据集
        prediction_files: 预测文件路径列表
    
    Returns:
        Dict[sample_id, Dict[model_name, is_correct]]
    """
    # 构建原始数据映射
    dataset_map = {item['id']: item for item in dataset if 'id' in item}
    
    # 结果存储
    sample_performance = defaultdict(dict)
    
    # 初始化所有样本的所有模型结果为False
    model_names = [extract_model_name(f) for f in prediction_files]
    for sample_id in dataset_map.keys():
        for model_name in model_names:
            sample_performance[sample_id][model_name] = False
    
    # 处理每个预测文件
    for pred_file in prediction_files:
        model_name = extract_model_name(pred_file)
        logger.info(f"处理模型: {model_name}")
        
        try:
            predictions = read_jsonl(pred_file)
            pred_map = {item['id']: item for item in predictions if 'id' in item}
            
            for sample_id, original_item in dataset_map.items():
                pred_item = pred_map.get(sample_id)
                
                if not pred_item:
                    # 预测结果中没有该样本，默认为错误
                    continue
                
                # 获取预测结果
                predicted_atoms = pred_item.get('aligned_answer_set', [])
                
                # 获取正确答案集（优先从原始数据获取）
                golden_sets = original_item.get('answer_sets')
                if not golden_sets:
                    # 如果原始数据中没有，尝试从预测结果获取
                    golden_sets = pred_item.get('golden_answet_sets', [])  # 注意拼写错误
                
                if not golden_sets:
                    logger.warning(f"样本 {sample_id} 没有找到正确答案集")
                    continue
                
                # 判断是否完全匹配
                is_correct = is_exact_match(predicted_atoms, golden_sets)
                sample_performance[sample_id][model_name] = is_correct
                
        except Exception as e:
            logger.error(f"处理预测文件 {pred_file} 时出错: {e}")
    
    return sample_performance


def calculate_sample_difficulty(
    sample_performance: Dict[str, Dict[str, bool]]
) -> List[Tuple[str, float, Dict[str, bool]]]:
    """
    计算每个样本的困难度（平均正确率，越低越困难）
    
    Returns:
        List of (sample_id, avg_accuracy, model_results)
    """
    sample_difficulty = []
    
    for sample_id, model_results in sample_performance.items():
        if not model_results:
            continue
        
        correct_count = sum(1 for is_correct in model_results.values() if is_correct)
        total_models = len(model_results)
        avg_accuracy = correct_count / total_models if total_models > 0 else 0.0
        
        sample_difficulty.append((sample_id, avg_accuracy, model_results))
    
    # 按平均正确率升序排序（正确率越低越困难）
    sample_difficulty.sort(key=lambda x: (x[1], x[0]))
    
    return sample_difficulty


def extract_hardest_samples(
    dataset_path: str,
    results_dir: str, 
    output_path: str,
    num_hardest: int,
    hardest_percentage: float = 70.0
) -> None:
    """
    提取最困难的ASC样本（混合最困难样本和随机样本）
    
    Args:
        dataset_path: 原始数据集路径
        results_dir: 预测结果目录路径  
        output_path: 输出文件路径
        num_hardest: 要提取的总样本数量
        hardest_percentage: 最困难样本的百分比，剩余部分随机抽取（默认70%）
    """
    logger.info("开始提取ASC样本（混合最困难样本和随机样本）...")
    
    # 转换为Path对象
    dataset_path = Path(dataset_path)
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    
    # 验证输入
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    if not results_dir.exists():
        raise FileNotFoundError(f"结果目录不存在: {results_dir}")
    
    # 读取原始数据集
    logger.info(f"读取数据集: {dataset_path}")
    dataset = read_jsonl(dataset_path)
    logger.info(f"数据集包含 {len(dataset)} 个样本")
    
    # 查找预测文件
    prediction_files = find_prediction_files(results_dir)
    if not prediction_files:
        raise ValueError(f"在 {results_dir} 中没有找到预测文件 (*_generated.jsonl)")
    
    # 评估样本表现
    logger.info("评估样本在各模型上的表现...")
    sample_performance = evaluate_sample_performance(dataset, prediction_files)
    
    # 计算样本困难度
    logger.info("计算样本困难度...")
    sample_difficulty = calculate_sample_difficulty(sample_performance)
    
    if not sample_difficulty:
        logger.warning("没有有效的样本评估结果")
        return
    
    # 确定要提取的样本数量
    actual_num_total = min(num_hardest, len(sample_difficulty))
    logger.info(f"将提取总共 {actual_num_total} 个样本")
    
    # 计算最困难样本的数量
    num_hardest_samples = int(actual_num_total * hardest_percentage / 100)
    num_random_samples = actual_num_total - num_hardest_samples
    
    logger.info(f"最困难样本: {num_hardest_samples} 个 ({hardest_percentage}%)")
    logger.info(f"随机样本: {num_random_samples} 个 ({100-hardest_percentage:.1f}%)")
    
    # 选择最困难的样本
    hardest_samples_info = sample_difficulty[:num_hardest_samples]
    logger.info(f"已选择最困难的 {len(hardest_samples_info)} 个样本")
    
    # 从剩余样本中随机选择
    if num_random_samples > 0:
        remaining_samples = sample_difficulty[num_hardest_samples:]
        if len(remaining_samples) > 0:
            import random
            random.seed(42)  # 设置随机种子以确保可重复性
            random_samples = random.sample(remaining_samples, min(num_random_samples, len(remaining_samples)))
            hardest_samples_info.extend(random_samples)
            logger.info(f"已随机选择 {len(random_samples)} 个额外样本")
        else:
            logger.warning("没有剩余样本可供随机选择")
    
    # 统计最终选择的样本
    final_correct_count = sum(1 for _, avg_acc, _ in hardest_samples_info if avg_acc > 0)
    final_correct_percentage = (final_correct_count / len(hardest_samples_info)) * 100 if hardest_samples_info else 0
    logger.info(f"最终选择的样本中，{final_correct_count}/{len(hardest_samples_info)} ({final_correct_percentage:.1f}%) 曾被正确预测过")
    
    # 构建输出数据
    dataset_map = {item['id']: item for item in dataset}
    output_data = []
    
    for sample_id, avg_accuracy, model_results in hardest_samples_info:
        if sample_id not in dataset_map:
            logger.warning(f"样本 {sample_id} 在原始数据集中未找到")
            continue
        
        # 复制原始样本数据
        sample_data = dataset_map[sample_id].copy()
        
        # 添加困难度统计信息
        correct_models = [model for model, is_correct in model_results.items() if is_correct]
        incorrect_models = [model for model, is_correct in model_results.items() if not is_correct]
        
        sample_data['extraction_metrics'] = {
            'average_exact_match_across_models': round(avg_accuracy, 4),
            'correct_predictions_count': len(correct_models),
            'total_models_evaluated_against': len(model_results),
            'correct_model_names': correct_models,
            'incorrect_model_names': incorrect_models
        }
        
        output_data.append(sample_data)
    
    # 写入输出文件
    write_jsonl(output_data, output_path)
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("提取完成！统计信息:")
    logger.info(f"总样本数: {len(dataset)}")
    logger.info(f"评估的模型数: {len(prediction_files)}")
    logger.info(f"提取的总样本数: {len(output_data)}")
    logger.info(f"最困难样本比例: {hardest_percentage}%")
    logger.info(f"随机样本比例: {100-hardest_percentage:.1f}%")
    
    if output_data:
        final_correct_count = sum(1 for _, avg_acc, _ in hardest_samples_info if avg_acc > 0)
        final_correct_percentage = (final_correct_count / len(hardest_samples_info)) * 100
        logger.info(f"实际正确预测过的样本: {final_correct_count}/{len(hardest_samples_info)} ({final_correct_percentage:.1f}%)")
        
        # 分别统计最困难样本和随机样本的信息
        num_hardest_actual = min(int(len(hardest_samples_info) * hardest_percentage / 100), len(hardest_samples_info))
        hardest_part = hardest_samples_info[:num_hardest_actual]
        random_part = hardest_samples_info[num_hardest_actual:]
        
        if hardest_part:
            hardest_correct_count = sum(1 for _, avg_acc, _ in hardest_part if avg_acc > 0)
            logger.info(f"最困难样本中正确预测过的: {hardest_correct_count}/{len(hardest_part)} ({hardest_correct_count/len(hardest_part)*100:.1f}%)")
        
        if random_part:
            random_correct_count = sum(1 for _, avg_acc, _ in random_part if avg_acc > 0)
            logger.info(f"随机样本中正确预测过的: {random_correct_count}/{len(random_part)} ({random_correct_count/len(random_part)*100:.1f}%)")
        
        logger.info("\n最困难的前5个样本:")
        for i, (sample_id, avg_accuracy, model_results) in enumerate(hardest_samples_info[:5]):
            correct_count = sum(1 for is_correct in model_results.values() if is_correct)
            total_models = len(model_results)
            status = "✓" if avg_accuracy > 0 else "✗"
            sample_type = "困难" if i < num_hardest_actual else "随机"
            logger.info(f"  {i+1}. [{sample_type}] ID: {sample_id}, 平均正确率: {avg_accuracy:.4f} ({correct_count}/{total_models}) {status}")


def main():
    parser = argparse.ArgumentParser(
        description="提取ASC任务中最困难的样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python 00_extract_hardest_asc_samples.py \\
    --dataset_path datasets/openkg_subset/openkg_ASC.jsonl \\
    --results_dir experiments/openkg/full \\
    --output_path datasets/hardest_samples/openkg_ASC_hardest_100.jsonl \\
    --num_hardest 100 \\
    --hardest_percentage 70.0
        """
    )
    
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        required=True,
        help='原始数据集文件路径 (JSONL格式)'
    )
    
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True,
        help='包含预测结果文件的目录路径'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True,
        help='输出文件路径 (JSONL格式)'
    )
    
    parser.add_argument(
        '--num_hardest', 
        type=int, 
        required=True,
        help='要提取的最困难样本数量'
    )
    
    parser.add_argument(
        '--hardest_percentage', 
        type=float, 
        default=50.0,
        help='最困难样本的百分比，剩余部分将随机抽取 (默认: 70.0)'
    )
    
    args = parser.parse_args()
    
    try:
        extract_hardest_samples(
            dataset_path=args.dataset_path,
            results_dir=args.results_dir,
            output_path=args.output_path,
            num_hardest=args.num_hardest,
            hardest_percentage=args.hardest_percentage
        )
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 