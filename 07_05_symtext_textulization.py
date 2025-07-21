# -*- coding: utf-8 -*-
"""
脚本：07_05_symtext_textulization.py
功能：对指定输入文件夹中的所有 .jsonl 文件进行文本化处理。
      使用 TextulizationFramework 将符号化的事实和规则转换为自然语言描述，
      并将结果保存到指定的输出文件夹中，文件名与原文件相同。

用法：
python 07_05_symtext_textulization.py --input-dir <输入文件夹路径> [--output-dir <输出文件夹路径>] [--model-name <模型名称>] [--num-threads <线程数>] [--save-interval <保存间隔>] [--log-level <日志级别>]

示例：
python 07_05_symtext_textulization.py --input-dir datasets/symtex_final --output-dir datasets/symtex_final_textual --model-name mmm_gpt_4o_mini --num-threads 8 --log-level INFO
"""

import os
import sys
import argparse
import logging
import glob
from typing import List, Dict, Any

# 确保 src 目录在 Python 路径中，以便导入自定义模块
# 获取当前脚本文件所在的目录 (项目根目录)
project_root = os.path.dirname(os.path.abspath(__file__))
# 将 src 目录添加到 sys.path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    # 从原始模块导入核心处理类
    from src.dataset_generation.textulization import TextulizationFramework
except ImportError as e:
    print(f"Error importing TextulizationFramework: {e}")
    print("Please ensure the script is run from the project root directory and the 'src' folder exists.")
    sys.exit(1)

# --- 日志配置 ---
# 配置基础日志记录器
# 日志级别可以通过命令行参数覆盖
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)
# 获取此脚本的日志记录器实例
logger = logging.getLogger(__name__)


def process_directory(input_dir: str, output_dir: str, model_name: str, num_threads: int, save_interval: int) -> None:
    """
    处理输入目录中的所有 .jsonl 文件。

    :param input_dir: 包含 .jsonl 文件的输入目录路径。
    :type input_dir: str
    :param output_dir: 保存文本化结果的输出目录路径。
    :type output_dir: str
    :param model_name: 用于文本化的 LLM 模型名称。
    :type model_name: str
    :param num_threads: 用于处理每个文件的线程数。
    :type num_threads: int
    :param save_interval: 写入输出文件时刷新缓冲区的频率（项目数）。
    :type save_interval: int
    :return: None
    :rtype: None
    """
    logger.critical(f"Starting textualization process for directory: {input_dir}")
    logger.critical(f"Output will be saved to: {output_dir}")
    logger.critical(f"Using model: {model_name}, Threads per file: {num_threads}, Save interval: {save_interval}")

    # --- 1. 确保输出目录存在 ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
        return # 无法创建输出目录则退出

    # --- 2. 初始化 TextulizationFramework ---
    try:
        # 注意：TextulizationFramework 内部会配置自己的日志记录器
        # 这里的日志级别设置会影响其子日志记录器，除非它们被单独配置
        framework = TextulizationFramework(model_name=model_name)
        logger.info("TextulizationFramework initialized successfully.")
    except RuntimeError as e:
        logger.error(f"Framework initialization failed: {e}", exc_info=True)
        return # 框架初始化失败则退出
    except Exception as e:
        logger.error(f"An unexpected error occurred during framework initialization: {e}", exc_info=True)
        return # 其他初始化错误

    # --- 3. 查找输入目录中的所有 .jsonl 文件 ---
    # 使用 glob 查找输入目录下一级的 .jsonl 文件
    # 注意：这不会递归查找子目录
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))

    if not jsonl_files:
        logger.warning(f"No .jsonl files found directly in the input directory: {input_dir}")
        return

    logger.critical(f"Found {len(jsonl_files)} .jsonl files to process.")

    # --- 4. 逐个处理文件 ---
    total_files = len(jsonl_files)
    for i, input_file_path in enumerate(jsonl_files):
        filename = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, filename)

        logger.critical(f"--- Processing file {i+1}/{total_files}: {filename} ---")
        logger.critical(f"Input: {input_file_path}")
        logger.critical(f"Output: {output_file_path}")

        try:
            # 调用框架的处理函数
            # process_dataset 会处理加载、多线程处理、断点续存和写入
            framework.process_dataset(
                input_file_path=input_file_path,
                output_file_path=output_file_path,
                num_threads=num_threads,
                save_interval=save_interval
            )
            logger.critical(f"--- Finished processing file: {filename} ---")
        except Exception as e:
            # 捕获 process_dataset 可能抛出的任何未处理异常
            logger.error(f"An unexpected error occurred while processing file {filename}: {e}", exc_info=True)
            # 继续处理下一个文件

    logger.info("All files in the directory have been processed.")


def main():
    """
    主函数：解析命令行参数并启动处理流程。
    """
    parser = argparse.ArgumentParser(description="Textualize all .jsonl files in a directory.")

    parser.add_argument("--input-dir", default='datasets/symtex_final', type=str,
                        help="Path to the input directory containing .jsonl files.")
    parser.add_argument("--output-dir", type=str, default="datasets/symtex_final_textual",
                        help="Path to the output directory to save textualized files. Defaults to 'datasets/symtex_final_textual'.")
    parser.add_argument("--model-name", type=str, default="mmm_gpt_4o_mini",
                        help="Name of the LLM model to use for textualization. Defaults to 'mmm_gpt_4o_mini'.")
    parser.add_argument("--num-threads", type=int, default=32,
                        help="Number of threads to use for processing each file. Defaults to 8.")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Flush output file buffer every N processed items. Defaults to 10.")
    parser.add_argument("--log-level", type=str, default="CRITICAL",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Defaults to INFO.")

    args = parser.parse_args()

    # --- 设置日志级别 ---
    log_level_numeric = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level_numeric, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    # 设置根日志记录器级别，这将影响所有子日志记录器（包括框架内部的）
    logging.getLogger().setLevel(log_level_numeric)
    # 也可以单独设置此脚本的日志记录器级别
    logger.setLevel(log_level_numeric)
    logger.info(f"Logging level set to: {args.log_level}")


    # --- 启动处理 ---
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_threads=args.num_threads,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
