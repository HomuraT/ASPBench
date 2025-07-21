#!/bin/bash

# 定义 Python 脚本的路径 (假设此 bash 脚本从项目根目录运行)
PYTHON_SCRIPT="01_raw_symtex_generation.py"
# 定义存放所有批量运行结果的基础目录
BASE_OUTPUT_DIR="datasets/symtex_batch_runs"
# 定义 Python 脚本自动生成文件存放的具体目录
AUTO_GEN_DIR="${BASE_OUTPUT_DIR}/auto_generated_files"

# 确保输出目录存在
mkdir -p "$AUTO_GEN_DIR"

echo "Starting SymTex generation batch run..."
echo "Output will be saved in subdirectories under: $AUTO_GEN_DIR"
echo "========================================"

# 循环遍历参数范围
for num_nodes in $(seq 2 10); do
  # 计算 num_edges 的范围 (确保是整数)
  min_edges=$((num_nodes - 1))
  # 使用 bc 进行浮点数计算并向上取整
  max_edges=$(echo "scale=0; ($num_nodes * 1.1 + 0.999)/1" | bc)
  # 确保 min_edges 不小于 0
  if [ $min_edges -lt 0 ]; then
    min_edges=0
  fi
  # 如果 max_edges 小于 min_edges (例如 num_nodes=1), 则将 max_edges 设置为 min_edges
  if [ $max_edges -lt $min_edges ]; then
      max_edges=$min_edges
  fi
  # 如果 min_edges > max_edges (理论上不应发生，但作为保险)
  if [ $min_edges -gt $max_edges ]; then
      echo "Warning: Calculated min_edges ($min_edges) > max_edges ($max_edges) for num_nodes=$num_nodes. Skipping edges loop."
      continue
  fi


  for num_edges in $(seq $min_edges $max_edges); do
    for extra_predicate_num in $(seq 1 $num_edges); do
      min_extra_edge=$extra_predicate_num
      max_extra_edge=$((extra_predicate_num + 3))
      for extra_edge_num in $(seq $min_extra_edge $max_extra_edge); do
        for max_predicates_per_rule in $(seq 3 4); do
          # 注意：num_noise_rules_per_type 从 5 递减到 0
          for num_noise_rules_per_type in $(seq 3 -1 0); do
            for m in $(seq 0 3); do
              for largest in $(seq 0 3); do
                largest=$((m + 1))

                # --- 文件存在性检查 ---
                # 定义 Python 脚本使用的默认概率值
                strong_negation_prob_default=0.5
                default_negation_prob_default=1.0

                # 根据 Python 脚本的 _generate_output_filename 函数构造预期文件名
                # 注意：文件名不包含 num_noise_rules_per_type
                expected_filename="symtex_dataset_nodes${num_nodes}_edges${num_edges}_extraP${extra_predicate_num}_extraE${extra_edge_num}_sNeg${strong_negation_prob_default}_dNeg${default_negation_prob_default}_maxPred${max_predicates_per_rule}_m${m}_l${largest}.jsonl"
                expected_file_path="${AUTO_GEN_DIR}/${expected_filename}"

                # 检查文件是否存在
                if [ -f "$expected_file_path" ]; then
                  echo "Skipping: Output file already exists: $expected_file_path"
                  break # 跳过当前 largest 循环的迭代
                fi
                # --- 文件存在性检查结束 ---


                # 构建传递给 Python 脚本的参数
                # 使用 --output_dir 让 Python 脚本自动管理文件名并放入指定目录
                args="--num_nodes $num_nodes \
                      --num_edges $num_edges \
                      --extra_predicate_num $extra_predicate_num \
                      --extra_edge_num $extra_edge_num \
                      --max_predicates_per_rule $max_predicates_per_rule \
                      --num_noise_rules_per_type $num_noise_rules_per_type \
                      --m $m \
                      --largest $largest \
                      --output_dir $AUTO_GEN_DIR" # 指定输出目录

                echo "Running: python $PYTHON_SCRIPT $args"
                # 执行 Python 脚本
                python $PYTHON_SCRIPT $args

                # 检查 Python 脚本的退出状态
                status=$?
                if [ $status -ne 0 ]; then
                  echo "Error (Exit Code: $status) running script with params: $args"
                  # 您可以选择在这里添加错误处理逻辑，例如记录错误或退出脚本
                  # echo "Failed command: python $PYTHON_SCRIPT $args" >> error_log.txt
                  # exit 1 # 如果希望在第一个错误时停止
                else
                  echo "Successfully completed run for params above."
                fi
                echo "----------------------------------------"
                # 可选：添加短暂延迟以避免系统过载
                # sleep 0.1

                # 确保标记文件存在 (如果不存在则创建空文件)
                touch "$expected_file_path"

              done # largest
            done # m
          done # num_noise_rules_per_type
        done # max_predicates_per_rule
      done # extra_edge_num
    done # extra_predicate_num
  done # num_edges
done # num_nodes

echo "========================================"
echo "All parameter combinations processed."
echo "Check $AUTO_GEN_DIR for generated datasets."
