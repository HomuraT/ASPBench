#!/bin/bash

# 定义 Python 脚本的路径 (假设此 bash 脚本从项目根目录运行)
PYTHON_SCRIPT="01_raw_symtex_generation.py"
# 定义存放所有批量运行结果的基础目录
BASE_OUTPUT_DIR="datasets/symtex_batch_runs"
# 定义 Python 脚本自动生成文件存放的具体目录
AUTO_GEN_DIR="${BASE_OUTPUT_DIR}/auto_generated_files"
# 定义最大并行进程数
MAX_PROCS=128

# 确保输出目录存在
mkdir -p "$AUTO_GEN_DIR"

echo "Starting SymTex generation batch run in parallel (Max Procs: $MAX_PROCS)..."
echo "Output will be saved in subdirectories under: $AUTO_GEN_DIR"
echo "========================================"

# 计数器，用于跟踪当前运行的后台作业数
job_count=0

# 循环遍历参数范围
for num_nodes in $(seq 6 7); do
  # 计算 num_edges 的范围 (确保是整数)
  min_edges=$((num_nodes - 1))
  # 使用 bc 进行浮点数计算并向上取整
  max_edges=$(echo "scale=0; ($num_nodes * 1.5 + 0.999)/1" | bc)
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
      # max_extra_edge=$((extra_predicate_num + 5))
      max_extra_edge=$((extra_predicate_num + 3))
      for extra_edge_num in $(seq $min_extra_edge $max_extra_edge); do
        # for max_predicates_per_rule in $(seq 2 8); do
        for max_predicates_per_rule in $(seq 2 5); do
          # 注意：num_noise_rules_per_type 从 5 递减到 0
          for num_noise_rules_per_type in $(seq 3 -1 0); do
            for m in $(seq 0 3); do
              largest=$((m + 1)) # largest 循环现在只有一个值

              # --- 文件存在性检查 (优化) ---
              # 定义 Python 脚本使用的默认概率值
              strong_negation_prob_default=0.5
              default_negation_prob_default=1.0

              # 根据 Python 脚本的 _generate_output_filename 函数构造预期文件名
              # 注意：文件名不包含 num_noise_rules_per_type
              expected_filename="symtex_dataset_nodes${num_nodes}_edges${num_edges}_extraP${extra_predicate_num}_extraE${extra_edge_num}_sNeg${strong_negation_prob_default}_dNeg${default_negation_prob_default}_maxPred${max_predicates_per_rule}_m${m}_l${largest}.jsonl"
              expected_file_path="${AUTO_GEN_DIR}/${expected_filename}"

              # 检查文件是否存在
              if [ -f "$expected_file_path" ]; then
                # echo "Skipping: Output file already exists: $expected_file_path" # 可以取消注释以查看跳过信息
                continue # 跳过当前参数组合
              fi
              # --- 文件存在性检查结束 ---


              # --- 并行控制 ---
              # 如果当前作业数达到最大值，等待一个作业完成
              if [ $job_count -ge $MAX_PROCS ]; then
                  # echo "Max processes ($MAX_PROCS) reached, waiting for one to finish..." # 可以取消注释以查看等待信息
                  wait -n # 等待任意一个后台作业结束
                  job_count=$((job_count - 1)) # 减少计数器
              fi
              # --- 并行控制结束 ---


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

              # echo "Launching background job: python $PYTHON_SCRIPT $args" # 可以取消注释以查看启动信息

              # 先创建标记文件，减少竞争条件
              touch "$expected_file_path"

              # 在后台执行 Python 脚本 (使用子 shell 隔离输出和错误)
              (
                python $PYTHON_SCRIPT $args
                status=$?
                if [ $status -ne 0 ]; then
                  echo "Error (Exit Code: $status) in background job with params: $args" >&2 # 输出错误到 stderr
                  # 可选：记录失败的参数到日志文件
                  # echo "Failed: $args" >> error_log.txt
                  # 可选：如果失败，删除标记文件，以便下次重试
                  # rm -f "$expected_file_path"
                else
                  # echo "Background job completed successfully for params: nodes=$num_nodes, edges=$num_edges, ..." # 可以取消注释以查看成功信息
                  : # No-op, do nothing on success
                fi
              ) & # 放到后台执行

              # 增加作业计数器
              job_count=$((job_count + 1))

              # 可选：添加短暂延迟以避免瞬间启动过多进程冲击系统
              # sleep 0.05

            done # m
          done # num_noise_rules_per_type
        done # max_predicates_per_rule
      done # extra_edge_num
    done # extra_predicate_num
  done # num_edges
done # num_nodes

echo "========================================"
echo "All jobs launched. Waiting for remaining background jobs to complete..."
wait # 等待所有后台作业完成
echo "========================================"
echo "All parameter combinations processed."
echo "Check $AUTO_GEN_DIR for generated datasets."
echo "Check standard error output or error_log.txt (if enabled) for any errors."
