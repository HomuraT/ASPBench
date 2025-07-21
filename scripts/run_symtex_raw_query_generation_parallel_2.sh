#!/bin/bash

# --- 配置 ---
# 脚本和路径
PYTHON_SCRIPT="01_raw_symtex_generation.py"
BASE_OUTPUT_DIR="datasets/symtex_batch_runs"
AUTO_GEN_DIR="${BASE_OUTPUT_DIR}/auto_generated_files"
LOG_DIR="./logs/generate_raw_data"

# 并行控制
MAX_PROCS=128

# 参数范围 (循环)
NUM_NODES_START=1
NUM_NODES_END=10
# num_edges 范围是动态计算的，基于 num_nodes (min=N-1, max=ceil(N*1.5))
# extra_predicate_num 范围是动态计算的，基于 num_edges (1 to num_edges)
# extra_edge_num 范围是动态计算的，基于 extra_predicate_num (min=extra_pred, max=extra_pred+EXTRA_EDGE_OFFSET)
EXTRA_EDGE_OFFSET=3 # max_extra_edge = extra_predicate_num + EXTRA_EDGE_OFFSET
M_START=0
M_END=3
# largest 范围是动态计算的，基于 m (m+1)

# 固定参数值
MAX_PREDICATES_PER_RULE=3
NUM_NOISE_RULES_PER_TYPE=2

# 文件检查相关默认值 (与 Python 脚本中的默认值匹配)
STRONG_NEGATION_PROB_DEFAULT=0.5
DEFAULT_NEGATION_PROB_DEFAULT=1.0

# --- 配置结束 ---

# --- 初始化 ---
# 确保输出和日志目录存在
mkdir -p "$AUTO_GEN_DIR"
mkdir -p "$LOG_DIR"

# 定义日志文件路径
LOG_FILE="$LOG_DIR/$(date +'%Y_%m_%d_%H_%M').log"
touch "$LOG_FILE" # 创建空日志文件以确认权限

echo "Starting SymTex generation batch run in parallel (Max Procs: $MAX_PROCS)..."
echo "Python output will be logged to: $LOG_FILE"
echo "Output files will be saved in subdirectories under: $AUTO_GEN_DIR"
echo "========================================"

# --- 计算总任务数 ---
echo "Calculating total number of tasks..."
total_iterations=0
for num_nodes in $(seq $NUM_NODES_START $NUM_NODES_END); do
  min_edges=$((num_nodes - 1))
  max_edges=$(echo "scale=0; ($num_nodes * 1.5 + 0.999)/1" | bc) # 保持动态计算
  if [ $min_edges -lt 0 ]; then min_edges=0; fi
  if [ $max_edges -lt $min_edges ]; then max_edges=$min_edges; fi
  if [ $min_edges -gt $max_edges ]; then continue; fi

  for num_edges in $(seq $min_edges $max_edges); do
    for extra_predicate_num in $(seq 1 $num_edges); do # 保持动态计算
      min_extra_edge=$extra_predicate_num
      max_extra_edge=$((extra_predicate_num + EXTRA_EDGE_OFFSET)) # 使用配置变量
      for extra_edge_num in $(seq $min_extra_edge $max_extra_edge); do
        # 使用配置中的固定值
        max_predicates_per_rule=$MAX_PREDICATES_PER_RULE
        num_noise_rules_per_type=$NUM_NOISE_RULES_PER_TYPE
        for m in $(seq $M_START $M_END); do # 使用配置变量
          largest=$((m + 1)) # 保持动态计算
          total_iterations=$((total_iterations + 1))
        done # m
      done # extra_edge_num
    done # extra_predicate_num
  done # num_edges
done # num_nodes
echo "Total tasks to process (including potential skips): $total_iterations"
echo "========================================"
# --- 计算结束 ---


# --- 时间格式化函数 ---
format_time() {
  local total_seconds=$1
  if [[ $total_seconds -lt 0 ]]; then
      echo "N/A"
      return
  fi
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" $hours $minutes $seconds
}
# --- 时间格式化函数结束 ---


# --- 进度条函数 ---
# 参数: current_count, total_count, start_time_seconds
update_progress() {
    local current=$1
    local total=$2
    local start_time=$3
    local current_time=$(date +%s)
    local elapsed_seconds=$((current_time - start_time))
    local percentage=0
    local eta_seconds=-1 # 默认为 N/A

    if [ $total -gt 0 ]; then
        percentage=$((current * 100 / total))
    fi

    if [ $current -gt 0 ] && [ $elapsed_seconds -gt 0 ]; then
        # 计算 ETA
        local avg_time_per_iter=$(echo "scale=4; $elapsed_seconds / $current" | bc)
        local remaining_iterations=$((total - current))
        eta_seconds=$(echo "scale=0; $avg_time_per_iter * $remaining_iterations / 1" | bc)
    fi

    local elapsed_formatted=$(format_time $elapsed_seconds)
    local eta_formatted=$(format_time $eta_seconds)

    local progress_bar_width=40 # 稍微缩短以容纳时间信息
    local completed_width=0
    if [ $total -gt 0 ]; then
        completed_width=$((percentage * progress_bar_width / 100))
    fi
    local remaining_width=$((progress_bar_width - completed_width))

    # 构建进度条字符串
    local progress_string="["
    for ((i=0; i<completed_width; i++)); do progress_string+="="; done
    if [ $completed_width -lt $progress_bar_width ] && [ $total -gt 0 ]; then progress_string+=">"; fi
    for ((i=0; i<remaining_width; i++)); do progress_string+=" "; done
    progress_string+="]"

    # 使用 \r 回到行首并打印，-ne 禁止换行
    echo -ne "Progress: $progress_string ${percentage}% ($current/$total) | Elapsed: $elapsed_formatted | ETA: $eta_formatted \r"
}
# --- 进度条函数结束 ---


# --- 主处理循环 ---
job_count=0
processed_iterations=0
start_time_main_loop=$(date +%s) # 记录主循环开始时间

# 循环遍历参数范围
for num_nodes in $(seq $NUM_NODES_START $NUM_NODES_END); do # 使用配置变量
  # 计算 num_edges 的范围 (确保是整数)
  min_edges=$((num_nodes - 1))
  # 使用 bc 进行浮点数计算并向上取整
  max_edges=$(echo "scale=0; ($num_nodes * 1.5 + 0.999)/1" | bc) # 保持动态计算
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
    for extra_predicate_num in $(seq 1 $num_edges); do # 保持动态计算
      min_extra_edge=$extra_predicate_num
      # max_extra_edge=$((extra_predicate_num + 5)) # 旧注释
      max_extra_edge=$((extra_predicate_num + EXTRA_EDGE_OFFSET)) # 使用配置变量
      for extra_edge_num in $(seq $min_extra_edge $max_extra_edge); do
        # 使用配置中的固定值
        max_predicates_per_rule=$MAX_PREDICATES_PER_RULE
        num_noise_rules_per_type=$NUM_NOISE_RULES_PER_TYPE
        # for max_predicates_per_rule in $(seq 2 5); do # 已移除循环
          # for num_noise_rules_per_type in $(seq 3 -1 0); do # 已移除循环
            for m in $(seq $M_START $M_END); do # 使用配置变量
              largest=$((m + 1)) # largest 循环现在只有一个值, 保持动态计算

              # --- 文件存在性检查 (优化) ---
              # 使用配置中定义的默认概率值
              strong_negation_prob_default=$STRONG_NEGATION_PROB_DEFAULT
              default_negation_prob_default=$DEFAULT_NEGATION_PROB_DEFAULT

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

              # 更新进度 (在启动任务前或文件检查后更新，确保即使跳过也计数)
              processed_iterations=$((processed_iterations + 1))
              update_progress $processed_iterations $total_iterations $start_time_main_loop

              # --- 文件存在性检查结束 --- <<<<< Redundant check removed


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

              # 在后台执行 Python 脚本，并将输出重定向到日志文件
              (
                # 执行 Python 脚本，追加 stdout 和 stderr 到日志文件
                python $PYTHON_SCRIPT $args >> "$LOG_FILE" 2>&1
                status=$?
                # 脚本执行后，无论成功与否或是否生成文件，都创建标记文件以跳过此参数组合
                touch "$expected_file_path"
                if [ $status -ne 0 ]; then
                  # 仅在主脚本的 stderr 中报告错误发生，详细信息在日志文件中
                  echo "Error (Exit Code: $status) occurred for params: nodes=$num_nodes, edges=$num_edges, ... Check log: $LOG_FILE" >&2
                  # 可选：如果失败，删除标记文件，以便下次重试
                  # rm -f "$expected_file_path"
                fi
                # 成功时不再打印信息到控制台
              ) & # 放到后台执行

              # 增加作业计数器
              job_count=$((job_count + 1))

              # 可选：添加短暂延迟以避免瞬间启动过多进程冲击系统
              # sleep 0.05

            done # m
          # done # num_noise_rules_per_type # 已移除循环
        # done # max_predicates_per_rule # 已移除循环
      done # extra_edge_num
    done # extra_predicate_num
  done # num_edges
done # num_nodes

# 确保进度条后的输出在新行
echo ""
echo "========================================"
echo "All jobs launched. Waiting for remaining background jobs to complete..."
wait # 等待所有后台作业完成
echo "========================================"
echo "All parameter combinations processed."
echo "Check $AUTO_GEN_DIR for generated datasets."
echo "Check Python output and errors in the log file: $LOG_FILE"
echo "Check this script's standard error output for any job launch errors."
