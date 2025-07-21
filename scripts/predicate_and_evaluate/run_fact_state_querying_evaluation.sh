#!/bin/bash

# Check if a custom common_vars.sh path is provided
if [ -n "$1" ]; then
    COMMON_VARS_PATH="$1"
else
    COMMON_VARS_PATH="$(dirname "$0")/common_vars.sh"
fi

# Source common variables
. "$COMMON_VARS_PATH"

# 设置日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Define API names
# api_names=(
#     "local_qwen3_8b"
#     "local_qwen2_7b"
#     "local_qwen1_5b"
#     "local_llama2_7b"
#     "local_llama2_13b"
# )

# 定义输入文件列表
input_files=(
    "datasets/SymTex/fact_state_querying_textual.jsonl"
    "datasets/SymTex/fact_state_querying_symbolic.jsonl"
)

# 输出目录
OUTPUT_DIR="experiments/fact_state_querying/w_few_shot"
mkdir -p $OUTPUT_DIR

# 记录开始时间
start_time=$(date +%s)

# 循环处理每个API和输入文件
for api_name in "${api_names[@]}"; do
    # 从 common_vars.sh 获取线程数，如果未指定则使用默认值
    num_threads=${api_thread_counts[$api_name]:-$default_threads}

    for input_file in "${input_files[@]}"; do
        echo "================================================"
        echo "开始处理: API=$api_name, 输入文件=$input_file, 线程数=$num_threads"
        echo "时间: $(date)"
        
        # 运行Python脚本
        python 09_01_evaluate_symtex_fact_state_querying.py \
            --input_file "$input_file" \
            --api_name "$api_name" \
            --output_dir "$OUTPUT_DIR" \
            --threads "$num_threads" \
            --save_interval 1 \
            --json_parsing_model_name "$DEFAULT_JSON_PARSING_MODEL_NAME"
        
        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "成功完成: API=$api_name, 输入文件=$input_file"
        else
            echo "处理失败: API=$api_name, 输入文件=$input_file"
        fi
        
        echo "================================================"
    done
done

# 计算总运行时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "总运行时间: ${hours}小时 ${minutes}分钟 ${seconds}秒" 