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

# 定义输入文件列表 (symbolic 和 textual)
input_files=(
    "datasets/openkg_subset/openkg_ASC.jsonl"
)

# 定义输出目录 (Python脚本会将生成结果和评估指标放在这里)
OUTPUT_DIR="experiments/openkg/full/"
mkdir -p $OUTPUT_DIR

# 记录开始时间
start_time=$(date +%s)

# 循环处理每个API和输入文件
for api_name in "${api_names[@]}"; do
    # 从 common_vars.sh 获取线程数，如果未指定则使用默认值
    curl -X POST   http://127.0.0.1:5824/v1/stop_all_models

    num_threads=${api_thread_counts[$api_name]:-$default_threads}

    for input_file in "${input_files[@]}"; do # 添加内层循环遍历输入文件
        echo "================================================"
        echo "开始处理: API=$api_name, 输入文件=$input_file, 线程数=$num_threads" # 更新日志信息
        echo "时间: $(date)"

        # 运行Python脚本 (包含生成和评估)
        python 09_03_evaluate_symtex_answer_set_generation.py \
            --input_file "$input_file" \
            --api_name "$api_name" \
            --output_dir "$OUTPUT_DIR" \
            --threads "$num_threads" \
            --save_interval 1 \
            --json_parsing_model_name "$DEFAULT_JSON_PARSING_MODEL_NAME"
            # --save_interval 20 \ # 可选：指定保存间隔，默认为50
            # --skip_evaluation # 可选：如果只想生成不想评估，取消此行注释

        # 检查上一个命令的退出状态

        if [ $? -eq 0 ]; then
            echo "成功完成: API=$api_name, 输入文件=$input_file" # 更新日志信息
        else
            echo "处理失败: API=$api_name, 输入文件=$input_file" # 更新日志信息
        fi
        echo "================================================"
    done # 结束内层循环
done # 结束外层循环



# 计算总运行时间

end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
echo "脚本总运行时间: ${hours}小时 ${minutes}分钟 ${seconds}秒"