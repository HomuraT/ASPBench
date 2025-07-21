#!/bin/bash



# 获取脚本所在的目录

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"



# Check if a custom common_vars.sh path is provided to this master script

# This path will be passed down to the individual evaluation scripts

CUSTOM_COMMON_VARS_PATH="$1"



echo "=================================================="

echo "开始运行所有评估脚本..."

echo "时间: $(date)"

echo "=================================================="

# 记录总开始时间

total_start_time=$(date +%s)



# 运行 Fact State Querying 评估

echo ">>> 开始运行 Fact State Querying 评估脚本..."

bash "${SCRIPT_DIR}/run_fact_state_querying_evaluation.sh" "$CUSTOM_COMMON_VARS_PATH"

if [ $? -ne 0 ]; then

    echo "!!! Fact State Querying 评估脚本运行失败!"

    # exit 1 # 如果希望某个脚本失败时停止整个流程，取消此行注释

fi

echo "<<< Fact State Querying 评估脚本运行完成."

echo "--------------------------------------------------"



# 运行 Answer Set Generation 评估

echo ">>> 开始运行 Answer Set Generation 评估脚本..."

bash "${SCRIPT_DIR}/run_answer_set_generation_evaluation.sh" "$CUSTOM_COMMON_VARS_PATH"

if [ $? -ne 0 ]; then

    echo "!!! Answer Set Generation 评估脚本运行失败!"

    # exit 1 # 如果希望某个脚本失败时停止整个流程，取消此行注释

fi

echo "<<< Answer Set Generation 评估脚本运行完成."

echo "--------------------------------------------------"



# 运行 Answer Set Decision 评估

echo ">>> 开始运行 Answer Set Decision 评估脚本..."

bash "${SCRIPT_DIR}/run_answer_set_decision_evaluation.sh" "$CUSTOM_COMMON_VARS_PATH"

if [ $? -ne 0 ]; then

    echo "!!! Answer Set Decision 评估脚本运行失败!"

    # exit 1 # 如果希望某个脚本失败时停止整个流程，取消此行注释

fi

echo "<<< Answer Set Decision 评估脚本运行完成."

echo "--------------------------------------------------"





# 计算总运行时间

total_end_time=$(date +%s)

total_duration=$((total_end_time - total_start_time))

total_hours=$((total_duration / 3600))

total_minutes=$(( (total_duration % 3600) / 60 ))

total_seconds=$((total_duration % 60))



echo "=================================================="

echo "所有评估脚本运行完毕!"

echo "总运行时间: ${total_hours}小时 ${total_minutes}分钟 ${total_seconds}秒"

echo "=================================================="