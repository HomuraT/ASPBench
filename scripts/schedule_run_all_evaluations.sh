#!/bin/bash

# 脚本：在下一个凌晨 00:30 执行指定的命令

# --- 配置 ---
# 目标脚本的相对路径，相对于此调度脚本所在的目录。
# 假设此调度脚本 (schedule_run_all_evaluations.sh) 和 predicate_and_evaluate 目录
# 都在同一个父目录下 (例如，都在 'scripts/' 目录下)。
TARGET_SCRIPT_RELATIVE_PATH="predicate_and_evaluate/run_all_evaluations.sh"

# 目标执行时间
TARGET_HOUR=0  # 0点 (午夜)
TARGET_MINUTE=30 # 30分

# --- 脚本主体 ---

# 获取此调度脚本所在的目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TARGET_COMMAND="bash \"${SCRIPT_DIR}/${TARGET_SCRIPT_RELATIVE_PATH}\"" # Corrected quoting for eval

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 调度脚本启动 (${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}"))。"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目标命令: ${TARGET_COMMAND}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目标执行时间: ${TARGET_HOUR}点${TARGET_MINUTE}分"

current_epoch=$(date +%s)
target_today_epoch=$(date -d "$(date +%Y-%m-%d) ${TARGET_HOUR}:${TARGET_MINUTE}:00" +%s)

if [ "${current_epoch}" -ge "${target_today_epoch}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 当前时间已过今日 ${TARGET_HOUR}:${TARGET_MINUTE}。将任务安排在明天。"
    target_epoch=$(date -d "tomorrow ${TARGET_HOUR}:${TARGET_MINUTE}:00" +%s)
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 任务将安排在今日 ${TARGET_HOUR}:${TARGET_MINUTE}。"
    target_epoch="${target_today_epoch}"
fi

sleep_seconds=$((target_epoch - current_epoch))

if [ "${sleep_seconds}" -lt 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误：计算出的等待时间为负 (${sleep_seconds}s)。将重新计算为明天 ${TARGET_HOUR}:${TARGET_MINUTE}。"
    target_epoch=$(date -d "tomorrow ${TARGET_HOUR}:${TARGET_MINUTE}:00" +%s)
    sleep_seconds=$((target_epoch - current_epoch))
    if [ "${sleep_seconds}" -lt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 错误：重新计算后等待时间仍为负。请检查系统时间和脚本逻辑。"
        exit 1
    fi
fi

sleep_hours=$((${sleep_seconds} / 3600))
sleep_minutes=$((${sleep_seconds} % 3600 / 60))
sleep_remaining_seconds=$((${sleep_seconds} % 60))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 将等待 ${sleep_seconds} 秒 (即 ${sleep_hours} 小时 ${sleep_minutes} 分钟 ${sleep_remaining_seconds} 秒)."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预计执行时间: $(date -d "@${target_epoch}" '+%Y-%m-%d %H:%M:%S')"

sleep "${sleep_seconds}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 等待结束。现在执行命令。"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 执行: ${TARGET_COMMAND}"

# 执行命令
# eval 用于正确处理路径中的空格等特殊字符
eval "${TARGET_COMMAND}"
COMMAND_EXIT_CODE=$?

if [ ${COMMAND_EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 命令成功执行完毕。"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 命令执行失败，退出码: ${COMMAND_EXIT_CODE}。"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 调度脚本执行结束。"
exit ${COMMAND_EXIT_CODE} 