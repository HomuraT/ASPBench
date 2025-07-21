#!/bin/bash

# 定义 Python 脚本的路径 (假设此 bash 脚本从项目根目录运行)
PYTHON_SCRIPT="02_marge_raw_symtex.py"

# 定义默认的输入目录和工作线程数
DEFAULT_INPUT_DIR="datasets/symtex_batch_runs/auto_generated_files"
DEFAULT_NUM_WORKERS=4

# 允许通过命令行参数覆盖默认值 (可选)
INPUT_DIR=${1:-$DEFAULT_INPUT_DIR}
NUM_WORKERS=${2:-$DEFAULT_NUM_WORKERS}

echo "Starting JSONL merge process..."
echo "Input Directory: $INPUT_DIR"
echo "Number of Workers: $NUM_WORKERS"
echo "========================================"

# 构建传递给 Python 脚本的参数
args="--input_dir $INPUT_DIR \
      --num_workers $NUM_WORKERS"

# 执行 Python 脚本
python $PYTHON_SCRIPT $args

# 检查 Python 脚本的退出状态
status=$?
if [ $status -ne 0 ]; then
  echo "Error (Exit Code: $status) running merge script."
  exit $status
else
  echo "========================================"
  echo "Merge script completed successfully."
fi

exit 0
