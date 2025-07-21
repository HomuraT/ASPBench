#!/bin/bash

# 定义包含原始合并数据集的目录
DATA_DIR="datasets/symtex_merged_raw_dataset"
# 定义要执行的 Python 清理脚本
PYTHON_SCRIPT="04_clean_raw_data.py"

echo "Searching for the first JSONL file in: $DATA_DIR"

# 查找目录中按字母顺序排列的第一个 .jsonl 文件
# -maxdepth 1 确保只搜索当前目录，不搜索子目录
# -name '*.jsonl' 匹配文件名
# -print 打印找到的文件路径
# -quit 找到第一个匹配项后立即退出 find 命令
INPUT_FILE=$(find "$DATA_DIR" -maxdepth 1 -name '*.jsonl' -print -quit)

# 检查是否找到了文件
if [ -z "$INPUT_FILE" ]; then
  echo "Error: No .jsonl file found in $DATA_DIR"
  exit 1
elif [ ! -f "$INPUT_FILE" ]; then
  # 额外的检查，确保找到的是一个文件而不是目录或其他
  echo "Error: Found path is not a valid file: $INPUT_FILE"
  exit 1
fi

echo "Found input file: $INPUT_FILE"
echo "Executing Python script: $PYTHON_SCRIPT"
echo "========================================"

# 执行 Python 脚本，并将找到的文件路径作为参数传递
python "$PYTHON_SCRIPT" --input_path "$INPUT_FILE"

# 检查 Python 脚本的退出状态
status=$?
if [ $status -ne 0 ]; then
  echo "========================================"
  echo "Error (Exit Code: $status) running Python script: $PYTHON_SCRIPT"
  exit $status
else
  echo "========================================"
  echo "Python script completed successfully."
fi

exit 0
