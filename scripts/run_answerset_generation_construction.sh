#!/bin/bash

# === Configuration ===
# Updated Python script name for the simplified version (no incorrect set generation)
PYTHON_SCRIPT="07_03_answerset_generation_dataset_construction.py"
# Define the two input data directories
INPUT_DIR_W="datasets/symtex_dlv2/w_disjunction"
INPUT_DIR_WO="datasets/symtex_dlv2/wo_disjunction"
# Define the output directory for this specific task
OUTPUT_DIR="datasets/a_symtex_task_answerset_generation"

# === Validate Python Script ===
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT"
  exit 1
fi
echo "Using Python script: $PYTHON_SCRIPT"

# === Validate Input Directories ===
if [ ! -d "$INPUT_DIR_W" ]; then
  echo "Error: Input data directory (w_disjunction) not found: $INPUT_DIR_W"
  exit 1
fi
if [ ! -d "$INPUT_DIR_WO" ]; then
  echo "Error: Input data directory (wo_disjunction) not found: $INPUT_DIR_WO"
  exit 1
fi
echo "Input data directory (w_disjunction): $INPUT_DIR_W"
echo "Input data directory (wo_disjunction): $INPUT_DIR_WO"

# === Create Output Directory ===
echo "Creating output directory (if it doesn't exist): $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "Output directory ensured."
echo "========================================"

# === Process Each Pair of JSONL Files ===
echo "*** Processing JSONL file pairs from $INPUT_DIR_W and $INPUT_DIR_WO ***"

# Find all .jsonl files in the w_disjunction directory
find "$INPUT_DIR_W" -maxdepth 1 -name '*.jsonl' -print0 | while IFS= read -r -d $'\0' input_file_w; do
  if [ -f "$input_file_w" ]; then
    # Extract the filename prefix (e.g., P_style, random_word, related_word)
    filename_w=$(basename "$input_file_w")
    prefix=""
    if [[ "$filename_w" == P_style* ]]; then
      prefix="P_style"
    elif [[ "$filename_w" == random_word* ]]; then
      prefix="random_word"
    elif [[ "$filename_w" == related_word* ]]; then
      prefix="related_word"
    else
      echo "Warning: Unrecognized prefix for file $filename_w. Skipping."
      echo "========================================"
      continue # Skip to the next file in the loop
    fi

    # Find the corresponding file in the wo_disjunction directory based on the prefix
    # Use find -print -quit to get the first match only
    input_file_wo=$(find "$INPUT_DIR_WO" -maxdepth 1 -name "${prefix}*.jsonl" -print -quit)

    echo "Attempting to process pair based on prefix '$prefix':"
    echo "  w_disj:  $input_file_w"
    # Check if a corresponding file was found
    if [ -n "$input_file_wo" ] && [ -f "$input_file_wo" ]; then
      echo "  wo_disj: $input_file_wo (Found)"
      echo "----------------------------------------"
      echo "Executing Python script..."

      # Execute Python script with both input paths
      python "$PYTHON_SCRIPT" --input_path_w_disj "$input_file_w" --input_path_wo_disj "$input_file_wo" --output_dir "$OUTPUT_DIR"

      # Check exit status
      status=$? # Capture status immediately after execution
      if [ $status -ne 0 ]; then
        echo "----------------------------------------"
        echo "Error (Exit Code: $status) running Python script for pair:"
        echo "  w_disj:  $input_file_w"
        echo "  wo_disj: $input_file_wo"
        # Decide if you want to exit the whole script on error or continue
        # exit $status # Uncomment to exit on first error
      else
        echo "----------------------------------------"
        echo "Python script completed successfully for pair:"
        echo "  w_disj:  $input_file_w"
        echo "  wo_disj: $input_file_wo"
      fi
    else
      # If input_file_wo is empty or not a file
      echo "  wo_disj: (Not Found)"
      echo "----------------------------------------"
      echo "Warning: Corresponding file with prefix '$prefix' not found in $INPUT_DIR_WO. Skipping pair."
    fi
    echo "========================================"
  else
     echo "Warning: Found path in $INPUT_DIR_W is not a valid file, skipping: $input_file_w"
     echo "========================================"
  fi
done

echo "All processing runs for file pairs attempted."
exit 0
