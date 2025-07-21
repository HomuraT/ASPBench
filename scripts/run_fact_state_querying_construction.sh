#!/bin/bash

# === Configuration ===
PYTHON_SCRIPT="07_01_fact_state_querying_dataset_construction.py"
INPUT_DATA_DIR="datasets/symtex_dlv2/wo_disjunction"
# Use the default output directory from the python script
OUTPUT_DIR="datasets/a_symtex_task_fact_state_querying"

# === Validate Python Script ===
if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script not found: $PYTHON_SCRIPT"
  exit 1
fi
echo "Using Python script: $PYTHON_SCRIPT"

# === Validate Input Directory ===
if [ ! -d "$INPUT_DATA_DIR" ]; then
  echo "Error: Input data directory not found: $INPUT_DATA_DIR"
  exit 1
fi
echo "Input data directory: $INPUT_DATA_DIR"

# === Create Output Directory ===
echo "Creating output directory (if it doesn't exist): $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "Output directory ensured."
echo "========================================"

# === Process Each JSONL File ===
echo "*** Processing JSONL files in $INPUT_DATA_DIR ***"

# Find all .jsonl files in the input directory (handles spaces in names)
find "$INPUT_DATA_DIR" -maxdepth 1 -name '*.jsonl' -print0 | while IFS= read -r -d $'\0' input_file; do
  if [ -f "$input_file" ]; then
    echo "Processing input file: $input_file"
    echo "----------------------------------------"

    # Execute Python script
    # Pass the output directory explicitly
    python "$PYTHON_SCRIPT" --input_path "$input_file" --output_dir "$OUTPUT_DIR"

    # Check exit status
    status=$? # Capture status immediately after execution
    if [ $status -ne 0 ]; then
      echo "----------------------------------------"
      echo "Error (Exit Code: $status) running Python script for: $input_file"
      # Decide if you want to exit the whole script on error or continue
      # exit $status # Uncomment to exit on first error
    else
      echo "----------------------------------------"
      echo "Python script completed successfully for: $input_file"
    fi
    echo "========================================"
  else
     echo "Warning: Found path is not a valid file, skipping: $input_file"
     echo "========================================"
  fi
done

echo "All processing runs attempted."
exit 0
