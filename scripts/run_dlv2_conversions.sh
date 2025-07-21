#!/bin/bash

# === Configuration ===
INPUT_DATA_DIR="datasets/symtex_filter_from_clean_data"
PYTHON_SCRIPT="06_transfer_dict_to_dlv2.py"
BASE_OUTPUT_DIR="datasets/symtex_dlv2"
OUTPUT_SUBDIR_W_DISJUNCTION="w_disjunction"
OUTPUT_SUBDIR_WO_DISJUNCTION="wo_disjunction"
# ConceptNet graph path (assuming default from 06_transfer_dict_to_dlv2.py if needed)
# CONCEPTNET_GRAPH_PATH="datasets/conceptnet/bird_graph.graphml" # Optional: uncomment and adjust if needed

# === Find Input File ===
echo "Searching for the first JSONL file in: $INPUT_DATA_DIR"
INPUT_FILE=$(find "$INPUT_DATA_DIR" -maxdepth 1 -name '*.jsonl' -print -quit)

if [ -z "$INPUT_FILE" ]; then
  echo "Error: No .jsonl file found in $INPUT_DATA_DIR"
  exit 1
elif [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Found path is not a valid file: $INPUT_FILE"
  exit 1
fi
echo "Found input file: $INPUT_FILE"

# === Create Output Directories ===
OUTPUT_DIR_W_DIS="$BASE_OUTPUT_DIR/$OUTPUT_SUBDIR_W_DISJUNCTION"
OUTPUT_DIR_WO_DIS="$BASE_OUTPUT_DIR/$OUTPUT_SUBDIR_WO_DISJUNCTION"

echo "Creating output directories (if they don't exist):"
echo " - $OUTPUT_DIR_W_DIS"
echo " - $OUTPUT_DIR_WO_DIS"
mkdir -p "$OUTPUT_DIR_W_DIS"
mkdir -p "$OUTPUT_DIR_WO_DIS"
echo "Output directories created."
echo "========================================"

# === Helper Function to Run Python Script ===
# $1: Output file path
# $2: Additional flags for the python script (e.g., "--use_disjunction")
run_conversion() {
  local output_path="$1"
  local flags="$2"
  local timestamp=$(date +"%Y_%m_%d_%H_%M") # Generate timestamp inside function for each run

  # Construct the final output path with timestamp (adjust filename based on flags)
  local filename_prefix="P_style" # Default
  if [[ "$flags" == *"--use_random_conceptnet_predicates"* ]]; then
    filename_prefix="random_word"
  elif [[ "$flags" == *"--use_related_conceptnet_predicates"* ]]; then
    filename_prefix="related_word"
  fi

  local final_output_path="${output_path}/${filename_prefix}_${timestamp}.jsonl"

  echo "Running conversion:"
  echo "  Input: $INPUT_FILE"
  echo "  Output: $final_output_path"
  echo "  Flags: $flags"
  echo "----------------------------------------"

  # Execute Python script
  python "$PYTHON_SCRIPT" --input_path "$INPUT_FILE" --output_path "$final_output_path" $flags

  # Check exit status
  local status=$?
  if [ $status -ne 0 ]; then
    echo "----------------------------------------"
    echo "Error (Exit Code: $status) running Python script with flags: $flags"
    echo "Output was intended for: $final_output_path"
    # Decide if you want to exit the whole script on error or continue
    # exit $status # Uncomment to exit on first error
  else
    echo "----------------------------------------"
    echo "Python script completed successfully for: $final_output_path"
  fi
  echo "========================================"
}

# === Execute the 6 Combinations ===

# --- Without Disjunction ---
echo "*** Running conversions WITHOUT disjunction (output to $OUTPUT_DIR_WO_DIS) ***"
# 1. Default predicates, no disjunction
run_conversion "$OUTPUT_DIR_WO_DIS" "--check_query_in_answer"
# 2. Random predicates, no disjunction
run_conversion "$OUTPUT_DIR_WO_DIS" "--use_random_conceptnet_predicates --check_query_in_answer"
# 3. Related predicates, no disjunction
run_conversion "$OUTPUT_DIR_WO_DIS" "--use_related_conceptnet_predicates --check_query_in_answer"

# --- With Disjunction ---
echo "*** Running conversions WITH disjunction (output to $OUTPUT_DIR_W_DIS) ***"
# 4. Default predicates, with disjunction
run_conversion "$OUTPUT_DIR_W_DIS" "--use_disjunction --check_query_in_answer"
# 5. Random predicates, with disjunction
run_conversion "$OUTPUT_DIR_W_DIS" "--use_disjunction --use_random_conceptnet_predicates --check_query_in_answer"
# 6. Related predicates, with disjunction
run_conversion "$OUTPUT_DIR_W_DIS" "--use_disjunction --use_related_conceptnet_predicates --check_query_in_answer"

echo "All conversion runs attempted."
exit 0
