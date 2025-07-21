#!/bin/bash

# run_construct_symtex.sh
# Executes the 07_04_construct_symtex.py script for multiple input directories.

# --- Configuration ---
PYTHON_EXECUTABLE="python" # Or specify the full path if needed, e.g., "/usr/bin/python3"
SCRIPT_TO_RUN="./07_04_construct_symtex.py"
OUTPUT_DIRECTORY="./datasets/symtex_final" # Output directory used by the python script

# Input directories to process
INPUT_DIRECTORIES=(
    "./datasets/a_symtex_task_answerset_generation"
    "./datasets/a_symtex_task_answerset_selection"
    "./datasets/a_symtex_task_fact_state_querying"
)

# --- Execution ---
echo "Starting SymTex construction process..."

# Ensure the output directory exists (optional, as the python script might handle this)
if [ ! -d "$OUTPUT_DIRECTORY" ]; then
    echo "Creating output directory: $OUTPUT_DIRECTORY"
    mkdir -p "$OUTPUT_DIRECTORY"
fi

# Loop through each input directory
for input_dir in "${INPUT_DIRECTORIES[@]}"; do
    echo "--------------------------------------------------"
    echo "Processing input directory: $input_dir"
    echo "--------------------------------------------------"

    # Construct the command
    command="$PYTHON_EXECUTABLE $SCRIPT_TO_RUN --input_dir \"$input_dir\" --output_dir \"$OUTPUT_DIRECTORY\""

    # Execute the command
    echo "Executing: $command"
    eval $command # Using eval to handle potential spaces in paths correctly if quoted

    # Check the exit status of the python script
    if [ $? -ne 0 ]; then
        echo "Error processing $input_dir. Exiting."
        exit 1 # Exit the script if any command fails
    else
        echo "Successfully processed $input_dir."
    fi
    echo "" # Add a blank line for readability
done

echo "=================================================="
echo "SymTex construction script finished successfully."
echo "=================================================="

exit 0
