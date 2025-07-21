# 03_statistic_raw_data.py

import argparse
from pathlib import Path
import argparse
from pathlib import Path
import sys
# import json # Replaced by orjson
import os # Added for path operations
from datetime import datetime # Added for timestamp
from collections import Counter
import multiprocessing
try:
    import orjson as json # Use orjson if available, faster parser
except ImportError:
    print("Warning: orjson not found, falling back to standard json library. Install orjson for faster parsing.", file=sys.stderr)
    import json
# 导入你的工具库中的函数和错误类
# from src.utils.json_utils import read_jsonl_lazy, JsonUtilsError # We are replacing the loading logic
from tqdm import tqdm

# --- Helper function for parallel parsing ---
def parse_line(line):
    """Parses a single JSON line using orjson, handles errors."""
    try:
        # orjson loads directly from bytes or string
        return json.loads(line)
    except json.JSONDecodeError:
        # Log or handle the error appropriately, here we skip the line
        # print(f"Warning: Skipping invalid JSON line: {line[:100]}...", file=sys.stderr)
        return None
    except Exception as e:
        # print(f"Warning: Unexpected error parsing line: {e}", file=sys.stderr)
        return None

# --- End Plotting Functions --- # Removed plotting functions


def main():
    # Modified description to reflect cleaning purpose
    parser = argparse.ArgumentParser(description="Load a JSONL file, clean duplicates based on specific keys, and save the cleaned data.")
    parser.add_argument("--input_path", type=str, default='datasets/symtex_merged_raw_dataset/2025_04_24_08_54.jsonl', help="Path to the input JSONL file.")
    # Removed --output_dir argument as plots are removed
    args = parser.parse_args()

    input_file_path = Path(args.input_path)
    # Removed base_output_dir as plots are removed

    # --- Get current time for timestamp ---
    now = datetime.now() # Keep for cleaned data timestamp
    # Removed plot directory creation

    # --- Determine number of lines for tqdm --- # Keep this part
    print(f"Counting lines in {input_file_path} for progress bar...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"Found {total_lines} lines.")
    except Exception as e:
        print(f"Warning: Could not count lines accurately ({e}), progress bar might be inaccurate.", file=sys.stderr)
        total_lines = None # Set to None if counting fails

    all_data = [] # Collect data here
    pool = None # Initialize pool to None
    print(f"Reading and parsing data from {input_file_path} using multiprocessing...")
    try:
        # Determine number of processes (leave one core free or use cpu_count())
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_processes} processes.")
        pool = multiprocessing.Pool(processes=num_processes)

        # Open the file again for reading lines
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Use imap_unordered for potentially faster processing when order doesn't matter
            # chunksize helps in reducing overhead by sending batches of lines to workers
            results_iterator = pool.imap_unordered(parse_line, f, chunksize=1000)

            # Process results using tqdm for progress
            all_data = [result for result in tqdm(results_iterator, total=total_lines, desc="Parsing JSONL") if result is not None]

        print(f"Successfully parsed {len(all_data)} entries.")

        if not all_data:
            print("No data found to analyze.")
            sys.exit(0)

        # --- Data Cleaning ---
        print("Starting data cleaning based on structure signature...")
        signature_keys = [
            'num_facts', 'num_rules', 'max_depth_of_rule_graph',
            'average_depth_of_rule_graph', 'max_ary_for_predicates',
            'max_idx_for_variables', 'target_query_in_answerset',
            'max_ary_for_predicates',
            'max_predicates_per_rule'
        ]
        cleaned_data = []
        seen_signatures = set()
        float_precision = 6 # Precision for comparing average_depth_of_rule_graph

        for item in tqdm(all_data, desc="Cleaning Data"):
            try:
                # Extract values safely, using None if a key is missing
                signature_values = [item.get(key, None) for key in signature_keys]

                # Handle the floating point number specifically
                avg_depth_float = item.get('average_depth_of_rule_graph', None)
                if avg_depth_float is not None:
                    # Format float to string with fixed precision for consistent comparison
                    avg_depth_str = f"{avg_depth_float:.{float_precision}f}"
                    # Find the index of average_depth_of_rule_graph and replace the float with the string
                    avg_depth_index = signature_keys.index('average_depth_of_rule_graph')
                    signature_values[avg_depth_index] = avg_depth_str
                else:
                     # If the float value is None, ensure it's represented consistently (e.g., as None)
                    avg_depth_index = signature_keys.index('average_depth_of_rule_graph')
                    signature_values[avg_depth_index] = None # Or another placeholder if needed

                # Create the signature tuple (must be hashable)
                signature = tuple(signature_values)

                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    cleaned_data.append(item) # Append the original item
            except TypeError as e:
                # Handle cases where a value might not be hashable (e.g., a list used accidentally)
                print(f"Warning: Skipping item due to non-hashable value in signature: {item}. Error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Skipping item due to unexpected error during signature creation: {item}. Error: {e}", file=sys.stderr)


        print(f"Cleaning complete. Original items: {len(all_data)}, Cleaned items: {len(cleaned_data)}")

        # --- Save Cleaned Data ---
        clean_data_output_dir = Path("datasets/symtex_merged_clean_dataset")
        try:
            os.makedirs(clean_data_output_dir, exist_ok=True)
            clean_timestamp_str = now.strftime("%Y_%m_%d_%H_%M") # Use the same timestamp as plots for consistency
            clean_output_filename = clean_data_output_dir / f"{clean_timestamp_str}.jsonl"

            print(f"Saving cleaned data to: {clean_output_filename}")
            with open(clean_output_filename, 'wb' if 'orjson' in sys.modules else 'w') as outfile:
                 for entry in tqdm(cleaned_data, desc="Saving Cleaned Data"):
                     if 'orjson' in sys.modules:
                         # orjson expects bytes
                         outfile.write(json.dumps(entry) + b'\n')
                     else:
                         # standard json expects string
                         outfile.write(json.dumps(entry) + '\n')
            print("Cleaned data saved successfully.")

        except Exception as e:
            print(f"Error saving cleaned data to {clean_output_filename}: {e}", file=sys.stderr)
            # Decide if script should exit or just warn
            # sys.exit(1) # Optional: Exit if saving clean data is critical

        # --- Removed DataFrame Conversion, Feature Extraction, and Plotting ---
        print("Script finished after cleaning and saving data.")

    except FileNotFoundError as e: # Handle file not found during initial open or count
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr) # Modified error message slightly
        sys.exit(1)
    # Removed JsonUtilsError as we are not using the util function anymore for reading
    # Keep JSONDecodeError in case parse_line re-raises it or for non-parallel fallback
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}", file=sys.stderr) # Might occur if fallback json is used
        sys.exit(1)
    except Exception as e: # Catch other potential errors during processing
        print(f"An unexpected error occurred during processing: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Ensure the pool is always closed and joined
        if pool:
            pool.close()
            pool.join()
            print("Process pool closed.")


if __name__ == "__main__":
    # Add freeze_support() for Windows compatibility if creating executables
    # multiprocessing.freeze_support()
    main()
