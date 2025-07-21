# 03_statistic_raw_data.py

import argparse
from pathlib import Path
import argparse
from pathlib import Path
import sys
# import json # Replaced by orjson
import os
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# --- Plotting Functions ---
# (Keep existing plotting functions as they are)
def plot_numerical_distributions(df: pd.DataFrame, save_dir: Path):
    """Generates and saves histograms for numerical columns."""
    print(f"Plotting numerical distributions to {save_dir}...")
    numerical_cols = [
        'average_depth_of_rule_graph', 'max_depth_of_rule_graph',
        'max_ary_for_predicates', 'max_idx_for_variables',
        'max_predicates_per_rule', 'num_facts', 'num_noiseless_rules',
        'num_related_predicates', #'num_rules_with_default_negation', # Replaced by ratio
        'num_noisy_rules',
        'ratio_default_negation_rules', # Added ratio plot
        'num_min_facts_for_query' # Added min facts count
    ]

    # Filter out columns that might not have been extracted successfully (all NaN)
    valid_cols = [col for col in numerical_cols if col in df.columns and not df[col].isnull().all()]

    if not valid_cols:
        print("No valid numerical columns found to plot.")
        return

    for col in tqdm(valid_cols, desc="Plotting Numerical"):
        plt.figure(figsize=(10, 6))

        # --- Filter data for num_min_facts_for_query ---
        if col == 'num_min_facts_for_query':
            # Only plot for samples where the target query is in the answerset
            plot_data = df.loc[df['target_query_in_answerset'] == True, col].dropna()
            if plot_data.empty:
                print(f"Warning: No data found for {col} where target_query_in_answerset is True. Skipping plot.", file=sys.stderr)
                plt.close()
                continue # Skip to the next column
        else:
            # For other columns, use all available data
            plot_data = df[col].dropna()
            if plot_data.empty:
                print(f"Warning: No data found for {col} after dropping NaNs. Skipping plot.", file=sys.stderr)
                plt.close()
                continue # Skip to the next column

        # Use the potentially filtered plot_data for the histogram
        sns.histplot(plot_data, kde=True)
        # --- End filter ---

        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        save_path = save_dir / f"distribution_{col}.png"
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"Warning: Failed to save plot {save_path}: {e}", file=sys.stderr)
        plt.close() # Close the figure to free memory

def plot_categorical_distributions(df: pd.DataFrame, save_dir: Path):
    """Generates and saves count plots for categorical columns."""
    print(f"Plotting categorical distributions to {save_dir}...")
    categorical_cols = [
        'target_in_answer', 'target_query_strong_negation',
        'target_query_predicateIdx'
    ]

    valid_cols = [col for col in categorical_cols if col in df.columns and not df[col].isnull().all()]

    if not valid_cols:
        print("No valid categorical columns found to plot.")
        return

    for col in tqdm(valid_cols, desc="Plotting Categorical"):
        plt.figure(figsize=(10, 6))
        # Use countplot for categorical data
        # Order by index (category value) if it makes sense (like for predicateIdx)
        order = sorted(df[col].dropna().unique()) if col == 'target_query_predicateIdx' else None
        sns.countplot(x=df[col].dropna(), order=order)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        # Rotate x-axis labels if there are many categories or they are long
        if df[col].nunique() > 10:
             plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        save_path = save_dir / f"countplot_{col}.png"
        try:
            plt.savefig(save_path)
        except Exception as e:
            print(f"Warning: Failed to save plot {save_path}: {e}", file=sys.stderr)
        plt.close()

def plot_noisy_rule_types(df: pd.DataFrame, save_dir: Path):
    """Counts and plots the distribution of noisy rule types ('a', 'b', etc.)."""
    print(f"Plotting noisy rule type distributions to {save_dir}...")

    noisy_rule_type_counts = Counter()

    # Define the safe_get function locally or ensure it's accessible
    def safe_get(data, keys, default=None):
        """Safely get nested dictionary values."""
        if not isinstance(data, dict):
            return default
        temp = data
        for key in keys:
            if isinstance(temp, dict) and key in temp:
                temp = temp[key]
            else:
                return default
        return temp

    # Iterate through the DataFrame to count noisy rule types
    for dict_structure in tqdm(df['dict_structure'], desc="Counting Noisy Rules"):
        noisy_rules = safe_get(dict_structure, ['noisy_rules'], {})
        if isinstance(noisy_rules, dict):
            for rule_type, rules_list in noisy_rules.items():
                 # Count the number of rules for this type in this sample
                 if isinstance(rules_list, list):
                     noisy_rule_type_counts[rule_type] += len(rules_list)

    if not noisy_rule_type_counts:
        print("No noisy rule data found to plot.")
        return

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame.from_dict(noisy_rule_type_counts, orient='index', columns=['count'])
    plot_data = plot_data.sort_index() # Sort by rule type ('a', 'b', ...)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=plot_data.index, y=plot_data['count'])
    plt.title('Total Count of Each Noisy Rule Type')
    plt.xlabel('Noisy Rule Type')
    plt.ylabel('Total Count')
    plt.tight_layout()
    save_path = save_dir / "countplot_noisy_rule_types.png"
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Warning: Failed to save plot {save_path}: {e}", file=sys.stderr)
    plt.close()

# --- End Plotting Functions ---


def main():
    parser = argparse.ArgumentParser(description="Load a JSONL file, perform statistics, and save plots.")

    # data1/renlin/pypro/SymTex/datasets/symtex_merged_clean_dataset/2025_04_19_14_39.jsonl
    # parser.add_argument("--input_path", type=str, default='datasets/symtex_merged_raw_dataset/2025_04_19_14_13.jsonl', help="Path to the input JSONL file.")
    parser.add_argument("--input_path", type=str, default='/data1/renlin/pypro/SymTex/datasets/symtex_filter_from_clean_data/20250424_092614_from_2025_04_24_09_18_seed42_n1000_minprob0.5.jsonl', help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", type=str, default='experiments/raw_data_statistic', help="Base directory to save the plots.")
    args = parser.parse_args()

    input_file_path = Path(args.input_path)
    base_output_dir = Path(args.output_dir)

    # --- Create timestamped output directory ---
    now = datetime.now()
    timestamp_str = now.strftime("%Y_%m_%d_%H_%M")
    final_output_dir = base_output_dir / timestamp_str
    try:
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"Plots will be saved to: {final_output_dir}")
    except OSError as e:
        print(f"Error creating output directory {final_output_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Create an empty file with the input filename in the output directory ---
    input_filename_marker = final_output_dir / input_file_path.name
    try:
        input_filename_marker.touch(exist_ok=True) # Create empty file, do nothing if exists
        print(f"Created marker file: {input_filename_marker}")
    except OSError as e:
        print(f"Warning: Could not create marker file {input_filename_marker}: {e}", file=sys.stderr)
    # --- ---

    # --- Determine number of lines for tqdm ---
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

        # --- Convert to DataFrame and Extract Features ---
        print("Converting data to DataFrame and extracting features...")
        try:
            df = pd.DataFrame(all_data)

            # --- Feature Extraction ---
            def safe_get(data, keys, default=None):
                """Safely get nested dictionary values."""
                if not isinstance(data, dict):
                    return default
                temp = data
                for key in keys:
                    if isinstance(temp, dict) and key in temp:
                        temp = temp[key]
                    else:
                        return default
                return temp

            # Numerical features
            df['average_depth_of_rule_graph'] = df['average_depth_of_rule_graph']
            df['max_depth_of_rule_graph'] = df['max_depth_of_rule_graph']
            df['max_ary_for_predicates'] = df['max_ary_for_predicates']
            df['max_idx_for_variables'] = df['max_idx_for_variables']
            df['max_predicates_per_rule'] = df['max_predicates_per_rule']
            df['num_facts'] = df['num_facts']
            df['num_noiseless_rules'] = df['num_rules']
            df['num_related_predicates'] = df['num_related_predicates']
            # Keep the original count column if needed for other calculations, but don't plot it directly
            df['num_rules_with_default_negation'] = df['num_rules_with_default_negation']
            df['num_noisy_rules'] = df['dict_structure'].apply(lambda x: sum(len(v) for v in safe_get(x, ['noisy_rules'], {}).values()) if isinstance(safe_get(x, ['noisy_rules']), dict) else 0)
            # Add calculation for min_fact_dicts_for_query count
            df['num_min_facts_for_query'] = df['dict_structure'].apply(lambda x: len(safe_get(x, ['min_fact_dicts_for_query'], [])) if isinstance(safe_get(x, ['min_fact_dicts_for_query']), (list, dict)) else 0)


            # Calculate the ratio
            # Avoid division by zero: if num_noiseless_rules is 0, the ratio is 0.
            df['ratio_default_negation_rules'] = df.apply(
                lambda row: row['num_rules_with_default_negation'] / row['num_rules'] if row['num_noiseless_rules'] > 0 else 0,
                axis=1
            )

            # Categorical features
            df['target_in_answer'] = df['target_query_in_answerset']
            df['target_query_strong_negation'] = df['target_query'].apply(lambda x: safe_get(x, ['strong negation']))
            df['target_query_predicateIdx'] = df['target_query'].apply(lambda x: safe_get(x, ['predicateIdx']))

            print("Feature extraction complete.")
            # Display some info about extracted data
            # print(df.info())
            # print(df.head())

        except Exception as e:
            print(f"Error during DataFrame creation or feature extraction: {e}", file=sys.stderr)
            sys.exit(1)
        # --- ---

        # --- Call Plotting Functions ---
        print("Generating plots...")
        try:
            plot_numerical_distributions(df, final_output_dir)
            plot_categorical_distributions(df, final_output_dir)
            plot_noisy_rule_types(df, final_output_dir)
            # plot_feature_relations(df, final_output_dir) # Optional
            print("Plots generated successfully.")
        except Exception as e:
            print(f"Error during plot generation: {e}", file=sys.stderr)
            # Decide if script should exit on plot error
        # --- ---

    except FileNotFoundError as e: # Handle file not found during initial open or count
        print(f"Error: File not found at {input_file_path}", file=sys.stderr)
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
