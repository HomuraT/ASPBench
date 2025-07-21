# 05_filter_data_for_symtex.py

import argparse
from pathlib import Path
import sys
import multiprocessing # Keep for freeze_support if needed
import random
import os
from datetime import datetime

# Import functions from utils
try:
    # Import read_jsonl_parallel for loading, write_jsonl for saving
    from src.utils.json_utils import read_jsonl_parallel, write_jsonl, JsonUtilsError
except ImportError:
    print("Error: Could not import functions from src.utils.json_utils.", file=sys.stderr)
    print("Ensure the 'src' directory is in your Python path or run this script from the project root.", file=sys.stderr)
    sys.exit(1)

# --- Helper function ---
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
# --- End Helper function ---

def main(args):
    """Loads, filters, samples, and saves data based on provided arguments."""
    input_file_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    print(f"Attempting to load data from: {input_file_path}")

    try:
        # 1. Load Data
        all_data = read_jsonl_parallel(input_file_path) # num_processes=args.num_processes, chunksize=args.chunksize

        if not all_data:
            print("No data loaded from the file. Exiting.")
            sys.exit(0)
        else:
            print(f"Successfully loaded {len(all_data)} entries from {input_file_path}.")

        # 2. Filter Data
        print(f"Filtering data: keeping entries where (num_rules_with_default_negation / num_rules) > {args.min_neg_prob}...")
        filtered_data = []
        for item in all_data:
            num_rules_with_neg = item.get('num_rules_with_default_negation', 0)
            num_rules = item.get('num_rules', 0) # Get num_rules, default to 0 if missing
            # Calculate ratio safely, avoiding division by zero
            ratio = (num_rules_with_neg / num_rules) if num_rules > 0 else 0.0
            if ratio > args.min_neg_prob:
                filtered_data.append(item)

        print(f"Filtered down to {len(filtered_data)} entries.")

        if not filtered_data:
            print("No entries remaining after filtering. Exiting.")
            sys.exit(0)

        # 3. Sample Data
        num_to_sample = args.num_samples
        if len(filtered_data) < num_to_sample:
            print(f"Warning: Requested {num_to_sample} samples, but only {len(filtered_data)} entries available after filtering. Sampling all available entries.", file=sys.stderr)
            num_to_sample = len(filtered_data)

        if num_to_sample <= 0:
             print("No samples to select (either requested 0 or none available after filtering). Exiting.")
             sys.exit(0)

        # --- Calculate num_min_facts_for_query for each item ---
        print("Calculating 'num_min_facts_for_query' for filtered data...")
        for item in filtered_data:
            min_facts = safe_get(item, ['dict_structure', 'min_fact_dicts_for_query'], [])
            item['num_min_facts_for_query'] = len(min_facts) if isinstance(min_facts, (list, dict)) else 0
        print("Calculation complete.")
        # --- End Calculation ---

        print(f"Preparing to sample {num_to_sample} entries using initial random sample + iterative adjustment:")
        print(f"  - Target: {num_to_sample} samples")
        print(f"  - Constraint: Max 20% with max_ary_for_predicates == 0")
        print(f"  - Constraint: Max 20% with num_min_facts_for_query == 0")
        print(f"  - Constraint: Max 20% with num_min_facts_for_query == 1")
        print(f"Using random seed {args.seed}...")

        random.seed(args.seed)

        # --- Initial Random Sample ---
        if len(filtered_data) < num_to_sample:
             print(f"Warning: Not enough filtered data ({len(filtered_data)}) to reach target sample size ({num_to_sample}). Sampling all available.", file=sys.stderr)
             sampled_data = filtered_data[:] # Take all
             unsampled_data = []
             num_to_sample = len(sampled_data) # Adjust target size
        else:
            sampled_indices = random.sample(range(len(filtered_data)), k=num_to_sample)
            sampled_data = [filtered_data[i] for i in sampled_indices]
            unsampled_data = [filtered_data[i] for i in range(len(filtered_data)) if i not in sampled_indices]
        print(f"Initial random sample size: {len(sampled_data)}")
        # --- End Initial Sample ---


        # --- Iterative Adjustment ---
        print("Starting iterative adjustment to meet constraints...")
        max_iterations = 10 # Prevent infinite loops
        for iteration in range(max_iterations):
            made_swap = False
            print(f"Adjustment Iteration {iteration + 1}/{max_iterations}")

            # Define constraints based on the *actual* number of samples we have
            current_sample_size = len(sampled_data)
            if current_sample_size == 0: break # Avoid division by zero if sample is empty

            max_ary0_allowed = int(current_sample_size * 0.20)
            max_mf0_allowed = int(current_sample_size * 0.20)
            max_mf1_allowed = int(current_sample_size * 0.20)

            # --- Adjust ary0 ---
            sampled_ary0_indices = [i for i, item in enumerate(sampled_data) if item.get('max_ary_for_predicates', -1) == 0]
            count_ary0 = len(sampled_ary0_indices)
            if count_ary0 > max_ary0_allowed:
                excess_ary0 = count_ary0 - max_ary0_allowed
                unsampled_non_ary0_indices = [i for i, item in enumerate(unsampled_data) if item.get('max_ary_for_predicates', -1) != 0]
                num_swaps = min(excess_ary0, len(sampled_ary0_indices), len(unsampled_non_ary0_indices))
                if num_swaps > 0:
                    print(f"  Adjusting ary0: Swapping {num_swaps} samples...")
                    indices_to_remove_from_sampled = random.sample(sampled_ary0_indices, k=num_swaps)
                    indices_to_add_from_unsampled = random.sample(unsampled_non_ary0_indices, k=num_swaps)

                    items_to_move_to_unsampled = [sampled_data[i] for i in sorted(indices_to_remove_from_sampled, reverse=True)]
                    items_to_move_to_sampled = [unsampled_data[i] for i in sorted(indices_to_add_from_unsampled, reverse=True)]

                    # Perform removal (iterate backwards to avoid index issues)
                    for index in sorted(indices_to_remove_from_sampled, reverse=True):
                        del sampled_data[index]
                    for index in sorted(indices_to_add_from_unsampled, reverse=True):
                        del unsampled_data[index]

                    # Perform addition
                    sampled_data.extend(items_to_move_to_sampled)
                    unsampled_data.extend(items_to_move_to_unsampled)
                    made_swap = True

            # --- Adjust mf0 ---
            sampled_mf0_indices = [i for i, item in enumerate(sampled_data) if item.get('num_min_facts_for_query', -1) == 0]
            count_mf0 = len(sampled_mf0_indices)
            if count_mf0 > max_mf0_allowed:
                excess_mf0 = count_mf0 - max_mf0_allowed
                unsampled_non_mf0_indices = [i for i, item in enumerate(unsampled_data) if item.get('num_min_facts_for_query', -1) != 0]
                num_swaps = min(excess_mf0, len(sampled_mf0_indices), len(unsampled_non_mf0_indices))
                if num_swaps > 0:
                    print(f"  Adjusting mf0: Swapping {num_swaps} samples...")
                    indices_to_remove_from_sampled = random.sample(sampled_mf0_indices, k=num_swaps)
                    indices_to_add_from_unsampled = random.sample(unsampled_non_mf0_indices, k=num_swaps)
                    # (Similar swap logic as above)
                    items_to_move_to_unsampled = [sampled_data[i] for i in sorted(indices_to_remove_from_sampled, reverse=True)]
                    items_to_move_to_sampled = [unsampled_data[i] for i in sorted(indices_to_add_from_unsampled, reverse=True)]
                    for index in sorted(indices_to_remove_from_sampled, reverse=True): del sampled_data[index]
                    for index in sorted(indices_to_add_from_unsampled, reverse=True): del unsampled_data[index]
                    sampled_data.extend(items_to_move_to_sampled)
                    unsampled_data.extend(items_to_move_to_unsampled)
                    made_swap = True

            # --- Adjust mf1 ---
            sampled_mf1_indices = [i for i, item in enumerate(sampled_data) if item.get('num_min_facts_for_query', -1) == 1]
            count_mf1 = len(sampled_mf1_indices)
            if count_mf1 > max_mf1_allowed:
                excess_mf1 = count_mf1 - max_mf1_allowed
                unsampled_non_mf1_indices = [i for i, item in enumerate(unsampled_data) if item.get('num_min_facts_for_query', -1) != 1]
                num_swaps = min(excess_mf1, len(sampled_mf1_indices), len(unsampled_non_mf1_indices))
                if num_swaps > 0:
                    print(f"  Adjusting mf1: Swapping {num_swaps} samples...")
                    indices_to_remove_from_sampled = random.sample(sampled_mf1_indices, k=num_swaps)
                    indices_to_add_from_unsampled = random.sample(unsampled_non_mf1_indices, k=num_swaps)
                    # (Similar swap logic as above)
                    items_to_move_to_unsampled = [sampled_data[i] for i in sorted(indices_to_remove_from_sampled, reverse=True)]
                    items_to_move_to_sampled = [unsampled_data[i] for i in sorted(indices_to_add_from_unsampled, reverse=True)]
                    for index in sorted(indices_to_remove_from_sampled, reverse=True): del sampled_data[index]
                    for index in sorted(indices_to_add_from_unsampled, reverse=True): del unsampled_data[index]
                    sampled_data.extend(items_to_move_to_sampled)
                    unsampled_data.extend(items_to_move_to_unsampled)
                    made_swap = True

            if not made_swap:
                print("No swaps made in this iteration. Stopping adjustment.")
                break
        else: # Loop finished without break (reached max_iterations)
             print(f"Warning: Reached maximum adjustment iterations ({max_iterations}). Constraints might not be fully met.", file=sys.stderr)
        # --- End Iterative Adjustment ---

        print(f"Finished sampling and adjustment. Final sample size: {len(sampled_data)} entries.")

        # Final check of the ratios in the actual sample
        if sampled_data:
            final_count_ary0 = sum(1 for item in sampled_data if item.get('max_ary_for_predicates', -1) == 0)
            final_count_mf0 = sum(1 for item in sampled_data if item.get('num_min_facts_for_query', -1) == 0)
            final_count_mf1 = sum(1 for item in sampled_data if item.get('num_min_facts_for_query', -1) == 1)
            total_sampled = len(sampled_data)

            print(f"Final counts in sample:")
            print(f"  - max_ary_for_predicates == 0: {final_count_ary0} ({final_count_ary0 / total_sampled:.4f})")
            print(f"  - num_min_facts_for_query == 0: {final_count_mf0} ({final_count_mf0 / total_sampled:.4f})")
            print(f"  - num_min_facts_for_query == 1: {final_count_mf1} ({final_count_mf1 / total_sampled:.4f})")


        # 4. Save Data
        print(f"Preparing to save sampled data to directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        # Construct filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use input filename stem for context if possible
        input_stem = input_file_path.stem.replace('.jsonl', '') # Basic cleaning
        output_filename = f"{timestamp}_from_{input_stem}_seed{args.seed}_n{len(sampled_data)}_minprob{args.min_neg_prob}.jsonl"
        output_file_path = output_dir / output_filename

        print(f"Saving data to: {output_file_path}")
        write_jsonl(sampled_data, str(output_file_path)) # write_jsonl expects string path
        print("Data saved successfully.")

        print("Script finished.")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except JsonUtilsError as e:
        print(f"Error during JSONL processing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, filter, and sample data from a JSONL file.")

    # Input/Output Arguments
    parser.add_argument("--input_path", type=str, default='datasets/symtex_merged_clean_dataset/2025_04_24_08_59.jsonl',
                        help="Path to the input JSONL file (e.g., cleaned data).")
    parser.add_argument("--output_dir", type=str, default='datasets/symtex_filter_from_clean_data',
                        help="Directory to save the output sampled JSONL file.")

    # Sampling Arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling.")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to draw.")

    # Filtering Arguments
    parser.add_argument("--min_neg_prob", type=float, default=0.5,
                        help="Minimum value for 'default_negation_prob' to keep an entry.")

    # Optional arguments for parallel reading (can be uncommented if needed)
    # parser.add_argument("--num_processes", type=int, default=None, help="Number of processes for parallel reading.")
    # parser.add_argument("--chunksize", type=int, default=1000, help="Chunk size for parallel reading.")

    args = parser.parse_args()

    # Add freeze_support() for Windows compatibility if creating executables
    # multiprocessing.freeze_support()
    main(args)
