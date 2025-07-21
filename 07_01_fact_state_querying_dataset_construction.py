# 07_01_fact_state_querying_dataset_construction.py

import argparse
from pathlib import Path
import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional # Added Optional
from datetime import datetime

# --- Add project root to sys.path ---
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Detected project root: {project_root}")
if project_root not in sys.path:
    print(f"[DEBUG] Adding {project_root} to sys.path.")
    sys.path.insert(0, project_root)
else:
    print(f"[DEBUG] {project_root} already in sys.path.")
print(f"[DEBUG] Current sys.path: {sys.path}")
# --- End sys.path modification ---

# Import utility functions
try:
    from src.utils.json_utils import read_jsonl_parallel, JsonUtilsError
    # --- Added imports for consistency check ---
    from src.utils.dlv2_runner import Dlv2Runner, Dlv2RunnerError
    # Removed format_dict_structure_to_asp as we use pre-formatted strings
    # --- End added imports ---
except ImportError as e:
    print(f"Error: Could not import required functions/modules: {e}", file=sys.stderr)
    print("Ensure the 'src' directory is in your Python path, dependencies are installed, and required modules exist.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper function for DLV2 consistency check ---
def check_query_presence(program_str: str, target_query: str, runner: Dlv2Runner, item_id: str, check_type: str) -> Optional[bool]:
    """
    Runs DLV2 on the given program string and checks if the target_query is present in the first answer set.

    :param program_str: The ASP program as a single string.
    :type program_str: str
    :param target_query: The target query string to check for.
    :type target_query: str
    :param runner: An initialized Dlv2Runner instance.
    :type runner: Dlv2Runner
    :param item_id: The ID of the item being processed (for logging).
    :type item_id: str
    :param check_type: Identifier for the type of check (e.g., "noiseless", "all").
    :type check_type: str
    :return: True if the query is present, False if not, None if DLV2 fails or no answer set exists.
    :rtype: Optional[bool]
    """
    log_prefix = f"[{item_id}][Check: {check_type}]"
    if not program_str:
        logging.warning(f"{log_prefix} Cannot run DLV2 check: Program string is empty.")
        return None
    if not target_query:
        logging.warning(f"{log_prefix} Cannot run DLV2 check: Target query is empty.")
        return None

    # Prepare target query for comparison (remove spaces, trailing period)
    target_query_comp = target_query.replace(' ', '').rstrip('.')
    if not target_query_comp:
         logging.warning(f"{log_prefix} Cannot run DLV2 check: Target query became empty after cleaning ('{target_query}').")
         return None

    try:
        logging.debug(f"{log_prefix} Running DLV2...")
        # Run DLV2, requesting only one answer set is enough for presence check
        result = runner.run(program_str, num_answer_sets=1)
        if result.get("success"):
            answer_sets = result.get("answer_sets")
            if answer_sets and isinstance(answer_sets, list) and len(answer_sets) > 0:
                first_answer_set = answer_sets[0]
                if isinstance(first_answer_set, list):
                    # Check for presence (case-sensitive, structure-sensitive after cleaning)
                    is_present = target_query_comp in first_answer_set
                    logging.debug(f"{log_prefix} Query '{target_query_comp}' presence: {is_present}")
                    return is_present
                else:
                    logging.warning(f"{log_prefix} DLV2 check failed: First answer set is not a list: {type(first_answer_set)}")
                    return None
            else:
                # No answer set found (program might be incoherent or have no models)
                logging.debug(f"{log_prefix} DLV2 check: No answer set found.")
                return False # Query cannot be present if there's no answer set
        else:
            # DLV2 execution failed
            error_msg = result.get('error_message', 'Unknown DLV2 error')
            logging.warning(f"{log_prefix} DLV2 check execution failed: {error_msg}")
            # Log raw output for debugging if available
            raw_output = result.get('raw_output')
            if raw_output:
                logging.debug(f"{log_prefix} DLV2 raw output (failure):\n{raw_output[:500]}...") # Log truncated output
            return None
    except (Dlv2RunnerError, ValueError, Exception) as e:
        logging.error(f"{log_prefix} Python error during DLV2 check execution: {e}", exc_info=True)
        return None

def _get_asp_list(asp_program_dict: Optional[Dict], key: str, item_id: str) -> List[str]:
    """Safely retrieves a list of ASP strings from the dictionary."""
    if not isinstance(asp_program_dict, dict):
        logging.warning(f"[{item_id}] Cannot get ASP list for '{key}': asp_program_dlv2 is not a dictionary.")
        return []
    asp_list = asp_program_dict.get(key, [])
    if not isinstance(asp_list, list):
        logging.warning(f"[{item_id}] Expected a list for ASP key '{key}', but got {type(asp_list)}. Returning empty list.")
        return []
    # Filter out non-string elements just in case
    return [item for item in asp_list if isinstance(item, str)]


def main(args: argparse.Namespace) -> None:
    """
    Main function to load fact state querying data, process it (initially just load),
    and prepare for saving the results.

    :param args: Command-line arguments parsed by argparse.
    :type args: argparse.Namespace
    :return: None
    :rtype: None
    """
    input_file_path: Path = Path(args.input_path)
    output_dir_path: Path = Path(args.output_dir)

    # --- Validate Input Path ---
    if not input_file_path.is_file():
        logging.error(f"Input file not found at {input_file_path}")
        sys.exit(1)
    logging.info(f"Input data file: {input_file_path}")

    # --- Determine Output Path ---
    # Get the base name of the input file without extension
    input_filename_base: str = input_file_path.stem
    output_filename: str = f"fact_state_querying_{input_filename_base}.jsonl"
    output_file_path: Path = output_dir_path / output_filename

    # Ensure output directory exists
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {output_dir_path}")
        logging.info(f"Output data file will be: {output_file_path}")
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir_path}: {e}")
        sys.exit(1)

    # --- Load Input Data ---
    logging.info(f"Attempting to load data from: {input_file_path}")
    try:
        all_data: List[Dict[str, Any]] = read_jsonl_parallel(input_file_path)

        if not all_data:
            logging.warning("No data loaded from the file. Exiting.")
            sys.exit(0)
        else:
            logging.info(f"Successfully loaded {len(all_data)} entries from {input_file_path}.")

        # --- Placeholder for further processing ---
        logging.info("Data loaded. Further processing steps would go here.")
        # For now, the script stops after loading the data as requested.

        # Example: Accessing the first item if needed
        # if all_data:
        #     logging.debug(f"First data item: {all_data[0]}")

        # --- Initialize DLV2 Runner ---
        try:
            runner = Dlv2Runner()
            logging.info(f"Dlv2Runner initialized. Using DLV2 executable at: {runner.dlv2_path}")
        except Dlv2RunnerError as e:
            logging.error(f"Failed to initialize Dlv2Runner: {e}")
            sys.exit(1)

        # --- Process Data ---
        logging.info("Starting data processing...")
        processed_samples: List[Dict[str, Any]] = []
        # Define keys to exclude from the final output sample
        keys_to_exclude_final: set = {
            'dict_structure',           # Original dict structure
            'dlv2_asp_str',             # Full ASP string from script 06 (if present)
            'dlv2_consistency_check_passed', # Old consistency check field from script 06
            'dlv2_result',              # Original DLV2 result from script 06
            'predicate_idx_to_desc',    # Mappings used for generation, not needed downstream
            'name_idx_to_desc',         # Mappings used for generation, not needed downstream
            'var_idx_to_name'           # Mappings used for generation, not needed downstream
        }
        consistency_check_failed_inconsistent = 0
        consistency_check_failed_error = 0
        consistency_check_skipped_missing_data = 0

        for i, data in enumerate(all_data):
            item_id = data.get('id', f'item_{i}') # Get item ID for logging
            logging.debug(f"Processing item {item_id}...")

            # --- Extract necessary data for consistency check ---
            asp_program_dict = data.get('asp_program_dlv2') # Dict containing lists of ASP strings
            target_query_str = data.get('target_query') # Formatted string query from script 06

            # Check if necessary data for the check is present
            if not isinstance(asp_program_dict, dict) or not isinstance(target_query_str, str):
                logging.warning(f"[{item_id}] Skipping consistency check: Missing or invalid 'asp_program_dlv2' (dict) or 'target_query' (str).")
                consistency_check_skipped_missing_data += 1
                # Decide whether to skip the item entirely or proceed without check
                # Let's proceed but log the skip, the label determination below doesn't depend on the check
                # continue # Uncomment this line to skip items that fail the pre-check
            else:
                # --- Perform Consistency Checks (Run DLV2 three times) ---
                result_noiseless: Optional[bool] = None
                result_all: Optional[bool] = None
                result_min_noisy: Optional[bool] = None
                check_error_occurred = False

                # 1. Check Noiseless: noiseless_facts + noiseless_rules
                nf = _get_asp_list(asp_program_dict, 'noiseless_facts', item_id)
                nr = _get_asp_list(asp_program_dict, 'noiseless_rules', item_id)
                program_noiseless_str = "\n".join(nf + nr)
                result_noiseless = check_query_presence(program_noiseless_str, target_query_str, runner, item_id, "noiseless")
                if result_noiseless is None: check_error_occurred = True

                # 2. Check All: noiseless_facts + noisy_facts + noiseless_rules + noisy_rules
                noisy_f = _get_asp_list(asp_program_dict, 'noisy_facts', item_id)
                noisy_r_dict = asp_program_dict.get('noisy_rules', {}) # Noisy rules are a dict of lists
                noisy_r_flat = [rule for sublist in noisy_r_dict.values() for rule in sublist if isinstance(sublist, list) and isinstance(rule, str)] if isinstance(noisy_r_dict, dict) else []
                program_all_str = "\n".join(nf + noisy_f + nr + noisy_r_flat)
                result_all = check_query_presence(program_all_str, target_query_str, runner, item_id, "all")
                if result_all is None: check_error_occurred = True

                # 3. Check Min + Noisy: min_fact_dicts_for_query + noisy_facts + noiseless_rules + noisy_rules
                min_f = _get_asp_list(asp_program_dict, 'min_fact_dicts_for_query', item_id)
                program_min_noisy_str = "\n".join(min_f + noisy_f + nr + noisy_r_flat)
                program_min_noisy_str = "\n".join(min_f  + nr + noisy_r_flat)
                result_min_noisy = check_query_presence(program_min_noisy_str, target_query_str, runner, item_id, "min_noisy")
                if result_min_noisy is None: check_error_occurred = True

                # --- Compare Results ---
                if check_error_occurred:
                    logging.warning(f"[{item_id}] Consistency check failed due to DLV2 error in one or more runs. Skipping item.")
                    consistency_check_failed_error += 1
                    continue # Skip this item if any check had an execution error

                results = [result_noiseless, result_all, result_min_noisy]
                if len(set(results)) > 1: # Check if all results are the same
                    logging.error(f"[{item_id}] Consistency check failed! Results differ: Noiseless={result_noiseless}, All={result_all}, Min+Noisy={result_min_noisy}. Skipping item.")
                    consistency_check_failed_inconsistent += 1
                    continue # Skip this item due to inconsistency
                else:
                    logging.debug(f"[{item_id}] Consistency check passed: Noiseless={result_noiseless}, All={result_all}, Min+Noisy={result_min_noisy}")
                    # Proceed with normal processing for this item

            # --- Original Processing Steps (Label determination, etc.) ---
            # Use the target_query_str extracted earlier
            # Use the original target_query_in_answerset from the input data for label determination
            target_in_answer_original: bool = data.get('target_query_in_answerset', False) # Use original truth value

            label: str
            if not target_in_answer_original:
                label = 'unknown'
            else:
                # Rule 2: Check if starts with '-' OR contains '-'
                # Use target_query_str here, which holds the string from the input data
                if target_query_str and (target_query_str.startswith('-') or '-' in target_query_str):
                    label = 'negative'
                else:
                    label = 'positive'

            # 2. Process target_query (remove leading negation if present)
            # Use target_query_str which is already the potentially name-mapped query string
            processed_target_query: str = target_query_str
            if target_query_str and target_query_str.startswith('-'):
                processed_target_query = target_query_str[1:]

            # 3. Create new sample, excluding specified keys and renaming 'ID' to 'id'
            new_sample: Dict[str, Any] = {}
            for key, value in data.items():
                if key not in keys_to_exclude_final:
                    # Rename 'ID' to 'id' if present, otherwise keep original key
                    output_key = 'id' if key == 'ID' else key
                    new_sample[output_key] = value

            # Add label and processed query (overwriting original target_query string)
            new_sample['label'] = label
            new_sample['target_query'] = processed_target_query # Store the potentially de-negated query string

            processed_samples.append(new_sample)
            # End of loop for item

        logging.info(f"Finished processing {len(all_data)} items.")
        logging.info(f"  Successfully processed and kept: {len(processed_samples)} samples.")
        logging.info(f"  Skipped due to inconsistency: {consistency_check_failed_inconsistent} samples.")
        logging.info(f"  Skipped due to DLV2 error during check: {consistency_check_failed_error} samples.")
        logging.info(f"  Skipped due to missing input data for check: {consistency_check_skipped_missing_data} samples.")

        # --- Save Processed Data ---
        logging.info(f"Attempting to save processed data to: {output_file_path}")
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for sample in processed_samples:
                    # Ensure JSON objects are written correctly per line
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            logging.info(f"Successfully saved {len(processed_samples)} processed samples to {output_file_path}")
        except IOError as e:
            logging.error(f"Failed to write output file {output_file_path}: {e}")
            sys.exit(1)
        except Exception as e: # Catch other potential errors during saving
            logging.error(f"An unexpected error occurred during saving: {e}", exc_info=True)
            sys.exit(1)

        logging.info("Script finished processing and saving.")

    except FileNotFoundError: # Should be caught by is_file() check, but good practice
        logging.error(f"Input file not found at {input_file_path}")
        sys.exit(1)
    except JsonUtilsError as e:
        logging.error(f"Error during JSONL reading: {e}")
        sys.exit(1)
    except Exception as e: # Catch other potential errors during loading
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        sys.exit(1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constructs a dataset for fact/state querying tasks. Reads a JSONL input file and prepares an output file."
    )

    # Input/Output Arguments
    parser.add_argument(
        "--input_path",
        type=str,
        default='datasets/symtex_dlv2/wo_disjunction/related_word_2025_04_24_09_36.jsonl',
        help="Path to the input JSONL file containing the base data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./datasets/a_symtex_task_fact_state_querying',
        help="Directory to save the output JSONL file."
    )

    args = parser.parse_args()
    main(args)
