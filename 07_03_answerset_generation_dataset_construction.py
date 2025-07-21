# 07_03_answerset_selection_dataset_construction.py
# Simplified script to find and store correct answer sets using DLV2,
# removing the incorrect answer set generation logic.

import argparse
from pathlib import Path
import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set # Added Optional, Tuple, Set
from datetime import datetime
import random
import re

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
    from src.utils.json_utils import read_jsonl_parallel, write_jsonl, JsonUtilsError # Added write_jsonl
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

# --- Helper Functions for Fact Manipulation ---

# Regex to capture predicate and constants within parentheses, handling quoted strings
FACT_REGEX = re.compile(r'^(-?)([a-zA-Z0-9_]+)\((.*?)\)$')
# Regex to split constants, respecting quotes
CONSTANTS_REGEX = re.compile(r'"[^"]*"|\'[^\']*\'|[a-zA-Z0-9_]+')

def parse_fact(fact_str: str) -> Optional[Tuple[bool, str, List[str]]]:
    """
    Parses a fact string into its components: negation, predicate, and constants.

    Handles facts like 'predicate("const1", "const2")' or '-neg_pred("c1")'.

    :param fact_str: The fact string to parse.
    :type fact_str: str
    :return: A tuple (is_negated, predicate, constants) if parsing is successful, otherwise None.
             is_negated (bool): True if the fact starts with '-'.
             predicate (str): The name of the predicate.
             constants (List[str]): A list of constant strings (including quotes if present).
    :rtype: Optional[Tuple[bool, str, List[str]]]
    """
    match = FACT_REGEX.match(fact_str.strip())
    if not match:
        # logging.warning(f"Could not parse fact string: {fact_str}")
        return None

    negation_prefix, predicate, constants_str = match.groups()
    is_negated = negation_prefix == '-'

    # Split constants carefully, handling quotes
    constants = CONSTANTS_REGEX.findall(constants_str)

    return is_negated, predicate, constants

def format_fact(is_negated: bool, predicate: str, constants: List[str]) -> str:
    """
    Formats fact components back into a standard fact string.

    :param is_negated: Whether the fact is negated.
    :type is_negated: bool
    :param predicate: The predicate name.
    :type predicate: str
    :param constants: A list of constant strings.
    :type constants: List[str]
    :return: The formatted fact string (e.g., '-predicate("c1","c2")').
    :rtype: str
    """
    constants_str = ",".join(constants)
    negation_prefix = "-" if is_negated else ""
    return f"{negation_prefix}{predicate}({constants_str})"

# --- End Helper Functions ---


def main(args: argparse.Namespace) -> None:
    """
    Main function to load data, run DLV2 to find answer sets, store them,
    and save the processed data.

    :param args: Command-line arguments parsed by argparse.
    :type args: argparse.Namespace
    :return: None
    :rtype: None
    """
    # --- Set Random Seed ---
    if args.seed is not None:
        random.seed(args.seed)
        logging.info(f"Random seed set to: {args.seed}")
    else:
        logging.info("No random seed provided. Using default random state.")

    input_file_path_w: Path = Path(args.input_path_w_disj)
    input_file_path_wo: Path = Path(args.input_path_wo_disj)
    output_dir_path: Path = Path(args.output_dir)

    # --- Validate Input Paths ---
    if not input_file_path_w.is_file():
        logging.error(f"Input file (w_disj) not found at {input_file_path_w}")
        sys.exit(1)
    if not input_file_path_wo.is_file():
        logging.error(f"Input file (wo_disj) not found at {input_file_path_wo}")
        sys.exit(1)
    logging.info(f"Input data file (w_disj): {input_file_path_w}")
    logging.info(f"Input data file (wo_disj): {input_file_path_wo}")

    # --- Determine Output Path ---
    # Use the base name from the w_disj path as a reference for the output filename
    input_filename_base: str = input_file_path_w.stem
    # Modify output filename to indicate combined source and task
    output_filename: str = f"answerset_generation_{input_filename_base}_combined.jsonl" # Changed prefix and added _combined
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
    logging.info(f"Attempting to load data from: {input_file_path_w}")
    data_w_disj: List[Dict[str, Any]] = read_jsonl_parallel(input_file_path_w)
    logging.info(f"Attempting to load data from: {input_file_path_wo}")
    data_wo_disj: List[Dict[str, Any]] = read_jsonl_parallel(input_file_path_wo)

    if not data_w_disj or not data_wo_disj:
        logging.warning("One or both input files are empty or failed to load. Exiting.")
        sys.exit(0)
    else:
        logging.info(f"Successfully loaded {len(data_w_disj)} entries from {input_file_path_w}.")
        logging.info(f"Successfully loaded {len(data_wo_disj)} entries from {input_file_path_wo}.")

    # --- Combine Data based on ID ---
    logging.info("Combining data based on ID, selecting half from each source randomly.")
    map_w: Dict[Any, Dict[str, Any]] = {item.get('id', item.get('ID')): item for item in data_w_disj if item.get('id') or item.get('ID')} # Handle both 'id' and 'ID' keys
    map_wo: Dict[Any, Dict[str, Any]] = {item.get('id', item.get('ID')): item for item in data_wo_disj if item.get('id') or item.get('ID')}

    # Find common IDs
    common_ids = list(set(map_w.keys()) & set(map_wo.keys()))
    if not common_ids:
        logging.error("No common IDs found between the two input files. Cannot combine.")
        sys.exit(1)

    # Log if there are non-common IDs (optional, but good for debugging)
    non_common_w = set(map_w.keys()) - set(map_wo.keys())
    non_common_wo = set(map_wo.keys()) - set(map_w.keys())
    if non_common_w:
        logging.warning(f"{len(non_common_w)} IDs found only in {input_file_path_w.name}: {list(non_common_w)[:5]}...") # Show first 5
    if non_common_wo:
        logging.warning(f"{len(non_common_wo)} IDs found only in {input_file_path_wo.name}: {list(non_common_wo)[:5]}...") # Show first 5

    # Shuffle common IDs and split
    random.shuffle(common_ids)
    num_common = len(common_ids)
    split_point = num_common // 2
    ids_from_w = set(common_ids[:split_point])
    ids_from_wo = set(common_ids[split_point:])

    combined_data: List[Dict[str, Any]] = []
    for id_val in common_ids:
        if id_val in ids_from_w:
            combined_data.append(map_w[id_val])
        elif id_val in ids_from_wo:
            combined_data.append(map_wo[id_val])

    logging.info(f"Combined data created with {len(combined_data)} entries ({len(ids_from_w)} from w_disj, {len(ids_from_wo)} from wo_disj).")

    # --- Initialize DLV2 Runner ---
    try:
        runner = Dlv2Runner()
        logging.info(f"Dlv2Runner initialized. Using DLV2 executable at: {runner.dlv2_path}")
    except Dlv2RunnerError as e:
        logging.error(f"Failed to initialize Dlv2Runner: {e}")
        sys.exit(1)

    # --- Process each data entry ---
    processed_data_count = 0
    # !!! IMPORTANT: Loop over combined_data now !!!
    for i, data in enumerate(combined_data):
        logging.debug(f"Processing item {i+1}/{len(combined_data)}") # Use len(combined_data)
        if 'asp_program_dlv2' not in data:
            logging.warning(f"Skipping item {i+1} due to missing 'asp_program_dlv2' key.")
            data['num_answer_sets'] = -1 # Indicate skipped
            data['answer_set_fact_counts'] = []
            data['dlv2_error'] = "Missing 'asp_program_dlv2' key"
            continue

        asp_program = data['asp_program_dlv2']
        all_facts = set()
        all_rules = set()

        # Extract facts
        # Facts are typically atoms ending with a period.
        fact_sources = ['min_fact_dicts_for_query', 'noiseless_facts', 'noisy_facts']
        for key in fact_sources:
            if key in asp_program and isinstance(asp_program[key], list):
                all_facts.update(fact for fact in asp_program[key] if isinstance(fact, str) and fact.strip().endswith('.'))
            elif key in asp_program:
                logging.warning(f"Expected list for '{key}' in item {i+1}, found {type(asp_program[key])}. Skipping facts from this source.")

        # Extract rules
        # Rules typically contain ':-'.
        rule_sources = ['noiseless_rules']
        for key in rule_sources:
             if key in asp_program and isinstance(asp_program[key], list):
                 all_rules.update(rule for rule in asp_program[key] if isinstance(rule, str) and ':-' in rule)
             elif key in asp_program:
                logging.warning(f"Expected list for '{key}' in item {i+1}, found {type(asp_program[key])}. Skipping rules from this source.")

        # Extract noisy rules (nested structure)
        if 'noisy_rules' in asp_program and isinstance(asp_program['noisy_rules'], dict):
            for sub_key, rule_list in asp_program['noisy_rules'].items():
                if isinstance(rule_list, list):
                    all_rules.update(rule for rule in rule_list if isinstance(rule, str) and ':-' in rule)
                else:
                     logging.warning(f"Expected list for 'noisy_rules.{sub_key}' in item {i+1}, found {type(rule_list)}. Skipping rules from this source.")
        elif 'noisy_rules' in asp_program:
             logging.warning(f"Expected dict for 'noisy_rules' in item {i+1}, found {type(asp_program['noisy_rules'])}. Skipping noisy rules.")

        # Combine facts and rules into a single program string
        # Ensure items are strings before joining
        program_parts = [str(f) for f in all_facts] + [str(r) for r in all_rules]
        program_str = "\n".join(program_parts)

        # --- Store extracted facts/rules and calculate counts ---
        # Convert sets to sorted lists for consistent output
        facts_list = sorted(list(all_facts))
        rules_list = sorted(list(all_rules))
        data['facts'] = facts_list
        data['rules'] = rules_list
        data['num_facts'] = len(facts_list)
        data['num_rules'] = len(rules_list)
        # Count rules containing default negation (' not ')
        data['num_rules_with_default_negation'] = sum(1 for rule in rules_list if ' not ' in rule)
        logging.debug(f"Item {i+1}: Stored {data['num_facts']} facts, {data['num_rules']} rules ({data['num_rules_with_default_negation']} with ' not ').")

        # Add comment for clarity in logs if needed
        # logging.debug(f"Running DLV2 for item {i+1} with program:\n{program_str[:500]}...") # Log first 500 chars

        try:
            # Run DLV2, requesting all answer sets
            # :param program_str: The ASP program as a single string.
            # :type program_str: str
            # :param num_answer_sets: Number of answer sets to compute (0 for all).
            # :type num_answer_sets: int
            # :return: Dictionary with 'answer_sets', 'error_message', 'raw_output', 'success'.
            # :rtype: Dict[str, Any]
            result: Dict[str, Any] = runner.run(program_str, num_answer_sets=0)
            # Store the new result temporarily
            dlv2_run_result = result

            if dlv2_run_result and dlv2_run_result.get('success'):
                answer_sets: List[List[str]] = dlv2_run_result.get('answer_sets', [])
                # Store answer sets at the top level
                data['answer_sets'] = answer_sets
                # :param answer_sets: A list of answer sets, where each answer set is a list of strings (facts).
                # :type answer_sets: List[List[str]]
                data['num_answer_sets'] = len(answer_sets)

                # Count all string items as facts in each answer set
                data['answer_set_fact_counts'] = [
                    sum(1 for item in ans_set if isinstance(item, str))
                    for ans_set in answer_sets
                ]
                logging.debug(f"Item {i+1}: Success. Found {data['num_answer_sets']} answer sets. Fact counts: {data['answer_set_fact_counts']}")
                # Store the raw DLV2 output details if needed for debugging, but it will be removed later
                data['answerset_selection_dlv2_result_details'] = {
                    'raw_output': dlv2_run_result.get('raw_output'),
                    'error_message': dlv2_run_result.get('error_message'),
                    'success': dlv2_run_result.get('success')
                }
                # Incorrect answer set generation logic removed.

            else: # Handle DLV2 failure case
                error_msg = dlv2_run_result.get('error_message', 'Unknown DLV2 error')
                logging.error(f"DLV2 execution failed for item {i+1}. Error: {error_msg}")
                data['answer_sets'] = [] # Ensure key exists even on failure
                data['num_answer_sets'] = 0
                data['answer_set_fact_counts'] = []
                data['dlv2_error'] = error_msg
                # Store failure details
                data['answerset_selection_dlv2_result_details'] = {
                     'raw_output': dlv2_run_result.get('raw_output'),
                     'error_message': error_msg,
                     'success': False
                 }

        except Dlv2RunnerError as e:
            logging.error(f"Dlv2RunnerError processing item {i+1}: {e}")
            data['num_answer_sets'] = 0
            data['answer_set_fact_counts'] = []
            data['dlv2_error'] = f"Dlv2RunnerError: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected error processing item {i+1}: {e}", exc_info=True) # Log traceback
            data['num_answer_sets'] = 0
            data['answer_set_fact_counts'] = []
            data['dlv2_error'] = f"Unexpected error: {str(e)}"

        processed_data_count += 1
        # Optional: Add a progress log
        if (i + 1) % 100 == 0 or (i + 1) == len(combined_data): # Use len(combined_data)
             logging.info(f"Processed {i + 1}/{len(combined_data)} items.") # Use len(combined_data)

    logging.info(f"Finished processing loop. Successfully processed {processed_data_count} items.")

    # --- Prepare Final Data for Saving (Remove specified keys) ---
    keys_to_exclude_final: set = {
        'dict_structure',           # Original dict structure
        'dlv2_asp_str',             # Full ASP string from script 06 (if present)
        'dlv2_consistency_check_passed', # Old consistency check field from script 06
        'dlv2_result',              # Original DLV2 result from script 06 (if present)
        'answerset_selection_dlv2_result_details', # Remove the temporary storage for DLV2 details
        'predicate_idx_to_desc',    # Mappings used for generation, not needed downstream
        'name_idx_to_desc',         # Mappings used for generation, not needed downstream
        'var_idx_to_name',           # Mappings used for generation, not needed downstream
        'target_query',             # Original query info, potentially redundant or not needed
        'target_query_in_answerset',# Original query info, potentially redundant or not needed
        'num_noise_rules_per_type',
        'num_related_predicates',
        'max_depth_of_rule_graph',
        'average_depth_of_rule_graph',
        'num_min_facts_for_query',
        'asp_program_dlv2',
    }
    logging.info(f"Preparing final data for saving by removing keys: {keys_to_exclude_final} and renaming 'ID' to 'id'")
    final_data_to_save: List[Dict[str, Any]] = []
    # !!! IMPORTANT: Iterate over combined_data for final saving !!!
    for data_item in combined_data:
        # Create new sample, excluding specified keys and renaming 'ID' to 'id'
        new_sample: Dict[str, Any] = {}
        for key, value in data_item.items():
            if key not in keys_to_exclude_final:
                # Rename 'ID' to 'id' if present, otherwise keep original key
                output_key = 'id' if key == 'ID' else key
                new_sample[output_key] = value
        final_data_to_save.append(new_sample)

    logging.info(f"Finished cleaning data. Final dataset size: {len(final_data_to_save)} items.")

    # --- Save Processed Data ---
    # Use the already determined output_file_path which includes '_combined' and correct task name
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Timestamp can be added if desired
    # output_filename_processed: str = f"answerset_generation_{input_filename_base}_combined_{timestamp}.jsonl"
    output_file_path_processed: Path = output_file_path # Use the path determined earlier
    logging.info(f"Attempting to save final combined and cleaned data to: {output_file_path_processed}")

    try:
        # Use the imported utility function to write the cleaned final_data_to_save list
        # :param file_path: Path to the output JSONL file.
        # :type file_path: Path | str
        # :param data: List of dictionaries to write.
        # :type data: List[Dict[str, Any]]
        # :return: None
        # :rtype: None
        write_jsonl(final_data_to_save, str(output_file_path_processed))
        logging.info(f"Successfully saved {len(final_data_to_save)} cleaned entries to {output_file_path_processed}.")
    except JsonUtilsError as e:
        logging.error(f"Failed to save cleaned data using write_jsonl: {e}")
        # Optionally, implement a fallback saving mechanism here if critical
        sys.exit(1)
    except NameError:
         logging.error("Failed to save processed data: 'write_jsonl' function not found. Make sure it's imported correctly.")
         sys.exit(1)
    except Exception as e: # Catch other potential errors during saving
        logging.error(f"An unexpected error occurred during saving: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constructs a dataset for answer set selection tasks. Reads a JSONL input file containing ASP programs, runs DLV2 to find correct answer sets, and saves the results along with metadata."
    )

    # Input/Output Arguments
    parser.add_argument(
        "--input_path_w_disj", # Changed from --input_path
        type=str,
        required=True, # Make required
        help="Path to the input JSONL file from the 'w_disjunction' directory."
    )
    parser.add_argument(
        "--input_path_wo_disj", # Added new argument
        type=str,
        required=True, # Make required
        help="Path to the input JSONL file from the 'wo_disjunction' directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./datasets/a_symtex_task_answerset_generation',
        help="Directory to save the output JSONL file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, # Default to None, meaning no seed unless specified
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()
    main(args)
