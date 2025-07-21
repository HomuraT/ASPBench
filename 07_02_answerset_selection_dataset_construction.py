# 07_01_fact_state_querying_dataset_construction.py

import argparse
from pathlib import Path
import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set # Added Optional, Tuple, Set
# Removed unused datetime import
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

# Regex to capture predicate and constants within parentheses (no trailing dot expected based on answer_sets)
FACT_REGEX = re.compile(r'^(-?)([a-zA-Z0-9_]+)\((.*?)\)$')
# Regex for 0-ary facts (no parentheses, no trailing dot expected based on answer_sets)
ZERO_ARY_FACT_REGEX = re.compile(r'^(-?)([a-zA-Z0-9_]+)$')
# Regex to split constants, respecting quotes
CONSTANTS_REGEX = re.compile(r'"[^"]*"|\'[^\']*\'|[a-zA-Z0-9_]+')

def parse_fact(fact_str: str) -> Optional[Tuple[bool, str, List[str]]]:
    """
    Parses a fact string into its components: negation, predicate, and constants.

    Handles facts like 'predicate("const1", "const2")', '-neg_pred("c1")', 'atom', '-neg_atom'. (No trailing dots)

    :param fact_str: The fact string to parse.
    :type fact_str: str
    :return: A tuple (is_negated, predicate, constants) if parsing is successful, otherwise None.
             is_negated (bool): True if the fact starts with '-'.
             predicate (str): The name of the predicate.
             constants (List[str]): A list of constant strings (including quotes if present).
    :rtype: Optional[Tuple[bool, str, List[str]]]
    """
    fact_str = fact_str.strip()
    match = FACT_REGEX.match(fact_str)
    if match:
        # Handles facts with parentheses, e.g., pred(a,b) or -pred(a)
        negation_prefix, predicate, constants_str = match.groups()
        is_negated = negation_prefix == '-'
        constants = CONSTANTS_REGEX.findall(constants_str)
        # Check if it might *also* match the 0-ary pattern (e.g., a predicate named "p()" parsed incorrectly)
        # This is unlikely if predicates don't contain parentheses in their names.
        # If it could happen, prioritize the FACT_REGEX match.
        return is_negated, predicate, constants
    else:
        # Try matching 0-ary fact format (e.g., 'atom' or '-atom')
        zero_ary_match = ZERO_ARY_FACT_REGEX.match(fact_str)
        if zero_ary_match:
            # Ensure it didn't accidentally match something with parentheses
            # (This check might be redundant depending on regex guarantees, but adds safety)
            if '(' in fact_str or ')' in fact_str:
                 # logging.warning(f"Potential ambiguous parse for: {fact_str}. Matched 0-ary but contains parentheses.")
                 return None # Or handle as error / prefer FACT_REGEX if it could match

            negation_prefix, predicate = zero_ary_match.groups()
            is_negated = negation_prefix == '-'
            return is_negated, predicate, [] # Return empty list for constants
        else:
            # logging.warning(f"Could not parse fact string: {fact_str}")
            return None

def format_fact(is_negated: bool, predicate: str, constants: List[str]) -> str:
    """
    Formats fact components back into a standard fact string.

    :param is_negated: Whether the fact is negated.
    :type is_negated: bool
    :param predicate: The predicate name.
    :type predicate: str
    :param constants: A list of constant strings.
    :type constants: List[str]
    :return: The formatted fact string (e.g., '-predicate("c1","c2")' or 'atom'). (No trailing dots)
    :rtype: str
    """
    negation_prefix = "-" if is_negated else ""
    if not constants: # Check if constants list is empty for 0-ary predicate
        return f"{negation_prefix}{predicate}"
    else:
        constants_str = ",".join(constants)
        # Format without trailing dot, matching the apparent format in answer_sets
        return f"{negation_prefix}{predicate}({constants_str})"

# --- End Helper Functions ---

# --- NEW: Helper Function for Program Component Extraction ---
def extract_program_components(asp_program: Dict[str, Any], item_index: int) -> Tuple[Set[str], Set[str]]:
    """
    Extracts facts and rules from the asp_program dictionary.

    :param asp_program: The dictionary containing ASP program parts.
    :type asp_program: Dict[str, Any]
    :param item_index: The index of the current item (for logging).
    :type item_index: int
    :return: A tuple containing a set of facts and a set of rules.
    :rtype: Tuple[Set[str], Set[str]]
    """
    all_facts: Set[str] = set()
    all_rules = set()

    # Extract facts (ending with '.')
    fact_sources = ['min_fact_dicts_for_query', 'noiseless_facts', 'noisy_facts']
    for key in fact_sources:
        if key in asp_program:
            if isinstance(asp_program[key], list):
                all_facts.update(fact for fact in asp_program[key] if isinstance(fact, str) and fact.strip().endswith('.'))
            else:
                logging.warning(f"Expected list for '{key}' in item {item_index+1}, found {type(asp_program[key])}. Skipping facts.")

    # Extract rules (containing ':-')
    rule_sources = ['noiseless_rules']
    for key in rule_sources:
         if key in asp_program:
             if isinstance(asp_program[key], list):
                 all_rules.update(rule for rule in asp_program[key] if isinstance(rule, str) and ':-' in rule)
             else:
                logging.warning(f"Expected list for '{key}' in item {item_index+1}, found {type(asp_program[key])}. Skipping rules.")

    # Extract noisy rules (nested structure)
    if 'noisy_rules' in asp_program:
        if isinstance(asp_program['noisy_rules'], dict):
            for sub_key, rule_list in asp_program['noisy_rules'].items():
                if isinstance(rule_list, list):
                    all_rules.update(rule for rule in rule_list if isinstance(rule, str) and ':-' in rule)
                else:
                     logging.warning(f"Expected list for 'noisy_rules.{sub_key}' in item {item_index+1}, found {type(rule_list)}. Skipping rules.")
        else:
             logging.warning(f"Expected dict for 'noisy_rules' in item {item_index+1}, found {type(asp_program['noisy_rules'])}. Skipping noisy rules.")

    return all_facts, all_rules
# --- END NEW Helper Function ---


# --- NEW: Helper Functions for Incorrect Answer Set Generation ---

def _generate_flip_negation(base_set: Set[str], base_set_list: List[str]) -> Optional[Set[str]]:
    """
    Strategy 1: Flip negation of a random fact from the base set.

    :param base_set: The set of facts in the base answer set.
    :type base_set: Set[str]
    :param base_set_list: The list version of the base answer set (for random.choice).
    :type base_set_list: List[str]
    :return: A new set with one fact's negation flipped, or None if modification fails.
    :rtype: Optional[Set[str]]
    """
    if not base_set_list: return None # Check list for random.choice
    fact_to_modify = random.choice(base_set_list)
    parsed = parse_fact(fact_to_modify)
    if not parsed: return None
    is_negated, pred, consts = parsed
    new_fact = format_fact(not is_negated, pred, consts)
    return (base_set - {fact_to_modify}) | {new_fact}

def _generate_delete_fact(base_set: Set[str], base_set_list: List[str]) -> Optional[Set[str]]:
    """
    Strategy 2: Delete a random fact from the base set.

    :param base_set: The set of facts in the base answer set.
    :type base_set: Set[str]
    :param base_set_list: The list version of the base answer set (for random.choice).
    :type base_set_list: List[str]
    :return: A new set with one fact removed, potentially empty. Returns None if base_set_list is empty.
    :rtype: Optional[Set[str]]
    """
    if not base_set_list: return None # Check list for random.choice
    fact_to_delete = random.choice(base_set_list)
    return base_set - {fact_to_delete}

def _generate_add_modified_constants(base_set: Set[str], all_facts_in_correct: List[str], all_constants: List[str]) -> Optional[Set[str]]:
    """
    Strategy 3: Add a fact (copied from any correct set) with modified constants to the base set.

    :param base_set: The set of facts in the base answer set to add to.
    :type base_set: Set[str]
    :param all_facts_in_correct: A flattened list of all facts appearing in any correct answer set.
    :type all_facts_in_correct: List[str]
    :param all_constants: A list of all possible constants available.
    :type all_constants: List[str]
    :return: A new set with the added modified fact, or None if modification fails or the new fact already exists.
    :rtype: Optional[Set[str]]
    """
    if not all_constants or not all_facts_in_correct: return None
    fact_to_copy_str = random.choice(all_facts_in_correct)
    parsed = parse_fact(fact_to_copy_str)
    if not parsed: return None
    _, pred, consts = parsed
    if not consts: return None # Cannot modify constants of 0-ary predicate
    num_consts = len(consts)
    # Ensure random.choices doesn't get an empty list if all_constants is somehow empty despite check
    if not all_constants: return None
    
    chosen_raw_constants = random.choices(all_constants, k=num_consts)
    new_consts_processed = []
    for c_val in chosen_raw_constants:
        # If the constant is numeric or already quoted (single or double), use as is.
        # Otherwise, add double quotes. This addresses cases like unquoted names (e.g., Latoya)
        # that should be treated as string literals in the generated fact.
        if c_val.isdigit() or \
           (c_val.startswith('"') and c_val.endswith('"')) or \
           (c_val.startswith("'") and c_val.endswith("'")):
            new_consts_processed.append(c_val)
        else:
            new_consts_processed.append(f'"{c_val}"')

    new_negated = random.choice([True, False])
    new_fact = format_fact(new_negated, pred, new_consts_processed) # Use processed constants
    if new_fact in base_set: return None # Avoid adding existing fact
    return base_set | {new_fact}

def _generate_add_modified_predicate(base_set: Set[str], all_facts_in_correct: List[str], all_predicates: List[str]) -> Optional[Set[str]]:
    """
    Strategy 4: Add a fact (copied from any correct set) with a modified predicate to the base set.

    :param base_set: The set of facts in the base answer set to add to.
    :type base_set: Set[str]
    :param all_facts_in_correct: A flattened list of all facts appearing in any correct answer set.
    :type all_facts_in_correct: List[str]
    :param all_predicates: A list of all possible predicates available.
    :type all_predicates: List[str]
    :return: A new set with the added modified fact, or None if modification fails or the new fact already exists.
    :rtype: Optional[Set[str]]
    """
    if not all_predicates or not all_facts_in_correct: return None
    fact_to_copy_str = random.choice(all_facts_in_correct)
    parsed = parse_fact(fact_to_copy_str)
    if not parsed: return None
    _, _, consts = parsed
    # Ensure random.choice doesn't get an empty list
    if not all_predicates: return None
    new_pred = random.choice(all_predicates)
    new_negated = random.choice([True, False])
    new_fact = format_fact(new_negated, new_pred, consts)
    if new_fact in base_set: return None # Avoid adding existing fact
    return base_set | {new_fact}


def generate_incorrect_answer_sets(
    answer_sets: List[List[str]],
    name_map: Optional[Dict[Any, str]],
    pred_map: Optional[Dict[Any, str]],
    num_related_predicates: Optional[int],
    item_index: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generates up to 4 unique incorrect answer sets based on the correct ones using various strategies.

    :param answer_sets: List of correct answer sets (each a list of fact strings).
    :type answer_sets: List[List[str]]
    :param name_map: Mapping from name index to description (constants).
    :type name_map: Optional[Dict[Any, str]]
    :param pred_map: Mapping from predicate index to description.
    :type pred_map: Optional[Dict[Any, str]]
    :param num_related_predicates: Number of related predicates (used if pred_map is missing).
    :type num_related_predicates: Optional[int]
    :param item_index: Index of the current data item (for logging).
    :type item_index: int
    :return: A tuple containing:
             - List of generated incorrect answer set dictionaries.
             - Count of generated incorrect sets identical to correct sets (should be 0).
    :rtype: Tuple[List[Dict[str, Any]], int]
    """
    incorrect_sets_formatted = []
    item_duplicate_count = 0

    if not answer_sets:
        logging.warning(f"Item {item_index+1}: No correct answer sets found, cannot generate incorrect sets.")
        return [], 0
    # Removed the check for name_map here, as it's too restrictive for 0-ary predicates.
    # Strategies requiring name_map (constants) will check internally or be skipped later.

    all_constants = list(name_map.values()) if name_map else [] # Handle potential None for name_map
    all_predicates = list(pred_map.values()) if pred_map else []

    # Generate default predicates if map is missing/empty but count is available
    if not all_predicates and isinstance(num_related_predicates, int) and num_related_predicates > 0:
        all_predicates = [f"P{j}" for j in range(num_related_predicates)]
        logging.debug(f"Item {item_index+1}: Generated default predicates: {all_predicates}")
    elif not all_predicates:
        logging.warning(f"Item {item_index+1}: Predicate map/count missing or invalid. Cannot generate incorrect sets requiring predicate modification.")
        # Allow generation to proceed, but strategy 4 might fail/be skipped.

    correct_sets_frozen = {frozenset(ans_set) for ans_set in answer_sets}
    all_facts_in_correct_sets = [fact for ans_set in answer_sets for fact in ans_set]

    # Check if essential components for generation are missing
    # Strategy 3 needs all_constants, Strategy 4 needs all_predicates
    # Strategy 1 & 2 need base sets, Strategy 3 & 4 need all_facts_in_correct_sets
    if not all_facts_in_correct_sets:
         logging.warning(f"Item {item_index+1}: No facts derived from correct answer sets. Cannot generate incorrect sets.")
         return [], 0
    # Removed check for all_predicates here, as generation can proceed partially without it.

    generation_strategies = {
        1: lambda base_set, base_list: _generate_flip_negation(base_set, base_list),
        2: lambda base_set, base_list: _generate_delete_fact(base_set, base_list),
        3: lambda base_set, _: _generate_add_modified_constants(base_set, all_facts_in_correct_sets, all_constants),
        4: lambda base_set, _: _generate_add_modified_predicate(base_set, all_facts_in_correct_sets, all_predicates),
    }
    error_categories = {
        1: "Flip Negation", 2: "Delete Fact",
        3: "Add Modified Fact (Constants)", 4: "Add Modified Fact (Predicate)"
    }

    generated_incorrect_tuples: List[Tuple[frozenset[str], str]] = []
    generated_incorrect_frozensets: Set[frozenset[str]] = set()
    max_total_attempts = 2000 # Increased attempts significantly
    generation_attempts = 0

    # Determine base sets to use for modification
    num_answer_sets = len(answer_sets)
    if num_answer_sets == 0: return [], 0 # Should be caught earlier, but safety check
    target_base_indices = list(range(num_answer_sets))
    if num_answer_sets >= 4:
        base_indices_to_use = random.sample(target_base_indices, 4)
    else:
        base_indices_to_use = [idx % num_answer_sets for idx in range(4)]
        random.shuffle(base_indices_to_use)

    current_base_idx_pointer = 0

    while len(generated_incorrect_tuples) < 4 and generation_attempts < max_total_attempts:
        generation_attempts += 1
        strategy_num = random.randint(1, 4)
        # Skip strategy 4 if no predicates are available
        if strategy_num == 4 and not all_predicates:
            continue
        # Skip strategy 3 if no constants are available
        if strategy_num == 3 and not all_constants:
            continue

        base_set_idx = base_indices_to_use[current_base_idx_pointer]
        base_set_list = answer_sets[base_set_idx]
        base_set = set(base_set_list)

        try:
            strategy_func = generation_strategies[strategy_num]
            modified_set = strategy_func(base_set, base_set_list)

            if modified_set is not None:
                modified_set_frozen = frozenset(modified_set)
                # Check uniqueness against correct and already generated incorrect sets
                if modified_set_frozen not in correct_sets_frozen and \
                   modified_set_frozen not in generated_incorrect_frozensets:
                    error_category = error_categories[strategy_num]
                    generated_incorrect_tuples.append((modified_set_frozen, error_category))
                    generated_incorrect_frozensets.add(modified_set_frozen)
                    # Move to the next base set index for the next successful generation
                    current_base_idx_pointer = (current_base_idx_pointer + 1) % 4
                    # logging.debug(f"Item {item_index+1}: Added unique incorrect set {len(generated_incorrect_tuples)}/4. Category: {error_category}.")
                # else: # Generated set was not unique or matched a correct set
                    # logging.debug(f"Item {item_index+1}: Generated set not unique. Attempt {generation_attempts}.")

        except IndexError:
            # logging.warning(f"Item {item_index+1}: IndexError during generation attempt {generation_attempts}. Retrying.")
            continue
        except Exception as gen_e:
            logging.error(f"Item {item_index+1}: Unexpected error during incorrect set generation attempt {generation_attempts} (Strategy {strategy_num}): {gen_e}", exc_info=False)
            continue

    if len(generated_incorrect_tuples) < 4:
        logging.warning(f"Item {item_index+1}: Could only generate {len(generated_incorrect_tuples)} unique incorrect answer sets after {generation_attempts} attempts.")

    # Final check for duplicates (should ideally be prevented by generation logic)
    for incorrect_set_frozen, _ in generated_incorrect_tuples:
        if incorrect_set_frozen in correct_sets_frozen:
            item_duplicate_count += 1
            logging.error(f"Item {item_index+1}: CRITICAL - Found an incorrect answer set identical to a correct answer set despite checks! Set: {incorrect_set_frozen}")

    # Format output
    incorrect_sets_formatted = [
        {"answer_set": sorted(list(fs)), "error_category": category}
        for fs, category in generated_incorrect_tuples
    ]

    return incorrect_sets_formatted, item_duplicate_count


def _construct_options(
    answer_sets: List[List[str]],
    incorrect_answer_sets: List[Dict[str, Any]],
    item_index: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Constructs the list of 4 options (a mix of correct/incorrect answer sets) for the selection task.
    Handles sampling, combining, padding, and shuffling.

    :param answer_sets: List of correct answer sets (each a list of fact strings).
    :type answer_sets: List[List[str]]
    :param incorrect_answer_sets: List of generated incorrect answer set dictionaries (from generate_incorrect_answer_sets).
    :type incorrect_answer_sets: List[Dict[str, Any]]
    :param item_index: Index of the current data item (for logging).
    :type item_index: int
    :return: A tuple containing:
             - The final list of 4 option dictionaries.
             - The number of correct options included in the final list.
    :rtype: Tuple[List[Dict[str, Any]], int]
    """
    options = []
    correct_options_formatted = []
    incorrect_options_formatted = []

    # Determine number of correct options
    target_num_correct = random.randint(0, 4)
    actual_num_correct = 0 # Initialize

    # Sample correct options
    if answer_sets and target_num_correct > 0:
        num_to_sample_correct = min(target_num_correct, len(answer_sets))
        sampled_correct_indices = random.sample(range(len(answer_sets)), num_to_sample_correct)
        correct_options_formatted = [
            {"answer_set": sorted(list(answer_sets[idx])), "is_correct": True, "error_category": "Correct"}
            for idx in sampled_correct_indices
        ]
        actual_num_correct = len(correct_options_formatted) # Actual number sampled

    # Determine number of incorrect options needed
    num_incorrect_needed = 4 - actual_num_correct
    actual_num_incorrect = 0 # Initialize

    # Sample incorrect options
    if incorrect_answer_sets and num_incorrect_needed > 0:
        num_to_sample_incorrect = min(num_incorrect_needed, len(incorrect_answer_sets))
        sampled_incorrect_indices = random.sample(range(len(incorrect_answer_sets)), num_to_sample_incorrect)
        incorrect_options_formatted = [
            # Create a new dict to avoid modifying the original list items if padding occurs
            {"answer_set": incorrect_answer_sets[idx]["answer_set"], # Already sorted list
             "is_correct": False,
             "error_category": incorrect_answer_sets[idx]["error_category"]}
            for idx in sampled_incorrect_indices
        ]
        actual_num_incorrect = len(incorrect_options_formatted)

    # Combine sampled options
    options.extend(correct_options_formatted)
    options.extend(incorrect_options_formatted)

    # Pad if necessary to reach 4 options
    num_to_pad = 4 - len(options)
    if num_to_pad > 0:
        padding_source = incorrect_options_formatted if incorrect_options_formatted else correct_options_formatted
        if padding_source:
            logging.warning(f"Item {item_index+1}: Insufficient unique options ({len(options)}). Padding with {num_to_pad} duplicates.")
            padding = random.choices(padding_source, k=num_to_pad)
            options.extend(padding)
        else:
            # This case should be rare if generation works at all
            logging.error(f"Item {item_index+1}: Cannot create 4 options. No correct or incorrect options generated/sampled.")
            placeholder_option = {"answer_set": [], "is_correct": False, "error_category": "Padding Error"}
            options.extend([placeholder_option] * num_to_pad)

    # Shuffle the final list of 4 options
    if len(options) == 4:
        random.shuffle(options)
    else:
         logging.error(f"Item {item_index+1}: Final options list does not contain 4 elements ({len(options)}). Skipping shuffle.")

    return options, actual_num_correct # Return the actual number of correct options included


def _determine_answer_set_decision(
    answer_sets: List[List[str]],
    incorrect_answer_sets: List[Dict[str, Any]],
    item_index: int
) -> Dict[str, Any]:
    """
    Determines the final 'answer_set_decision' field by randomly choosing
    either a correct answer set or one of the generated incorrect answer sets.
    Handles cases where one or both types of sets might be unavailable.

    :param answer_sets: List of correct answer sets (each a list of fact strings).
    :type answer_sets: List[List[str]]
    :param incorrect_answer_sets: List of generated incorrect answer set dictionaries.
    :type incorrect_answer_sets: List[Dict[str, Any]]
    :param item_index: Index of the current data item (for logging).
    :type item_index: int
    :return: The dictionary representing the chosen answer set decision.
    :rtype: Dict[str, Any]
    """
    choice = random.randint(0, 1) # 0 for Correct, 1 for Incorrect
    decision = {'answerset': [], 'type': 'Error - No Sets Available'} # Default error state

    can_choose_correct = bool(answer_sets)
    can_choose_incorrect = bool(incorrect_answer_sets)

    if choice == 0: # Prefer Correct
        if can_choose_correct:
            chosen_set = random.choice(answer_sets)
            decision = {'answerset': sorted(chosen_set), 'type': 'Correct'}
        elif can_choose_incorrect: # Fallback to Incorrect
            chosen_incorrect_dict = random.choice(incorrect_answer_sets)
            decision = {'answerset': sorted(chosen_incorrect_dict['answer_set']), 'type': chosen_incorrect_dict['error_category']}
            logging.warning(f"Item {item_index+1}: Defaulted answer_set_decision to Incorrect (no correct options).")
        else: # No options available
            logging.error(f"Item {item_index+1}: Cannot set answer_set_decision - no correct or incorrect sets.")
    else: # Prefer Incorrect (choice == 1)
        if can_choose_incorrect:
            chosen_incorrect_dict = random.choice(incorrect_answer_sets)
            decision = {'answerset': sorted(chosen_incorrect_dict['answer_set']), 'type': chosen_incorrect_dict['error_category']}
        elif can_choose_correct: # Fallback to Correct
            chosen_set = random.choice(answer_sets)
            decision = {'answerset': sorted(chosen_set), 'type': 'Correct'}
            logging.warning(f"Item {item_index+1}: Defaulted answer_set_decision to Correct (no incorrect options).")
        else: # No options available
            logging.error(f"Item {item_index+1}: Cannot set answer_set_decision - no correct or incorrect sets.")

    return decision

# --- END NEW Helper Functions ---


def main(args: argparse.Namespace) -> None:
    """
    Main function to load fact state querying data, process it (initially just load),
    and prepare for saving the results.

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
    output_filename: str = f"answerset_selection_{input_filename_base}_combined.jsonl" # Changed prefix and added _combined
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

    # --- Placeholder for further processing ---
    # The rest of the script will now operate on 'combined_data' instead of 'all_data'
    logging.info("Combined data ready. Proceeding with answer set processing.")
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

    # --- Process each data entry ---
    processed_data_count = 0
    total_duplicate_incorrect_sets = 0 # Initialize counter for duplicates across all items
    # !!! IMPORTANT: Loop over combined_data now !!!
    for i, data in enumerate(combined_data):
        logging.debug(f"Processing item {i+1}/{len(combined_data)}") # Use len(combined_data)
        data['duplicate_incorrect_set_found'] = False # Initialize flag for the current item
        data['duplicate_incorrect_set_count'] = 0     # Initialize count for the current item
        if 'asp_program_dlv2' not in data:
            logging.warning(f"Skipping item {i+1} due to missing 'asp_program_dlv2' key.")
            data['num_answer_sets'] = -1 # Indicate skipped
            data['answer_set_fact_counts'] = []
            data['dlv2_error'] = "Missing 'asp_program_dlv2' key"
            continue

        # --- Extract Program Components using Helper ---
        asp_program = data['asp_program_dlv2']
        all_facts, all_rules = extract_program_components(asp_program, i)
        facts_list = sorted(list(all_facts))
        rules_list = sorted(list(all_rules))
        program_str = "\n".join(facts_list + rules_list) # Combine sorted lists

        # --- Store extracted facts/rules and calculate counts ---
        data['facts'] = facts_list
        data['rules'] = rules_list
        data['num_facts'] = len(facts_list)
        data['num_rules'] = len(rules_list)
        # Count rules containing default negation (' not ')
        data['num_rules_with_default_negation'] = sum(1 for rule in rules_list if ' not ' in rule)
        logging.debug(f"Item {i+1}: Stored {data['num_facts']} facts, {data['num_rules']} rules ({data['num_rules_with_default_negation']} with ' not ').")

        # Add comment for clarity in logs if needed
        # logging.debug(f"Running DLV2 for item {i+1} with program:\n{program_str[:500]}...") # Log first 500 chars

        try: # <<< Outer try block starts here (Line 294 approx) >>>
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

                # --- Generate Incorrect Answer Sets using Helper ---
                incorrect_sets_formatted, item_duplicate_count = generate_incorrect_answer_sets(
                    answer_sets=answer_sets,
                    name_map=data.get('name_idx_to_desc'),
                    pred_map=data.get('predicate_idx_to_desc'),
                    num_related_predicates=data.get('num_related_predicates'),
                    item_index=i
                )
                data['incorrect_answer_sets'] = incorrect_sets_formatted
                if item_duplicate_count > 0:
                    total_duplicate_incorrect_sets += item_duplicate_count
                    data['duplicate_incorrect_set_found'] = True
                    data['duplicate_incorrect_set_count'] = item_duplicate_count

                # --- Construct the 'options' field using Helper ---
                options, actual_num_correct_options = _construct_options(
                    answer_sets=answer_sets,
                    incorrect_answer_sets=data['incorrect_answer_sets'],
                    item_index=i
                )
                data['options'] = options
                data['num_correct_options_in_final_set'] = actual_num_correct_options # Store the actual count

                # --- Add answer_set_decision field using Helper ---
                data['answer_set_decision'] = _determine_answer_set_decision(
                    answer_sets=data.get('answer_sets', []),
                    incorrect_answer_sets=data.get('incorrect_answer_sets', []),
                    item_index=i
                )
            # <<< Outer try block ends here >>>

        # <<< Correctly indented except blocks for the outer try >>>
        except Dlv2RunnerError as e:
            logging.error(f"Item {i+1}: DLV2 execution failed: {e}")
            data['num_answer_sets'] = -1
            data['answer_set_fact_counts'] = []
            data['dlv2_error'] = str(e)
            data['answerset_selection_dlv2_result_details'] = {'success': False, 'error_message': str(e)} # Store error details
        except Exception as e: # Catch any other unexpected errors during processing
            logging.error(f"Item {i+1}: An unexpected error occurred during processing: {e}", exc_info=True)
            data['num_answer_sets'] = -1
            data['answer_set_fact_counts'] = []
            data['dlv2_error'] = f"Unexpected error: {e}"
            data['answerset_selection_dlv2_result_details'] = {'success': False, 'error_message': f"Unexpected error: {e}"} # Store error details

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

    # --- Print Duplicate Statistics ---
    logging.info(f"--- Incorrect vs Correct Answer Set Comparison Statistics ---")
    logging.info(f"Total instances where a generated incorrect answer set was identical to a correct answer set: {total_duplicate_incorrect_sets}")
    # !!! IMPORTANT: Check combined_data for duplicate flags !!!
    items_with_duplicates = sum(1 for d in combined_data if d.get('duplicate_incorrect_set_found', False))
    logging.info(f"Number of data items containing at least one such duplicate: {items_with_duplicates}")
    logging.info(f"--- End Comparison Statistics ---")

    # --- Save Processed Data ---
    # Use the already determined output_file_path which includes '_combined'
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Timestamp is good, but base name is already set
    # output_filename_processed: str = f"answerset_selection_{input_filename_base}_combined_{timestamp}.jsonl" # Add timestamp if desired
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
        description="Constructs a dataset for fact/state querying tasks. Reads a JSONL input file and prepares an output file."
    )

    # Input/Output Arguments
    parser.add_argument(
        "--input_path_w_disj", # Changed from --input_path
        type=str,
        required=True, # Make required as there's no single default anymore
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
        default='./datasets/a_symtex_task_answerset_selection',
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
