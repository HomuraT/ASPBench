#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to check SymTex samples from a JSONL file for DLV2 safety errors.
Attempts to fix errors by modifying rule negations. Skips samples with
unfixable errors.
"""

import argparse
import os
import logging
import pathlib
import copy
import itertools
from typing import Dict, Optional, List, Tuple, Any, Set, FrozenSet

# Attempt to use orjson for faster JSON handling, fallback to standard json
try:
    import orjson as json
except ImportError:
    import json

from tqdm import tqdm

# Assuming dlv2_runner and formatting functions are accessible via src path
# Adjust imports based on your project structure if necessary
try:
    from src.utils.dlv2_runner import Dlv2Runner, Dlv2RunnerError
    from src.dataset_generation.symtex import _format_atom_dlv2, _format_rule_dict_to_dlv2
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure the script is run from the project root or the src directory is in PYTHONPATH.")
    exit(1)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Negation flags (assuming these values are consistent with symtex.py)
STRONG_NEGATION_VALUE = 2
DEFAULT_NEGATION_VALUE = 4


# --- Core Logic ---

def validate_and_modify_rule(
    rule_dict: Dict[str, Any],
    dlv2_runner: Optional[Dlv2Runner],
    idx_to_name: Dict[int, str],
    use_disjunction: bool, # Typically False for safety checks
    rule_context_info: str
) -> Optional[Dict[str, Any]]:
    """
    Checks a rule dictionary for DLV2 safety errors and attempts modifications.

    Replicates the logic from symtex.py's _validate_and_modify_rule_for_safety.

    :param rule_dict: The rule dictionary to validate (MODIFIED IN PLACE if fixable).
    :param dlv2_runner: The Dlv2Runner instance (can be None).
    :param idx_to_name: Mapping from predicate index to name.
    :param use_disjunction: Flag for formatting rules (usually False for safety).
    :param rule_context_info: String for logging context.
    :return: The validated (potentially modified) rule dictionary, or None if unfixable.
    """
    if not dlv2_runner:
        logger.warning(f"DLV2 runner not available. Skipping safety check for {rule_context_info}.")
        return rule_dict # Skip checks if runner is not available

    initial_dlv2_rules = _format_rule_dict_to_dlv2(rule_dict, idx_to_name, use_disjunction)
    if not initial_dlv2_rules: # Handle empty rules (e.g., empty head)
        logger.warning(f"Skipping empty or invalid rule structure for {rule_context_info}.")
        return rule_dict # Consider it valid if structure is just empty? Or None? Let's return original for now.

    initial_check_failed = False
    safety_error_found = False
    error_message = ""

    # Initial check
    for rule_str in initial_dlv2_rules:
        result = dlv2_runner.run(rule_str)
        if not result['success']:
            initial_check_failed = True
            error_message = result['error_message'] or "Unknown DLV2 error"
            if "Safety Error" in error_message:
                safety_error_found = True
                logger.debug(f"Safety Error found in {rule_context_info}: {rule_str}")
                break # Stop checking this rule dict if safety error found
            else:
                logger.debug(f"Other Error found in {rule_context_info}: {rule_str} -> {error_message}")
                # Treat other errors as potentially problematic too? For now, only fix safety errors.
                # If we want to skip on *any* error, return None here.
                # Let's assume only safety errors are targeted for fixing.
                pass # Continue checking other parts if not a safety error? Or break? Let's break on first error.
                break

    if not initial_check_failed:
        return rule_dict # Rule is fine as is

    if not safety_error_found:
        logger.warning(f"Non-safety DLV2 error encountered for {rule_context_info}, skipping modification attempts: {error_message}")
        # Decide policy: skip sample (return None) or keep original (return rule_dict)?
        # Let's be strict and skip the sample if *any* DLV2 error occurs that isn't a fixable safety error.
        return None # Indicate failure for non-safety errors too

    # --- Attempt Modification for Safety Error ---
    logger.debug(f"Attempting modification for safety error in {rule_context_info}...")
    candidates = []
    original_body = rule_dict.get('body', [])
    for i, pred_dict in enumerate(original_body):
        # Candidate must have default negation set
        if pred_dict.get('default negation', False):
            candidates.append(i)

    if not candidates:
        logger.warning(f"Safety error in {rule_context_info}, but no modification candidates (default negation in body) found. Rule cannot be fixed.")
        return None # Cannot fix

    modification_successful = False
    # No deepcopy here, modify rule_dict directly

    # Strategy 1: Try flipping one candidate
    logger.debug(f"Modification candidates (indices): {candidates}")
    for candidate_idx in candidates:
        pred_to_modify = rule_dict['body'][candidate_idx]
        # Store original state before modification
        original_default_neg = pred_to_modify.get('default negation', False)
        original_strong_neg = pred_to_modify.get('strong negation', False)

        # Flip negation flags directly on rule_dict
        pred_to_modify['default negation'] = not original_default_neg
        pred_to_modify['strong negation'] = not original_strong_neg
        logger.debug(f"Trying modification (Strategy 1): Flipping predicate at index {candidate_idx}")

        modified_dlv2_rules_attempt = _format_rule_dict_to_dlv2(rule_dict, idx_to_name, use_disjunction) # Check modified rule_dict
        modification_check_passed = True
        for rule_str_mod in modified_dlv2_rules_attempt:
            result_mod = dlv2_runner.run(rule_str_mod)
            if not result_mod['success']:
                modification_check_passed = False
                logger.debug(f"Modification failed check: {rule_str_mod} -> {result_mod['error_message']}")
                # --- REVERT CHANGE on failure ---
                pred_to_modify['default negation'] = original_default_neg
                pred_to_modify['strong negation'] = original_strong_neg
                logger.debug(f"Reverted changes for predicate at index {candidate_idx}")
                # --- End Revert ---
                break # Stop checking this modification attempt

        if modification_check_passed:
            logger.debug(f"Modification successful (Strategy 1 - flipped index {candidate_idx}) for {rule_context_info}!")
            # No need to adopt, rule_dict is already modified
            modification_successful = True
            break # Exit strategy 1 loop
        # else: Modification failed, changes were reverted above

    # Strategy 2: Try flipping two candidates (if strategy 1 failed and >= 2 candidates)
    if not modification_successful and len(candidates) >= 2:
        logger.debug("Strategy 1 failed. Trying Strategy 2 (flipping pairs)...")
        for combo in itertools.combinations(candidates, 2):
            idx1, idx2 = combo

            # Get predicates and store original states
            pred1 = rule_dict['body'][idx1]
            orig_def1 = pred1.get('default negation', False)
            orig_str1 = pred1.get('strong negation', False)
            pred2 = rule_dict['body'][idx2]
            orig_def2 = pred2.get('default negation', False)
            orig_str2 = pred2.get('strong negation', False)

            # Flip negation flags directly on rule_dict
            pred1['default negation'] = not orig_def1
            pred1['strong negation'] = not orig_str1
            pred2['default negation'] = not orig_def2
            pred2['strong negation'] = not orig_str2
            logger.debug(f"Trying modification (Strategy 2): Flipping predicates at indices {idx1} and {idx2}")

            modified_dlv2_rules_attempt = _format_rule_dict_to_dlv2(rule_dict, idx_to_name, use_disjunction) # Check modified rule_dict
            modification_check_passed = True
            for rule_str_mod in modified_dlv2_rules_attempt:
                result_mod = dlv2_runner.run(rule_str_mod)
                if not result_mod['success']:
                    modification_check_passed = False
                    logger.debug(f"Modification failed check: {rule_str_mod} -> {result_mod['error_message']}")
                    # --- REVERT CHANGES on failure ---
                    pred1['default negation'] = orig_def1
                    pred1['strong negation'] = orig_str1
                    pred2['default negation'] = orig_def2
                    pred2['strong negation'] = orig_str2
                    logger.debug(f"Reverted changes for predicates at indices {idx1} and {idx2}")
                    # --- End Revert ---
                    break

            if modification_check_passed:
                logger.debug(f"Modification successful (Strategy 2 - flipped indices {idx1}, {idx2}) for {rule_context_info}!")
                # No need to adopt, rule_dict is already modified
                modification_successful = True
                break # Exit strategy 2 loop
            # else: Modification failed, changes were reverted above

    if modification_successful:
        return rule_dict # Return the modified rule_dict
    else:
        logger.warning(f"Modification attempts failed for safety error in {rule_context_info}. Rule cannot be fixed.")
        # Ensure rule_dict is in its original state before returning None
        # (It should be due to reverts, but maybe add a final check/revert if needed, though complex)
        return None # Indicate failure


def process_sample(sample_dict: Dict[str, Any], dlv2_runner: Optional[Dlv2Runner], sample_index: int) -> Optional[Dict[str, Any]]:
    """
    Processes a single sample dictionary, checking and potentially modifying its rules.

    :param sample_dict: The dictionary representing the sample.
    :param dlv2_runner: The Dlv2Runner instance.
    :param sample_index: The index of the sample in the input file (for logging).
    :return: The original sample dictionary (potentially modified) if valid, otherwise None.
    """
    # sample_copy = copy.deepcopy(sample_dict) # REMOVED: Work directly on sample_dict
    is_valid = True

    try:
        # Operate directly on sample_dict
        original_dicts = sample_dict.get('dict_structure', {})
        noiseless_rules = original_dicts.get('noiseless_rules', [])
        noisy_rules_by_type = original_dicts.get('noisy_rules', {}) # Dict[str, List[Dict]]
        all_facts = original_dicts.get('noiseless_facts', []) + original_dicts.get('noisy_facts', [])

        # --- Determine Predicate Names and idx_to_name mapping ---
        # Collect all predicate indices used in this sample
        all_predicate_indices: Set[int] = set()
        for fact in all_facts:
            all_predicate_indices.add(fact['predicateIdx'])
        for rule in noiseless_rules:
            for p in rule.get('head', []) + rule.get('body', []):
                all_predicate_indices.add(p['predicateIdx'])
        for noise_type, rules in noisy_rules_by_type.items():
            for rule in rules:
                 for p in rule.get('head', []) + rule.get('body', []):
                    all_predicate_indices.add(p['predicateIdx'])

        if not all_predicate_indices:
             max_idx = -1
        else:
             max_idx = max(all_predicate_indices)

        # Create mapping (assuming P0, P1... naming if not provided otherwise)
        # We don't have the original predicate list here, so generate defaults.
        idx_to_name = {i: f"P{i}" for i in range(max_idx + 1)}

        # --- Validate Noiseless Rules ---
        validated_noiseless_rules = []
        for i, rule_dict in enumerate(noiseless_rules):
            context = f"sample {sample_index}, noiseless rule {i}"
            validated_rule = validate_and_modify_rule(rule_dict, dlv2_runner, idx_to_name, False, context)
            if validated_rule is None:
                is_valid = False
                logger.warning(f"Skipping sample {sample_index} due to unfixable error in noiseless rule {i}.")
                break # Stop processing this sample
            # validated_noiseless_rules.append(validated_rule) # No need to append, rule_dict is modified in place
            pass # Rule was modified in place if successful
        if not is_valid: return None
        # original_dicts['noiseless_rules'] = validated_noiseless_rules # No need to update, list contains modified dicts

        # --- Validate Noisy Rules ---
        # validated_noisy_rules_by_type = {} # No need for a new dict
        for noise_type, rules in noisy_rules_by_type.items():
            # validated_rules_for_type = [] # No need for a new list
            for i, rule_dict in enumerate(rules): # Iterate directly over rules in original_dicts
                context = f"sample {sample_index}, noisy rule type {noise_type}, index {i}"
                validated_rule = validate_and_modify_rule(rule_dict, dlv2_runner, idx_to_name, False, context) # Modify in place
                if validated_rule is None:
                    is_valid = False
                    logger.warning(f"Skipping sample {sample_index} due to unfixable error in noisy rule type {noise_type}, index {i}.")
                    break # Stop processing this type
                # validated_rules_for_type.append(validated_rule) # No need to append

            if not is_valid: break # Stop processing this sample
            # validated_noisy_rules_by_type[noise_type] = validated_rules_for_type # No need to update
        if not is_valid: return None
        # original_dicts['noisy_rules'] = validated_noisy_rules_by_type # No need to update

        # --- Final Check: Validate the entire program ---
        if dlv2_runner:
            logger.debug(f"Performing final DLV2 check for the complete program of sample {sample_index}...")
            final_asp_strings = []
            # Re-format facts (assuming facts don't change)
            for fact_dict in all_facts:
                 # Need to handle potential variable mapping for facts if they use integers
                 # For simplicity, assume facts use constants or _format_atom_dlv2 handles it
                 final_asp_strings.append(_format_atom_dlv2(fact_dict, idx_to_name, is_fact=True) + ".")

            # Re-format potentially modified rules
            for rule_dict in noiseless_rules:
                 final_asp_strings.extend(_format_rule_dict_to_dlv2(rule_dict, idx_to_name, False)) # Use False for use_disjunction
            for noise_type, rules in noisy_rules_by_type.items():
                 for rule_dict in rules:
                      final_asp_strings.extend(_format_rule_dict_to_dlv2(rule_dict, idx_to_name, False))

            full_program_text = "\n".join(final_asp_strings)
            final_result = dlv2_runner.run(full_program_text)

            if not final_result.get('success'):
                logger.warning(f"Skipping sample {sample_index} due to failure in final full program DLV2 check. Error: {final_result.get('error_message', 'Unknown')}")
                # Log the program that failed for debugging
                logger.debug(f"Failed program for sample {sample_index}:\n{full_program_text}")
                return None # Sample is invalid if the full program fails
            else:
                logger.debug(f"Final full program check passed for sample {sample_index}.")
        else:
             logger.debug(f"Skipping final full program check for sample {sample_index} as DLV2 runner is not available.")


        # If we reach here, the sample is valid (all rules passed/fixed AND final check passed)
        # The original sample_dict has been modified in place.
        return sample_dict

    except Exception as e:
        logger.error(f"Error processing sample {sample_index}: {e}", exc_info=True)
        return None # Skip sample on unexpected error


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check and fix DLV2 safety errors in SymTex JSONL samples.")
    parser.add_argument("-i", "--input-file", 
                        default='datasets/symtex_select_from_filter_data/2025_04_21_21_01/20250421_210052_from_2025_04_19_18_50_seed42_n1000_minprob0.8.jsonl'
                        , help="Path to the input JSONL file.")
    parser.add_argument("-o", "--output-dir", default='datasets/symtex_after_check', help="Path to the output directory.")
    parser.add_argument("--dlv2-path", default=None, help="Optional path to the DLV2 executable.")

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_file)
    output_dir = pathlib.Path(args.output_dir)

    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct output file path
    output_filename = f"checked_{input_path.name}"
    output_path = output_dir / output_filename

    # Initialize DLV2 Runner
    dlv2_runner: Optional[Dlv2Runner] = None
    try:
        dlv2_runner = Dlv2Runner()
    except Dlv2RunnerError as e:
        logger.warning(f"Failed to initialize Dlv2Runner: {e}. Safety checks will be skipped.")
    except Exception as e:
        logger.error(f"Unexpected error initializing Dlv2Runner: {e}", exc_info=True)
        # Decide if we should proceed without checks or exit
        logger.warning("Proceeding without DLV2 safety checks.")


    processed_count = 0
    written_count = 0

    logger.info(f"Starting processing of {input_path}")
    logger.info(f"Output will be written to {output_path}")

    try:
        # Estimate number of lines for tqdm
        try:
            with open(input_path, 'rb') as f:
                num_lines = sum(1 for _ in f)
        except Exception:
            num_lines = None # Unable to estimate

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for i, line in enumerate(tqdm(infile, total=num_lines, desc="Processing samples")):
                processed_count += 1
                try:
                    # Use loads from the imported json module (could be orjson)
                    sample_data = json.loads(line)
                except Exception as e:
                    logger.error(f"Failed to parse JSON on line {i+1}: {e}. Skipping line.")
                    continue

                # Process the sample
                valid_sample = process_sample(sample_data, dlv2_runner, i + 1)

                if valid_sample:
                    # Use dumps from the imported json module
                    try:
                         # For standard json:
                         # json.dump(valid_sample, outfile)
                         # outfile.write('\n')

                         # For orjson:
                         if hasattr(json, 'dumps') and hasattr(json, 'OPT_APPEND_NEWLINE'):
                              # Efficient orjson way
                              outfile.write(json.dumps(valid_sample, option=json.OPT_APPEND_NEWLINE).decode('utf-8'))
                         else:
                              # Fallback using standard json or orjson without newline option
                              json_str = json.dumps(valid_sample)
                              outfile.write(json_str + '\n')

                         written_count += 1
                    except Exception as e:
                         logger.error(f"Failed to write processed sample {i+1} to output file: {e}")


    except FileNotFoundError:
        logger.error(f"Input file not found during processing: {input_path}")
    except IOError as e:
        logger.error(f"IOError during file processing: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)

    logger.info(f"Processing complete.")
    logger.info(f"Total samples processed: {processed_count}")
    logger.info(f"Samples written to output (passed checks or fixed): {written_count}")
    logger.info(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
