# 06_transfer_dict_to_dlv2.py

import argparse
from pathlib import Path
import sys
import os # Added import
import json
import logging
import random
import re # Added for cleaning names
import networkx as nx # Added import
from typing import List, Dict, Any, Optional, Set, Tuple, Union # Added Set, Tuple, Union
from datetime import datetime
from collections import Counter # Added for error counting
import copy # Added for deep copying structures
from faker import Faker
from functools import partial

from src.conceptnet_utils.graph_operations import get_random_node
from src.conceptnet_utils.storage import load_graph_from_graphml
from src.utils.asp_classification import ASPProgramAnalyzer

from src.utils.sparse_utils import dense_to_sparse_serializable, sparse_serializable_to_dense

# --- Add project root to sys.path ---
# This allows importing modules from 'src' when running the script directly.
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Detected project root: {project_root}") # DEBUG Print
# If the script is in the root, project_root is correct.
# If the script is moved (e.g., to a 'scripts' folder), adjust accordingly:
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    print(f"[DEBUG] Adding {project_root} to sys.path.") # DEBUG Print
    sys.path.insert(0, project_root)
else:
    print(f"[DEBUG] {project_root} already in sys.path.") # DEBUG Print
print(f"[DEBUG] Current sys.path: {sys.path}") # DEBUG Print
# --- End sys.path modification ---


# Import functions from utils and dataset_generation
try:
    # Now imports should work
    from src.utils.json_utils import read_jsonl_parallel, write_jsonl, JsonUtilsError
    # Import both formatting functions
    from src.dataset_generation.asp_formatter import dict_to_asp_strings, format_dict_structure_to_asp
    # Imports for ConceptNet functionality - trying direct import
    from src.dataset_generation.index_name_mapper import (
        generate_unique_conceptnet_names,
        extract_concept_from_uri,
        complete_uris_by_edge
    )
    # Import DLV2 runner separately as it might have different dependencies or reasons to fail
    from src.utils.dlv2_runner import Dlv2Runner, Dlv2RunnerError # Added for DLV2 execution
except ImportError as e:
    print(f"Error: Could not import required functions/modules: {e}", file=sys.stderr)
    print("Ensure the 'src' directory is in your Python path, dependencies are installed, and required modules exist.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

ASP_KEYWORDS = {'not', 'count', 'sum', 'max', 'min'} # Add other keywords like 'count', 'sum', 'max', 'min' if needed, lowercase


def _remove_negations(data: Any) -> Any:
    """
    Recursively traverses a nested structure (dicts, lists) and sets
    'strong negation' and 'default negation' keys to False if they are True.

    :param data: The input data structure (dict, list, or other).
    :type data: Any
    :return: The modified data structure.
    :rtype: Any
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if k == 'strong negation' and v is True:
                new_dict[k] = False
            elif k == 'default negation' and v is True:
                new_dict[k] = False
            else:
                new_dict[k] = _remove_negations(v) # Recurse on value
        return new_dict
    elif isinstance(data, list):
        return [_remove_negations(item) for item in data] # Recurse on list items
    else:
        return data # Return non-dict/list items as is


def _normalize_predicate_name(name: Optional[str]) -> Optional[str]:
    """
    Cleans and normalizes a predicate name.
    1. Removes characters other than letters, numbers, and underscores.
    2. If the cleaned name starts with digits, moves the leading digits (and optional following underscore)
       to the end, separated by an underscore. e.g., "2_ears" -> "ears_2", "3birds" -> "birds_3".

    :param name: The original predicate name.
    :type name: Optional[str]
    :return: The cleaned and normalized name, or None if the input is None or becomes empty after cleaning.
    :rtype: Optional[str]
    """
    if not name:
        return None
    # 1. Remove invalid characters
    cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    if not cleaned_name: # Handle case where cleaning results in empty string
        return None

    # 2. Move leading digits to the end
    match = re.match(r'^(\d+)(_?)(.*)$', cleaned_name)
    if match:
        digits = match.group(1)
        underscore = match.group(2) # '_' or ''
        rest = match.group(3)
        if rest: # Only move if there's a non-digit part
            # Ensure single underscore separator
            if rest.endswith('_'):
                 # e.g., 12_abc_ -> abc_12
                 normalized_name = f"{rest}{digits}"
            elif underscore == '_':
                 # e.g., 12_abc -> abc_12
                 normalized_name = f"{rest}_{digits}"
            else:
                 # e.g., 12abc -> abc_12
                 normalized_name = f"{rest}_{digits}"
            # This return should be inside the 'if rest:' block
            return normalized_name
        else: # Name was only digits (e.g., "123"), directly use cleaned_name
            normalized_name = cleaned_name
    else: # Doesn't start with digits
        normalized_name = cleaned_name

    # 3. Check against ASP keywords (case-insensitive check, but keywords are stored lowercase)
    if normalized_name.lower() in ASP_KEYWORDS:
        normalized_name += "_"
        logging.debug(f"Normalized name '{cleaned_name}' was an ASP keyword, appended underscore: '{normalized_name}'")

    return normalized_name # Ensure the processed name is always returned

def process_item(item: Dict[str, Any], use_disjunction: bool, graph: Optional[nx.DiGraph], args: argparse.Namespace, runner: Dlv2Runner) -> \
tuple[dict[str, Any], str | None, bool]: # Removed is_positive and use_noisy parameters
    """
    Processes a single JSON object (item) from the input file.
    Generates predicate names based on args flags (default, random, related).
    Probabilistically determines whether to remove noise (based on args.noise_removal_prob)
    and whether to remove negations and modify graph data (based on args.negation_removal_prob).
    Generates fact variable names using Faker.
    Creates a new field 'asp_program_dlv2' containing DLV2 strings.
    Runs the generated program using DLV2 and stores the result.
    Removes intermediate and original dictionary structures.

    :param item: The input dictionary representing a single data entry.
    :type item: Dict[str, Any]
    :param use_disjunction: Whether to use disjunction ('|') for multi-head rules.
    :type use_disjunction: bool
    :param graph: The loaded ConceptNet graph (required if using ConceptNet name generation).
    :type graph: Optional[nx.DiGraph]
    :param args: Command-line arguments.
    :type args: argparse.Namespace
    :param runner: An initialized Dlv2Runner instance.
    :type runner: Dlv2Runner
    # Removed is_positive and use_noisy from docstring
    :return: A tuple containing the processed dictionary and an optional string indicating the DLV2 error type.
    :rtype: Tuple[Dict[str, Any], Optional[str]]
    """
    # --- Determine whether to remove negations and noise ---
    # 1. Determine if negations should be removed (is_positive)
    is_positive = random.random() < args.negation_removal_prob

    # 2. Determine if noise should be kept (use_noisy)
    # If is_positive is True, force noise removal (use_noisy = False)
    if is_positive:
        use_noisy = False
        logging.debug(f"Item {item.get('id', 'N/A')}: Determined is_positive=True (prob={args.negation_removal_prob:.2f}). Forcing noise removal (use_noisy=False).")
    else:
        # If is_positive is False, determine noise removal based on probability
        # use_noisy means "keep noisy data". We remove noise if random() < noise_removal_prob.
        # So, use_noisy (keep noise) is True if random() >= noise_removal_prob.
        use_noisy = not (random.random() < args.noise_removal_prob)
        logging.debug(f"Item {item.get('id', 'N/A')}: Determined is_positive=False (prob={args.negation_removal_prob:.2f}). Determined use_noisy={use_noisy} (noise removal prob={args.noise_removal_prob:.2f}).")

    # Use these locally determined boolean variables throughout the function.

    fake = Faker() # Instantiate Faker for fact variable names
    dlv2_error_type: Optional[str] = None # Initialize error type for this item
    original_idx_to_name: Dict[int, str] = item.get('idx_to_name', {})
    # Store the original truth value from the input item
    original_target_query_in_answerset_value = item.get('target_query_in_answerset', None)
    # Initialize the ground truth for consistency checks
    current_ground_truth = original_target_query_in_answerset_value
    rule_var_idx_to_name: Optional[Dict[int, str]] = item.get('var_idx_to_name', None) # For rule variables like V1, V2
    # Make a deep copy to potentially modify without affecting the original item dict
    current_structure: Dict[str, Any] = copy.deepcopy(item.get('dict_structure', {}))

    # --- 1. Optionally remove noisy data ---
    if not use_noisy:
        logging.debug(f"use_noisy is False. Removing noisy facts and rules for item {item.get('id', 'N/A')}.")
        current_structure['noisy_facts'] = []
        current_structure['noisy_rules'] = {}
        # Also remove noisy graph data if it exists in the original item, before it gets processed later
        # Pop the correct key first, then the potential typo
        item.pop('noisy_rule_predicate_operation_graph', None)
        item.pop('noisy_idx2type', None)


    # --- 2. Optionally remove negations from the structure ---
    if is_positive:
        logging.debug(f"is_positive is True. Removing negations from the current structure for item {item.get('id', 'N/A')}.")
        # Apply negation removal to the potentially noise-filtered structure
        processed_structure = _remove_negations(current_structure) # Modifies the current_structure copy
    else:
        processed_structure = current_structure # Use the potentially noise-filtered structure

    # --- 3. Generate Predicate Names (final_idx_to_name) ---
    # Based on the original names, but applied to the processed structure
    final_idx_to_name: Dict[int, str] = {}
    predicate_indices: List[int] = list(original_idx_to_name.keys())

    if args.use_related_conceptnet_predicates:
        if graph is None:
            logging.error("ConceptNet graph is required for --use_related_conceptnet_predicates but was not loaded. Skipping item.")
            # Return item unmodified or with an error marker? Let's skip for now.
            # Alternatively, fall back to default? Let's log error and use default.
            logging.debug("Falling back to original predicate names.") # Changed to debug
            final_idx_to_name = original_idx_to_name
        else:
            logging.debug(f"Generating related predicate names for indices: {predicate_indices}")
            # a) Extract rule structure for complete_uris_by_edge using the potentially modified structure
            rule_indices_list: List[Dict[str, List[int]]] = []
            raw_noiseless_rules = processed_structure.get('noiseless_rules', []) # Use processed_structure
            if isinstance(raw_noiseless_rules, list):
                 rule_indices_list.extend([
                     {'head': [h['predicateIdx'] for h in r.get('head', []) if isinstance(h, dict) and 'predicateIdx' in h], # Added isinstance check
                      'body': [b['predicateIdx'] for b in r.get('body', []) if isinstance(b, dict) and 'predicateIdx' in b]} # Added isinstance check
                     for r in raw_noiseless_rules if isinstance(r, dict)
                 ])
            raw_noisy_rules = processed_structure.get('noisy_rules', {}) # Use processed_structure
            if isinstance(raw_noisy_rules, dict):
                 for rule_list in raw_noisy_rules.values():
                      if isinstance(rule_list, list):
                           rule_indices_list.extend([
                               {'head': [h['predicateIdx'] for h in r.get('head', []) if isinstance(h, dict) and 'predicateIdx' in h], # Added isinstance check
                                'body': [b['predicateIdx'] for b in r.get('body', []) if isinstance(b, dict) and 'predicateIdx' in b]} # Added isinstance check
                               for r in rule_list if isinstance(r, dict)
                           ])
            # Removed duplicated lines from here - This section seems redundant as processed_structure is used above.
            # raw_noisy_rules =  original_structure.get('noisy_rules', {}) # This line caused the NameError
            # if isinstance(raw_noisy_rules, dict):
            #      for rule_list in raw_noisy_rules.values():
            #           if isinstance(rule_list, list):
            #                rule_indices_list.extend([
            #                    {'head': [h['predicateIdx'] for h in r.get('head', []) if 'predicateIdx' in h],
            #                     'body': [b['predicateIdx'] for b in r.get('body', []) if 'predicateIdx' in b]}
            #                    for r in rule_list if isinstance(r, dict)
            #                ])

            # b) Determine all unique predicate indices involved in rules AND original map
                           rule_indices_list.extend([
                               {'head': [h['predicateIdx'] for h in r.get('head', []) if 'predicateIdx' in h],
                                'body': [b['predicateIdx'] for b in r.get('body', []) if 'predicateIdx' in b]}
                               for r in rule_list if isinstance(r, dict)
                           ])

            # b) Determine all unique predicate indices involved in rules AND original map
            all_predicate_indices: Set[int] = set(predicate_indices)
            for d in rule_indices_list:
                all_predicate_indices.update(d.get('head', []))
                all_predicate_indices.update(d.get('body', []))
            all_predicate_indices_list = sorted(list(all_predicate_indices)) # For consistency

            # c) Implement retry logic for complete_uris_by_edge
            uris: Dict[int, str] = {}
            max_retries = len(all_predicate_indices_list) * 2 + 5 # Heuristic limit
            retries = 0
            assigned_uris_values: Set[str] = set() # Track assigned URIs to ensure uniqueness when randomly assigning

            # Start with a random node if possible and graph exists
            if all_predicate_indices_list and graph:
                 start_idx = random.choice(all_predicate_indices_list)
                 # Use full path for imported function
                 initial_uri = get_random_node(graph)
                 if initial_uri:
                     uris[start_idx] = initial_uri
                     assigned_uris_values.add(initial_uri)
                     logging.debug(f"Initial random URI for index {start_idx}: {initial_uri}")
                 else:
                     logging.debug("Could not get initial random node.") # Changed to debug


            while not all_predicate_indices.issubset(uris.keys()) and retries < max_retries:
                retries += 1
                logging.debug(f"Related names retry {retries}/{max_retries}. Current URIs: {len(uris)}/{len(all_predicate_indices)}")
                complete_uris_by_edge(graph, rule_indices_list, uris) # Modify uris in-place

                missing_indices = all_predicate_indices - uris.keys()
                if not missing_indices:
                    logging.debug("All predicate indices have URIs after complete_uris_by_edge.")
                    break

                # If still missing, pick one missing index and assign a random *unique* URI
                index_to_assign = random.choice(list(missing_indices))
                assigned_uris_values.update(uris.values()) # Update known URIs
                new_uri = None
                find_attempts = 0
                max_find_attempts = graph.number_of_nodes() # Limit attempts

                while find_attempts < max_find_attempts and graph: # Check graph exists
                    find_attempts += 1
                    # Use full path for imported function
                    candidate_uri = get_random_node(graph)
                    if candidate_uri and candidate_uri not in assigned_uris_values:
                        new_uri = candidate_uri
                        break

                if new_uri:
                    uris[index_to_assign] = new_uri
                    assigned_uris_values.add(new_uri)
                    logging.debug(f"Retry {retries}: Randomly assigned URI '{new_uri}' to missing index {index_to_assign}")
                else:
                    logging.debug(f"Retry {retries}: Could not find a unique random URI for index {index_to_assign} after {max_find_attempts} attempts. Stopping retry loop.") # Changed to debug
                    break # Stop if we cannot find a unique URI

            # d) Convert URIs to names
            temp_name_map: Dict[int, str] = {}
            assigned_names: Set[str] = set()
            for idx, uri in uris.items():
                 name = extract_concept_from_uri(uri)
                 # Clean and normalize the name
                 normalized_name = _normalize_predicate_name(name)
                 if normalized_name and normalized_name not in assigned_names:
                     temp_name_map[idx] = normalized_name
                     assigned_names.add(normalized_name)
                 elif name:
                      logging.debug(f"Name collision for index {idx}: URI '{uri}' -> name '{name}' already assigned. Skipping.") # Changed to debug
                 else:
                      logging.debug(f"Could not extract concept name from URI '{uri}' for index {idx}. Skipping.") # Changed to debug

            # e) Fill missing names with original names if any failed
            final_idx_to_name = original_idx_to_name.copy() # Start with original
            final_idx_to_name.update(temp_name_map) # Overwrite with generated names where successful
            if len(final_idx_to_name) != len(original_idx_to_name):
                 logging.debug("Number of final predicate names differs from original.") # Changed to debug


    elif args.use_random_conceptnet_predicates:
        if graph is None:
            logging.error("ConceptNet graph is required for --use_random_conceptnet_predicates but was not loaded. Skipping item.")
            logging.debug("Falling back to original predicate names.") # Changed to debug
            final_idx_to_name = original_idx_to_name
        else:
            logging.debug("<<<<< ENTERED 'else' block for use_random_conceptnet_predicates >>>>>") # ADDED ENTRY LOG
            # --- START: Added logic to collect all predicate indices from the processed structure ---
            all_predicate_indices: Set[int] = set()
            # Collect from facts
            fact_keys_for_preds = ['noiseless_facts', 'noisy_facts', 'min_fact_dicts_for_query']
            for key in fact_keys_for_preds:
                if key in processed_structure and isinstance(processed_structure[key], list): # Use processed_structure
                    for fact_dict in processed_structure[key]: # Use processed_structure
                        if isinstance(fact_dict, dict) and 'predicateIdx' in fact_dict:
                            all_predicate_indices.add(fact_dict['predicateIdx'])
            # Collect from rules (noiseless)
            if 'noiseless_rules' in processed_structure and isinstance(processed_structure['noiseless_rules'], list): # Use processed_structure
                for rule_dict in processed_structure['noiseless_rules']: # Use processed_structure
                    if isinstance(rule_dict, dict):
                        if 'head' in rule_dict and isinstance(rule_dict['head'], list): # Check head is list
                            for atom in rule_dict['head']:
                                if isinstance(atom, dict) and 'predicateIdx' in atom:
                                    all_predicate_indices.add(atom['predicateIdx'])
                        if 'body' in rule_dict and isinstance(rule_dict['body'], list): # Check body is list
                            for atom in rule_dict['body']:
                                if isinstance(atom, dict) and 'predicateIdx' in atom:
                                    all_predicate_indices.add(atom['predicateIdx'])
            # Collect from rules (noisy)
            if 'noisy_rules' in processed_structure and isinstance(processed_structure['noisy_rules'], dict): # Use processed_structure
                 for rule_list in processed_structure['noisy_rules'].values(): # Use processed_structure
                      if isinstance(rule_list, list):
                           for rule_dict in rule_list:
                               if isinstance(rule_dict, dict):
                                   if 'head' in rule_dict and isinstance(rule_dict['head'], list): # Check head is list
                                       for atom in rule_dict['head']:
                                           if isinstance(atom, dict) and 'predicateIdx' in atom:
                                               all_predicate_indices.add(atom['predicateIdx'])
                                   if 'body' in rule_dict and isinstance(rule_dict['body'], list): # Check body is list
                                       for atom in rule_dict['body']:
                                           if isinstance(atom, dict) and 'predicateIdx' in atom:
                                               all_predicate_indices.add(atom['predicateIdx'])
            # Collect from target_query
            target_query_struct = item.get('target_query')
            if isinstance(target_query_struct, dict) and 'predicateIdx' in target_query_struct:
                 all_predicate_indices.add(target_query_struct['predicateIdx'])

            all_predicate_indices_list = sorted(list(all_predicate_indices))
            logging.debug(f"Generating random unique predicate names for indices: {all_predicate_indices_list}")
            # --- END: Added logic ---

            # Original code modified to use the collected list
            logging.debug(f"Calling generate_unique_conceptnet_names for indices: {all_predicate_indices_list}") # ADDED LOG
            temp_name_map = generate_unique_conceptnet_names(graph, all_predicate_indices_list) # Use collected list
            logging.debug(f"Raw names from generate_unique_conceptnet_names: {temp_name_map}") # ADDED LOG
            # Clean and normalize the generated names
            logging.debug("Normalizing generated names...") # ADDED LOG
            normalized_temp_name_map = {}
            for idx, name in temp_name_map.items(): # ADDED LOOP FOR LOGGING
                normalized_name = _normalize_predicate_name(name)
                normalized_temp_name_map[idx] = normalized_name
                logging.debug(f"  Index {idx}: Raw='{name}' -> Normalized='{normalized_name}'") # ADDED LOG

            # --- START: Revised Filtering and Merging Logic ---
            # 1. Filter new names internally for uniqueness
            filtered_new_names = {}
            assigned_new_normalized_names = set() # Track uniqueness among new names
            logging.debug("Filtering normalized names for uniqueness AMONG NEWLY generated names...")
            for idx, norm_name in normalized_temp_name_map.items():
                if norm_name and norm_name not in assigned_new_normalized_names:
                    filtered_new_names[idx] = norm_name
                    assigned_new_normalized_names.add(norm_name) # Record newly assigned name
                    logging.debug(f"  Index {idx}: Kept unique NEW normalized name '{norm_name}'")
                else:
                   logging.debug(f"  Index {idx}: Skipping NEW normalized name '{norm_name}' (collision with other new names or None).")

            logging.debug(f"Filtered unique NEW normalized names: {filtered_new_names}")

            # 2. Merge with original names, handling collisions
            final_idx_to_name = original_idx_to_name.copy() if original_idx_to_name else {}
            original_names_set = set(final_idx_to_name.values()) # Set of original names for quick lookup

            indices_to_update = {} # Store changes to apply
            colliding_indices = set() # Track indices where new name conflicted

            logging.debug(f"Merging filtered new names with original names (Originals: {original_names_set})")
            for idx, new_name in filtered_new_names.items():
                if idx in final_idx_to_name: # Index exists in original map
                    original_name_at_idx = final_idx_to_name[idx]
                    if new_name != original_name_at_idx: # New name is different from original at this index
                        if new_name not in original_names_set: # And new name doesn't conflict with *any* other original name
                            indices_to_update[idx] = new_name # Mark for update
                            logging.debug(f"  Index {idx}: Will update original '{original_name_at_idx}' to new '{new_name}'.")
                        else: # New name conflicts with some *other* original name
                            colliding_indices.add(idx)
                            logging.warning(f"  Index {idx}: New name '{new_name}' conflicts with an existing original name. Keeping original '{original_name_at_idx}'.")
                    # else: new_name == original_name_at_idx, no change needed
                else: # Index is new (not in original map)
                    if new_name not in original_names_set: # And new name doesn't conflict with any original name
                        indices_to_update[idx] = new_name # Mark for addition
                        logging.debug(f"  Index {idx}: Will add new name '{new_name}'.")
                    else: # New name conflicts with some original name
                        colliding_indices.add(idx)
                        logging.warning(f"  Index {idx}: New name '{new_name}' conflicts with an existing original name. Skipping this new name.")

            # Apply the updates/additions
            final_idx_to_name.update(indices_to_update)
            logging.debug(f"Final merged idx_to_name map after handling collisions: {final_idx_to_name}")
            # --- END: Revised Filtering and Merging Logic ---

    else:
        # Default: Use original names
        logging.debug("Using original predicate names.")
        final_idx_to_name = original_idx_to_name # Fallback to original if ConceptNet not used/failed

    # --- 3. Generate Fact Variable Names (fact_var_idx_to_name) using Faker ---
    fact_var_indices = set()
    fact_keys_for_vars = ['noiseless_facts', 'noisy_facts', 'min_fact_dicts_for_query']
    for key in fact_keys_for_vars:
        # Use processed_structure here as well, although negations don't affect variable indices
        if key in processed_structure and isinstance(processed_structure[key], list):
            for fact_dict in processed_structure[key]:
                if isinstance(fact_dict, dict) and 'variables' in fact_dict and isinstance(fact_dict['variables'], list): # Check variables is list
                    for var in fact_dict['variables']:
                        if isinstance(var, int): # We only map integer variables
                            fact_var_indices.add(var)

    # Also collect from target_query if it's treated as a fact-like structure
    target_query_struct = item.get('target_query')
    if isinstance(target_query_struct, dict) and 'variables' in target_query_struct:
         for var in target_query_struct['variables']:
             if isinstance(var, int):
                 fact_var_indices.add(var)

    # Generate unique fake names for fact variables
    fact_var_idx_to_name: Dict[int, str] = {}
    assigned_fact_names: Set[str] = set() # 用于跟踪已分配的名称
    for idx in fact_var_indices:
        while True:
            name = fake.first_name()
            # 检查生成的名称是否已分配
            if name not in assigned_fact_names:
                fact_var_idx_to_name[idx] = name
                assigned_fact_names.add(name) # 将新名称添加到已分配集合中
                break # 找到唯一名称，跳出循环
            # else: # 如果需要，可以添加日志记录碰撞情况
            #     logging.debug(f"Fact name collision for index {idx}: '{name}'. Retrying...")
    logging.debug(f"Generated unique fact variable names: {fact_var_idx_to_name}") # 更新日志信息


    # --- 4. Create ASP Program Strings using format_dict_structure_to_asp ---
    # Pass the potentially modified structure (processed_structure)
    asp_program_dlv2 = format_dict_structure_to_asp(
        dict_structure=processed_structure, # Use the structure after potential negation removal
        idx_to_name=final_idx_to_name,
        fact_var_idx_to_name=fact_var_idx_to_name,
        rule_var_idx_to_name=rule_var_idx_to_name,
        use_disjunction=use_disjunction
    )
    item['asp_program_dlv2'] = asp_program_dlv2
    # --- START: Added logic to update min_fact_dicts_for_query if is_positive ---
    if is_positive:
        if 'asp_program_dlv2' in item and isinstance(item['asp_program_dlv2'], dict) and 'noiseless_facts' in item['asp_program_dlv2']:
            # Ensure noiseless_facts is actually a list before assigning
            if isinstance(item['asp_program_dlv2']['noiseless_facts'], list):
                item['asp_program_dlv2']['min_fact_dicts_for_query'] = item['asp_program_dlv2']['noiseless_facts']
                logging.debug(f"Item {item.get('id', 'N/A')}: is_positive is True. Updated 'min_fact_dicts_for_query' with 'noiseless_facts'.")
            else:
                logging.warning(f"Item {item.get('id', 'N/A')}: is_positive is True, but 'noiseless_facts' is not a list. Cannot update 'min_fact_dicts_for_query'.")
        else:
            logging.warning(f"Item {item.get('id', 'N/A')}: is_positive is True, but could not update 'min_fact_dicts_for_query'. Check 'asp_program_dlv2' structure or 'noiseless_facts' key.")
    # --- END: Added logic ---
    # --- End formatting ---


    # --- 5. Convert target_query (if it exists and is a dict, treat as fact) ---
    # Use final_idx_to_name and fact_var_idx_to_name
    # We still need to format the target_query separately as it's not part of the main structure
    target_query_str = None
    target_query_dict_original = item.get('target_query') # Get original query dict
    if isinstance(target_query_dict_original, dict):
        # Apply negation removal if needed
        target_query_dict_for_conv = _remove_negations(copy.deepcopy(target_query_dict_original)) if is_positive else target_query_dict_original
        try:
            # Use dict_to_asp_strings directly for the single target query dict
            # Ensure the var_idx_to_name maps the variables present in the query
            target_query_list = dict_to_asp_strings(
                target_query_dict_for_conv, final_idx_to_name, is_fact=True, var_idx_to_name=fact_var_idx_to_name
            )
            if target_query_list:
                target_query_str = target_query_list[0]
                logging.debug(f"Formatted target query: {target_query_str}")
            else:
                target_query_str = f"% Error: Empty result formatting target_query {target_query_dict_for_conv}"
                logging.warning(f"Empty result formatting target_query for item {item.get('id', 'N/A')}")
        except Exception as e:
            target_query_str = f"% Error formatting target_query: {e}"
            logging.warning(f"Error formatting target_query for item {item.get('id', 'N/A')}: {e}", exc_info=True)
        item['target_query'] = target_query_str # Store the formatted string (or error)
    # Note: original_target_query_dict_original holds the dict structure before this conversion

    # --- START: Added logic for negative target query ONLY when is_positive is True ---
    # Store the original truth value before potential modification
    original_target_query_in_answerset_value_before_neg_check = item.get('target_query_in_answerset')
    # *** Condition updated: Only modify if is_positive is True AND query starts with '- ' ***
    if is_positive and target_query_str and not target_query_str.startswith('% Error') and target_query_str.startswith('- '):
        logging.debug(f"Detected negative target query AND is_positive=True for item {item.get('id', 'N/A')}: '{target_query_str}'")
        # Remove '- ', trim whitespace, remove trailing dot if present, add dot back
        modified_target_query_str = target_query_str[2:].strip().rstrip('.') + '.'
        item['target_query'] = modified_target_query_str # Update the query string in the item
        logging.debug(f"Modified negative target query to: '{modified_target_query_str}'")

        # Re-calculate target_query_in_answerset based on the modified query and noiseless program
        # Build the noiseless program string (using final names)
        noiseless_asp_strings_for_neg_check: List[str] = []
        noiseless_facts_list_neg = asp_program_dlv2.get('noiseless_facts', [])
        noiseless_rules_list_neg = asp_program_dlv2.get('noiseless_rules', [])
        if isinstance(noiseless_facts_list_neg, list):
            noiseless_asp_strings_for_neg_check.extend(s for s in noiseless_facts_list_neg if not s.startswith('% Error'))
        if isinstance(noiseless_rules_list_neg, list):
            noiseless_asp_strings_for_neg_check.extend(s for s in noiseless_rules_list_neg if not s.startswith('% Error'))

        if noiseless_asp_strings_for_neg_check:
            noiseless_program_str_neg = "\n".join(noiseless_asp_strings_for_neg_check)
            try:
                logging.debug(f"Running DLV2 (noiseless check for modified negative query when is_positive=True) for item {item.get('id', 'N/A')}...")
                noiseless_dlv2_result_neg = runner.run(noiseless_program_str_neg, num_answer_sets=0)
                new_target_query_in_answerset = False # Default to False
                if noiseless_dlv2_result_neg.get("success") and isinstance(noiseless_dlv2_result_neg.get("answer_sets"), list) and noiseless_dlv2_result_neg["answer_sets"]:
                    # DLV2 result answer sets often don't have spaces or trailing dots
                    noiseless_answer_set_neg_cleaned = {atom.replace(' ', '').replace('.', '') for atom in noiseless_dlv2_result_neg["answer_sets"][0] if isinstance(atom, str)}
                    modified_target_query_str_for_comp = modified_target_query_str.replace(' ', '').replace('.', '')
                    new_target_query_in_answerset = modified_target_query_str_for_comp in noiseless_answer_set_neg_cleaned
                    logging.debug(f"Recalculated target_query_in_answerset for modified negative query (is_positive=True): {new_target_query_in_answerset}")
                else:
                    logging.warning(f"Could not recalculate truth value for modified negative query (is_positive=True): DLV2 (noiseless) failed or no answer sets.")
                # Update the item's truth value
                item['target_query_in_answerset'] = new_target_query_in_answerset
            except (Dlv2RunnerError, ValueError, Exception) as dlv_err_neg:
                logging.error(f"Error running DLV2 for modified negative query check (is_positive=True): {dlv_err_neg}")
                # Keep the original truth value if recalculation fails
                item['target_query_in_answerset'] = original_target_query_in_answerset_value_before_neg_check
                logging.warning(f"Failed to recalculate truth value for modified negative query (is_positive=True). Restoring original value: {original_target_query_in_answerset_value_before_neg_check}")
        else:
             logging.warning("Could not recalculate truth value for modified negative query (is_positive=True): No noiseless program parts found.")
             item['target_query_in_answerset'] = original_target_query_in_answerset_value_before_neg_check # Restore original if no program
    # --- END: Added logic for negative target query ---


    # --- 6. Run DLV2 on FULL program (including noise, potentially modified names/structure) ---
    # This result is stored in dlv2_result but NOT used for the final target_query_in_answerset check
    all_asp_strings: List[str] = []
    # Collect facts from the new structure
    fact_keys = ['noiseless_facts', 'noisy_facts', 'min_fact_dicts_for_query']
    for key in fact_keys:
        if key in asp_program_dlv2 and isinstance(asp_program_dlv2[key], list):
            all_asp_strings.extend(asp_program_dlv2[key])
    # Collect noiseless rules
    if 'noiseless_rules' in asp_program_dlv2 and isinstance(asp_program_dlv2['noiseless_rules'], list):
        all_asp_strings.extend(asp_program_dlv2['noiseless_rules'])
    # Collect noisy rules (which is a dict of lists)
    if 'noisy_rules' in asp_program_dlv2 and isinstance(asp_program_dlv2['noisy_rules'], dict):
        for rule_list in asp_program_dlv2['noisy_rules'].values():
            if isinstance(rule_list, list):
                all_asp_strings.extend(rule_list)

    # Filter out any potential error comments before joining
    full_asp_program = "\n".join(s for s in all_asp_strings if not s.startswith('% Error'))
    logging.debug(f"Running DLV2 for item {item.get('id', 'N/A')} with program:\n{full_asp_program[:500]}...") # Log start and truncated program

    try:
        # Run DLV2, get all answer sets (num_answer_sets=0)
        # Consider adding a timeout argument later if needed
        dlv2_result = runner.run(full_asp_program, num_answer_sets=0)
        item['dlv2_asp_str'] = full_asp_program
        item['dlv2_result'] = dlv2_result
        if dlv2_result.get("success"):
            logging.debug(f"DLV2 success for item {item.get('id', 'N/A')}. Found {len(dlv2_result.get('answer_sets', []))} answer sets.")
        else:
            logging.warning(f"DLV2 failed for item {item.get('id', 'N/A')}. Error: {dlv2_result.get('error_message')}")
            logging.debug(f"DLV2 raw output for failed item {item.get('id', 'N/A')}:\n{dlv2_result.get('raw_output')}")

    except (Dlv2RunnerError, ValueError, Exception) as dlv_err: # Catch potential errors from runner
        logging.error(f"Error running DLV2 for item {item.get('id', 'N/A')}: {dlv_err}", exc_info=True)
        item['dlv2_result'] = {
            "success": False,
            "answer_sets": None,
            "error_message": f"Python error during DLV2 execution: {type(dlv_err).__name__}: {dlv_err}",
            "raw_output": None
        }
        # Determine error type from Python exception
        dlv2_error_type = f"Python DLV2 Runner Error: {type(dlv_err).__name__}"


    # --- Determine Error Type if DLV2 failed ---
    if not item.get('dlv2_result', {}).get('success') and dlv2_error_type is None: # Check if DLV2 failed and Python error didn't already set type
        error_message = item['dlv2_result'].get('error_message', '')
        raw_output = item['dlv2_result'].get('raw_output', '')
        full_error_text = (error_message + "\n" + raw_output).lower() # Combine and lower for easier check

        if "incoherent" in full_error_text:
            dlv2_error_type = "INCOHERENT"
        elif "syntax error, unexpected naf" in full_error_text:
            dlv2_error_type = "Syntax Error: NAF"
        elif "syntax error, unexpected symbolic_constant" in full_error_text:
            dlv2_error_type = "Syntax Error: SYMBOLIC_CONSTANT"
        # Removed unnecessary else block below
        # else:
        # dlv2_error_type = "Other DLV2 Error" # This was likely intended to be the default if no specific error matched
        # Let's assign 'Other DLV2 Error' if no specific type was matched before this point
        if dlv2_error_type is None: # Check if a specific type was already assigned
             dlv2_error_type = "Other DLV2 Error"
        logging.debug(f"Identified DLV2 error type for item {item.get('id', 'N/A')}: {dlv2_error_type}")


    # --- 7. Consistency Check (Conditional) ---
    # This check verifies if the original target_query_in_answerset truth value
    # (determined in script 01 based on original names and noise consistency)
    # is reproducible in script 06 using the potentially new names,
    # BOTH for the noiseless program AND the full noisy program.
    if args.check_query_in_answer:
        consistency_check_passed = None # Default: cannot check
        check_noiseless_passed = None
        check_noisy_passed = None

        target_query_str = item.get('target_query') # The converted string query (with new names)
        # Ensure target_query_str is not an error comment before proceeding
        if target_query_str and not target_query_str.startswith('% Error'):
            target_query_str_for_comp = target_query_str.replace(' ', '').replace('.', '') # Prepare for comparison

            # --- START: Pre-calculate ground truth if noise/positivity affects it ---
            if is_positive or not use_noisy:
                 logging.debug(f"Condition (is_positive or not use_noisy) met for item {item.get('id', 'N/A')}. Potentially recalculating ground truth before checks.")
                 # 1. Re-format target query using original dict, final names, NO negation removal
                 if isinstance(target_query_dict_original, dict):
                     try:
                         reformatted_query_list = dict_to_asp_strings(
                             target_query_dict_original, # Use original dict structure
                             final_idx_to_name,
                             is_fact=True,
                             var_idx_to_name=fact_var_idx_to_name
                         )
                         if reformatted_query_list:
                             reformatted_original_query_str = reformatted_query_list[0]
                             logging.debug(f"Pre-check: Re-formatted original target query (no negation removal): {reformatted_original_query_str}")
                             # Update item's query string *if* recalculation is needed
                             item['target_query'] = reformatted_original_query_str

                             # 2. Re-calculate truth value using noiseless program and this re-formatted query
                             noiseless_asp_strings_recalc: List[str] = []
                             noiseless_facts_list_recalc = asp_program_dlv2.get('noiseless_facts', [])
                             noiseless_rules_list_recalc = asp_program_dlv2.get('noiseless_rules', [])
                             if isinstance(noiseless_facts_list_recalc, list):
                                 noiseless_asp_strings_recalc.extend(s for s in noiseless_facts_list_recalc if not s.startswith('% Error'))
                             if isinstance(noiseless_rules_list_recalc, list):
                                 noiseless_asp_strings_recalc.extend(s for s in noiseless_rules_list_recalc if not s.startswith('% Error'))

                             if noiseless_asp_strings_recalc:
                                 noiseless_program_str_recalc = "\n".join(noiseless_asp_strings_recalc)
                                 try:
                                     logging.debug(f"Running DLV2 (noiseless pre-check for ground truth recalc) for item {item.get('id', 'N/A')}...")
                                     noiseless_dlv2_result_recalc = runner.run(noiseless_program_str_recalc, num_answer_sets=0)
                                     recalculated_truth_value = False # Default
                                     if noiseless_dlv2_result_recalc.get("success") and isinstance(noiseless_dlv2_result_recalc.get("answer_sets"), list) and noiseless_dlv2_result_recalc["answer_sets"]:
                                         noiseless_answer_set_recalc_cleaned = {atom.replace(' ', '').replace('.', '') for atom in noiseless_dlv2_result_recalc["answer_sets"][0] if isinstance(atom, str)}
                                         reformatted_query_str_for_comp = reformatted_original_query_str.replace(' ', '').replace('.', '')
                                         recalculated_truth_value = reformatted_query_str_for_comp in noiseless_answer_set_recalc_cleaned
                                         logging.debug(f"Pre-check: Recalculated potential ground truth: {recalculated_truth_value}")
                                     else:
                                         logging.warning(f"Pre-check: Could not recalculate potential ground truth: DLV2 (noiseless) failed or no answer sets.")
                                     # Update the ground truth and the item's value
                                     current_ground_truth = recalculated_truth_value
                                     item['target_query_in_answerset'] = recalculated_truth_value
                                     logging.debug(f"Pre-check: Updated current_ground_truth to: {current_ground_truth}")
                                 except (Dlv2RunnerError, ValueError, Exception) as dlv_err_recalc:
                                     logging.error(f"Pre-check: Error running DLV2 for ground truth recalculation: {dlv_err_recalc}")
                                     # If recalc fails, ground truth remains the original value (already set)
                                     item['target_query_in_answerset'] = original_target_query_in_answerset_value # Ensure item reflects original
                                     logging.warning(f"Pre-check: Failed recalculation. Ground truth remains original: {current_ground_truth}")
                             else:
                                 logging.warning("Pre-check: Could not recalculate potential ground truth: No noiseless program parts found.")
                                 item['target_query_in_answerset'] = original_target_query_in_answerset_value # Ensure item reflects original
                         else:
                             logging.warning(f"Pre-check: Could not re-format original target query for item {item.get('id', 'N/A')}.")
                             item['target_query_in_answerset'] = original_target_query_in_answerset_value # Ensure item reflects original
                     except Exception as e_reformat:
                         logging.warning(f"Pre-check: Error re-formatting original target query for item {item.get('id', 'N/A')}: {e_reformat}")
                         item['target_query_in_answerset'] = original_target_query_in_answerset_value # Ensure item reflects original
                 else:
                     logging.warning(f"Pre-check: Cannot recalculate potential ground truth: Original target_query dict not found or invalid for item {item.get('id', 'N/A')}.")
                     item['target_query_in_answerset'] = original_target_query_in_answerset_value # Ensure item reflects original
            # --- END: Pre-calculate ground truth ---


            # --- Check 1: Noiseless Program (New Names) vs Determined Ground Truth ---
            # Note: target_query_str might have been updated in the pre-check if is_positive/not use_noisy
            # We need to re-fetch it and re-prepare for comparison
            target_query_str_for_check = item.get('target_query')
            if target_query_str_for_check and not target_query_str_for_check.startswith('% Error'):
                 target_query_str_for_comp = target_query_str_for_check.replace(' ', '').replace('.', '') # Re-prepare

                 noiseless_asp_strings: List[str] = []
                 # Access lists directly from asp_program_dlv2 dictionary
                 noiseless_facts_list = asp_program_dlv2.get('noiseless_facts', [])
                 noiseless_rules_list = asp_program_dlv2.get('noiseless_rules', [])

                 if isinstance(noiseless_facts_list, list):
                     noiseless_asp_strings.extend(s for s in noiseless_facts_list if not s.startswith('% Error'))
                 if isinstance(noiseless_rules_list, list):
                     noiseless_asp_strings.extend(s for s in noiseless_rules_list if not s.startswith('% Error'))

            if noiseless_asp_strings:
                noiseless_program_str = "\n".join(noiseless_asp_strings)
                logging.debug(f"Running DLV2 (noiseless check) for item {item.get('id', 'N/A')}...")
                try:
                    noiseless_dlv2_result = runner.run(noiseless_program_str, num_answer_sets=0)
                    if noiseless_dlv2_result.get("success") and isinstance(noiseless_dlv2_result.get("answer_sets"), list) and noiseless_dlv2_result["answer_sets"]:
                        noiseless_answer_set = noiseless_dlv2_result["answer_sets"][0]
                        if isinstance(noiseless_answer_set, list):
                            # Clean answer set atoms for comparison (remove spaces and dots)
                            noiseless_answer_set_cleaned = {atom.replace(' ', '').replace('.', '') for atom in noiseless_answer_set if isinstance(atom, str)}
                            presence_in_noiseless_new = target_query_str_for_comp in noiseless_answer_set_cleaned
                            if current_ground_truth is not None: # Compare against the current ground truth
                                check_noiseless_passed = (presence_in_noiseless_new == current_ground_truth)
                                logging.debug(f"Consistency Check (Noiseless) for item {item.get('id', 'N/A')}: GroundTruth={current_ground_truth}, Actual={presence_in_noiseless_new}, Passed={check_noiseless_passed}")
                            else:
                                logging.debug(f"Consistency Check (Noiseless) for item {item.get('id', 'N/A')}: Cannot compare, ground truth value is None.")
                        else:
                            logging.debug(f"Consistency Check (Noiseless) for item {item.get('id', 'N/A')}: Cannot perform check - noiseless answer set is not a list.")
                    else:
                        logging.debug(f"Consistency Check (Noiseless) for item {item.get('id', 'N/A')}: Cannot perform check - DLV2 (noiseless) failed or no answer sets found.")
                except (Dlv2RunnerError, ValueError, Exception) as dlv_err_noiseless:
                    logging.warning(f"Error running DLV2 (noiseless check) for item {item.get('id', 'N/A')}: {dlv_err_noiseless}")
                    check_noiseless_passed = False # Mark as failed if DLV2 run fails
            else:
                logging.debug(f"Consistency Check (Noiseless) for item {item.get('id', 'N/A')}: Cannot perform check - no noiseless facts/rules found.")


            # --- Check 2: Noisy Program (New Names) vs Original Truth Value ---
            dlv2_result_data = item.get('dlv2_result', {}) # Result from the main DLV2 run on the full program
            if dlv2_result_data.get("success") and isinstance(dlv2_result_data.get("answer_sets"), list) and dlv2_result_data["answer_sets"]:
                noisy_answer_set = dlv2_result_data["answer_sets"][0]
                if isinstance(noisy_answer_set, list):
                    # Clean answer set atoms for comparison (remove spaces and dots)
                    noisy_answer_set_cleaned = {atom.replace(' ', '').replace('.', '') for atom in noisy_answer_set if isinstance(atom, str)}
                    presence_in_noisy_new = target_query_str_for_comp in noisy_answer_set_cleaned
                    if current_ground_truth is not None: # Compare against the current ground truth
                        check_noisy_passed = (presence_in_noisy_new == current_ground_truth)
                        logging.debug(f"Consistency Check (Noisy) for item {item.get('id', 'N/A')}: GroundTruth={current_ground_truth}, Actual={presence_in_noisy_new}, Passed={check_noisy_passed}")
                    else:
                        logging.debug(f"Consistency Check (Noisy) for item {item.get('id', 'N/A')}: Cannot compare, ground truth value is None.")
                else:
                    logging.debug(f"Consistency Check (Noisy) for item {item.get('id', 'N/A')}: Cannot perform check - noisy answer set is not a list.")
            else:
                logging.debug(f"Consistency Check (Noisy) for item {item.get('id', 'N/A')}: Cannot perform check - DLV2 (noisy) failed or no answer sets found.")

            # --- Final Decision ---
            if check_noiseless_passed is not None and check_noisy_passed is not None:
                consistency_check_passed = check_noiseless_passed and check_noisy_passed
                logging.debug(f"Final Consistency Check for item {item.get('id', 'N/A')}: Noiseless Passed={check_noiseless_passed}, Noisy Passed={check_noisy_passed}, Overall Passed={consistency_check_passed}")
            else:
                # If either check couldn't be performed, the overall check is considered undetermined (None) or failed (False) depending on preference. Let's default to None.
                consistency_check_passed = None
                logging.debug(f"Final Consistency Check for item {item.get('id', 'N/A')}: Could not determine overall result as one or both checks failed or were skipped.")

            # --- START: Added detailed logging for genuine failures (moved here) ---
            # Determine if truth value was recalculated due to is_positive + negative query (needed for the check below)
            truth_value_recalculated_due_to_pos_neg = False
            if is_positive and target_query_str and not target_query_str.startswith('% Error') and target_query_str.startswith('- '):
                if item.get('target_query_in_answerset') != original_target_query_in_answerset_value_before_neg_check:
                     truth_value_recalculated_due_to_pos_neg = True

            # if consistency_check_passed is False and not truth_value_recalculated_due_to_pos_neg: # Check if genuinely failed
            #     logging.warning(f"Consistency Check FAILED for item ID: {item.get('id', 'N/A')}")
            #     logging.warning(f"  - Original target_query_in_answerset: {original_target_query_in_answerset_value}")
            #     logging.warning(f"  - Current Ground Truth used for check: {current_ground_truth}")
            #     logging.warning(f"  - is_positive flag: {is_positive}")
            #     logging.warning(f"  - use_noisy flag: {use_noisy}")
            #     # Need to re-fetch target_query_str_for_check as it might have been updated
            #     target_query_str_for_check_log = item.get('target_query', 'N/A')
            #     logging.warning(f"  - Formatted Target Query: {target_query_str_for_check_log}")
            #     logging.warning(f"  - Check Noiseless Passed: {check_noiseless_passed}")
            #     logging.warning(f"  - Check Noisy Passed: {check_noisy_passed}")
            #     # Log relevant parts of the item for inspection
            #     logging.warning(f"  - Final Predicate Map: {final_idx_to_name}")
            #     logging.warning(f"  - Final Fact Var Map: {fact_var_idx_to_name}")
            #     logging.warning(f"  - Noiseless ASP Program Snippet:\n{noiseless_program_str[:500] if 'noiseless_program_str' in locals() else 'N/A'}...")
            #     logging.warning(f"  - Full ASP Program Snippet:\n{full_asp_program[:500] if 'full_asp_program' in locals() else 'N/A'}...")
            #     logging.warning(f"  - Noiseless DLV2 Result Success: {noiseless_dlv2_result.get('success') if 'noiseless_dlv2_result' in locals() else 'N/A'}")
            #     logging.warning(f"  - Noisy DLV2 Result Success: {dlv2_result_data.get('success')}")
            # --- END: Added detailed logging ---

        else: # This else corresponds to the outer 'if target_query_str and not target_query_str.startswith('% Error')'
            logging.debug(f"Consistency Check for item {item.get('id', 'N/A')}: Cannot perform check - initial target_query string is missing or invalid.")

        item['dlv2_consistency_check_passed'] = consistency_check_passed # Assign the final result based on checks against current_ground_truth
    else: # This else corresponds to 'if args.check_query_in_answer:'
        # If check is not enabled, ensure the field exists and is None
        item['dlv2_consistency_check_passed'] = None
        logging.debug(f"Consistency check skipped for item {item.get('id', 'N/A')} as --check_query_in_answer is not set. Field set to None.")
    # --- END Consistency Check ---


    # --- 8. Ensure target_query_in_answerset key exists ---
    # The value might have been intentionally modified by the negative query handling
    # or the consistency failure recalculation logic above. We trust those modifications.
    # We just ensure the key exists, using the original value only if it's missing entirely.
    if 'target_query_in_answerset' not in item:
        # If the key was somehow missing in the input AND not set by previous logic
        item['target_query_in_answerset'] = original_target_query_in_answerset_value
        if original_target_query_in_answerset_value is None:
            logging.warning(f"Key 'target_query_in_answerset' was missing for item {item.get('id', 'N/A')} and original value was None.")
        else:
             logging.warning(f"Key 'target_query_in_answerset' was missing for item {item.get('id', 'N/A')}. Setting to original value: {original_target_query_in_answerset_value}")
    # Removed the check that forced the value back to the original if it was changed.

    # Remove the old consistency check field if it exists from previous runs/versions
    item.pop('dlv2_consistency_check', None) # Remove old field if present


    # --- 9. Modify Graph Data if is_positive ---
    if is_positive:
        logging.debug(f"is_positive is True. Modifying graph data for item {item.get('id', 'N/A')}.")
        graph_keys_to_modify = ['rule_predicate_operation_graph', 'noisy_rule_predicate_operation_graph']
        for key in graph_keys_to_modify:
            if key in item and isinstance(item[key], dict) and 'data' in item[key] and isinstance(item[key]['data'], list):
                original_data_len = len(item[key]['data'])
                item[key]['data'] = [1] * original_data_len # Replace with list of 1s
                logging.debug(f"  Modified '{key}' data list (length {original_data_len}) to all 1s.")
            else:
                logging.debug(f"  Could not modify '{key}' data: Key not found, not a dict, or 'data' key missing/not a list.")


    # --- 10. Remove other specified fields ---
    # Note: dict_structure is removed here. The modified version (processed_structure) was used for ASP generation.
    fields_to_remove = [
        'dict_structure', # Remove the original structure field
        'idx_to_name', # Remove original predicate mapping
        'var_idx_to_name', # Remove original rule variable mapping
        'rule_graph',
        'rule_predicate_graph',
        'rule_predicate_operation_graph',
        'nosiy_rule_predicate_operation_graph', # Typo in original request? Assuming 'noisy_'
        'idx2type',
        'nosiy_idx2type', # Typo in original request? Assuming 'noisy_'
        'strong_negation_prob', # Added based on request
        'default_negation_prob' # Added based on request
    ]
    # Also remove potential typo versions if they exist
    fields_to_remove.extend([
        'noisy_rule_predicate_operation_graph',
        'noisy_idx2type'
    ])

    # The ASPProgramAnalyzer lines seem out of place here as the fields they use are in fields_to_remove.
    # If analysis is needed, it should happen *before* removing the fields.
    # Commenting them out for now as they would cause errors after removal.
    # apa = ASPProgramAnalyzer(item['noisy_rule_predicate_operation_graph'], item['noisy_idx2type'])
    # apa_wo_n = ASPProgramAnalyzer(item.get('rule_predicate_operation_graph'), item.get('idx2type')) # Use .get()

    if use_noisy:
        M  = item.get('noisy_rule_predicate_operation_graph') # Use .get()
        id2type = item.get('noisy_idx2type') # Use .get()
    else:
        M = item.get('rule_predicate_operation_graph') # Use .get()
        id2type = item.get('idx2type') # Use .get()

    # Check if M and id2type were successfully retrieved before classification
    if M is not None and id2type is not None:
        asp_type = ASPProgramAnalyzer(M, id2type, use_disjunction=use_disjunction).classify()
        item.update(asp_type)
    else:
        logging.warning(f"Skipping ASP classification for item {item.get('id', 'N/A')} because M or id2type is None (likely due to noise removal or missing keys).")


    for field in fields_to_remove:
        item.pop(field, None) # Safely remove if exists

    # --- 11. Add the generated mappings (with string keys for JSON compatibility) to the output item ---
    logging.debug(f"Final predicate map for item {item.get('id', 'N/A')}: {final_idx_to_name}") # DEBUG: Check content before assignment
    item['predicate_idx_to_desc'] = {str(k): v for k, v in final_idx_to_name.items()}
    item['name_idx_to_desc'] = {str(k): v for k, v in fact_var_idx_to_name.items()}

    # Determine if truth value was recalculated due to is_positive + negative query
    truth_value_recalculated_due_to_pos_neg = False
    if is_positive and target_query_str and not target_query_str.startswith('% Error') and target_query_str.startswith('- '):
        # Check if the value actually changed compared to before the check
        if item.get('target_query_in_answerset') != original_target_query_in_answerset_value_before_neg_check:
             truth_value_recalculated_due_to_pos_neg = True

    return item, dlv2_error_type, truth_value_recalculated_due_to_pos_neg # Return processed item, error type, and flag

def main(args: argparse.Namespace) -> None:
    """
    Main function to load data, process each item to generate DLV2 strings,
    and save the augmented data to a new JSONL file.

    :param args: Command-line arguments parsed by argparse.
    :type args: argparse.Namespace
    :return: None
    :rtype: None
    """
    # --- Argument Validation ---
    if args.use_random_conceptnet_predicates and args.use_related_conceptnet_predicates:
        logging.error("Error: --use_random_conceptnet_predicates and --use_related_conceptnet_predicates are mutually exclusive.")
        sys.exit(1)

    input_file_path: Path = Path(args.input_path)
    conceptnet_graph_path: Path = Path(args.conceptnet_graph_path)

    # --- Determine output path ---
    if args.output_path:
        output_file_path: Path = Path(args.output_path)
    else:
        timestamp: str = datetime.now().strftime("%Y_%m_%d_%H_%M")
        default_output_dir: Path = Path("datasets/symtex_dlv2")
        output_file_path = default_output_dir / f"{timestamp}.jsonl"
        logging.info(f"No output path provided. Using default: {output_file_path}")

    logging.info(f"Input data file: {input_file_path}")
    logging.info(f"Output data file: {output_file_path}")
    logging.info(f"Use disjunction for rules: {args.use_disjunction}")
    logging.info(f"Use random ConceptNet predicates: {args.use_random_conceptnet_predicates}")
    logging.info(f"Use related ConceptNet predicates: {args.use_related_conceptnet_predicates}")
    logging.info(f"Perform consistency check: {args.check_query_in_answer}")
    # logging.info(f"Generate positive program (remove negations): {args.is_positive}") # Removed old flag log
    # logging.info(f"Use noisy data: {not args.no_noisy}") # Removed old flag log
    logging.info(f"Probability of removing negations: {args.negation_removal_prob}")
    logging.info(f"Probability of removing noise: {args.noise_removal_prob}")
    logging.info(f"Random seed: {args.random_seed}")

    # --- Set Random Seed ---
    if args.random_seed is not None:
        random.seed(args.random_seed)
        logging.info(f"Set random seed to: {args.random_seed}")

    # --- Load ConceptNet Graph (Conditionally) ---
    graph: Optional[nx.DiGraph] = None
    if args.use_random_conceptnet_predicates or args.use_related_conceptnet_predicates:
        logging.info(f"Loading ConceptNet graph from: {conceptnet_graph_path}")
        if not conceptnet_graph_path.is_file():
            logging.error(f"ConceptNet graph file not found at {conceptnet_graph_path}")
            sys.exit(1)
        try:
            # Use full path for imported function
            graph = load_graph_from_graphml(str(conceptnet_graph_path))
            if graph is not None:
                logging.info(f"ConceptNet graph loaded successfully ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges).")
            else:
                logging.error(f"load_graph_from_graphml returned None for path: {conceptnet_graph_path}")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Failed to load ConceptNet graph: {e}", exc_info=True)
            sys.exit(1)

    # --- Initialize DLV2 Runner ---
    try:
        runner = Dlv2Runner()
        logging.info(f"Dlv2Runner initialized. Using DLV2 executable at: {runner.dlv2_path}")
    except Dlv2RunnerError as e:
        logging.error(f"Failed to initialize Dlv2Runner: {e}")
        sys.exit(1)

    # --- Load Input Data ---
    logging.info(f"Attempting to load data from: {input_file_path}")
    if not input_file_path.is_file():
        logging.error(f"Input file not found at {input_file_path}")
        sys.exit(1)

    try:
        all_data: List[Dict[str, Any]] = read_jsonl_parallel(input_file_path)

        if not all_data:
            logging.warning("No data loaded from the file. Exiting.")
            sys.exit(0)
        else:
            logging.info(f"Successfully loaded {len(all_data)} entries from {input_file_path}.")

        # --- Process Data ---
        logging.info(f"Processing {len(all_data)} entries to generate DLV2 programs...")
        # Use partial to fix the graph, args, and runner parameters for process_item.
        # is_positive and use_noisy will be determined inside process_item based on args probabilities.
        # Note: If using multiprocessing later, ensure graph and runner objects are handled correctly
        # Removed: use_noisy_flag = not args.no_noisy
        processing_func = partial(process_item,
                                  use_disjunction=args.use_disjunction,
                                  graph=graph,
                                  args=args, # Pass the whole args object containing probabilities and seed
                                  runner=runner)
                                  # Removed is_positive and use_noisy from partial, as they are decided inside

        processed_data: List[Dict[str, Any]] = []
        dlv2_error_counts = Counter() # Initialize error counter here
        # Initialize consistency check counters
        consistency_passed_count = 0
        consistency_failed_count = 0
        consistency_skipped_count = 0
        consistency_mismatch_after_recalc_count = 0 # New counter

        # Initialize classification counters
        classification_counts = Counter({
            "is_positive": 0,
            "is_tight": 0,
            "is_disjunctive_stratified": 0,
            "is_stratified": 0,
        })

        # Single loop for processing and error counting
        for i, item in enumerate(all_data):
            if (i + 1) % 100 == 0: # Log progress
                 logging.info(f"  Processing {i + 1}/{len(all_data)} entries...")
            try:
                # Update to receive the third return value (the flag)
                processed_item, error_type, truth_recalculated_flag = processing_func(item) # Get item, error type, and flag
                if error_type:
                    dlv2_error_counts[error_type] += 1 # Update counter for DLV2 errors
                    logging.debug(f"Skipping item {item.get('id', 'N/A')} due to DLV2 error: {error_type}")
                else:
                    # Only append the item if there was no DLV2 error
                    processed_data.append(processed_item)
                    # Increment consistency counters if check was enabled and performed
                    if args.check_query_in_answer:
                        check_result = processed_item.get('dlv2_consistency_check_passed')
                        if check_result is True:
                            consistency_passed_count += 1
                        elif check_result is False:
                            # Check the flag before incrementing failed count
                            if truth_recalculated_flag:
                                consistency_mismatch_after_recalc_count += 1 # Increment new counter
                                logging.debug(f"Item {item.get('id', 'N/A')}: Consistency check 'failed' but truth value was recalculated due to is_positive+neg_query. Counted as 'mismatch_after_recalc'.")
                            else:
                                consistency_failed_count += 1 # Count as genuine failure
                        else: # Handles None or missing key (skipped/not applicable)
                            consistency_skipped_count += 1

                    # Update classification counters based on the results in the processed item
                    # These keys are added by item.update(asp_type) within process_item
                    if processed_item.get("is_positive") is True:
                        classification_counts["is_positive"] += 1
                    if processed_item.get("is_tight") is True:
                        classification_counts["is_tight"] += 1
                    # Note: is_head_cycle_free is merged into is_tight, so no separate counter needed

                    # Check stratification based on use_disjunction flag used during processing
                    if args.use_disjunction:
                        if processed_item.get("is_disjunctive_stratified") is True:
                            classification_counts["is_disjunctive_stratified"] += 1
                    else:
                        if processed_item.get("is_stratified") is True:
                            classification_counts["is_stratified"] += 1

            except Exception as proc_e:
                 logging.error(f"Error processing item {i}: {item.get('id', 'N/A')}. Error: {proc_e}", exc_info=True)
                 dlv2_error_counts["Item Processing Error"] += 1 # Count processing errors separately
                 # Do not append the item if a processing error occurred
                 # Optionally add a placeholder or skip the item in processed_data
                 # processed_data.append({"id": item.get('id', 'N/A'), "error": f"Item Processing Error: {proc_e}"})

        logging.info("Processing complete.")

        # --- Report Errors ---

        if dlv2_error_counts:
            logging.info("--- DLV2 Execution Error Summary ---")
            total_errors = sum(dlv2_error_counts.values())
            logging.info(f"Total items with DLV2 errors: {total_errors}")
            for error_type, count in dlv2_error_counts.items():
                logging.info(f"  - {error_type}: {count}")
            logging.info("------------------------------------")
        else:
            logging.info("No DLV2 execution errors detected during processing.")


        # --- Save Data ---
        logging.info(f"Preparing to save processed data to: {output_file_path}")
        output_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        write_jsonl(processed_data, str(output_file_path))
        logging.info(f"Successfully saved {len(processed_data)} processed entries to {output_file_path}.")

        # --- Report Consistency Check Stats ---
        if args.check_query_in_answer:
            logging.info("--- DLV2 Consistency Check Summary ---")
            # Total checked includes all categories
            total_checked = consistency_passed_count + consistency_failed_count + consistency_mismatch_after_recalc_count + consistency_skipped_count
            logging.info(f"Total items processed where consistency check was applicable: {total_checked}")
            logging.info(f"  - Passed: {consistency_passed_count}")
            logging.info(f"  - Failed (Unexpected Mismatch): {consistency_failed_count}")
            logging.info(f"  - Mismatch after Recalculation (is_positive+neg_query): {consistency_mismatch_after_recalc_count}") # Report new counter
            logging.info(f"  - Skipped/Not Applicable: {consistency_skipped_count}")
            logging.info("---------------------------------------")

        # --- Report Classification Stats ---
        logging.info("--- ASP Classification Summary ---")
        # Count based on successfully processed items where classification was likely attempted
        logging.info(f"Total items successfully processed (classification attempted): {len(processed_data)}")
        logging.info(f"  - is_positive (True count): {classification_counts['is_positive']}")
        logging.info(f"  - is_tight (True count): {classification_counts['is_tight']}")
        if args.use_disjunction:
            logging.info(f"  - is_disjunctive_stratified (True count): {classification_counts['is_disjunctive_stratified']}")
        else:
            # Only report is_stratified if use_disjunction is False (default)
            logging.info(f"  - is_stratified (True count): {classification_counts['is_stratified']}")
        logging.info("---------------------------------")


        logging.info("Script finished.")

    except FileNotFoundError: # Should be caught earlier, but good practice
        logging.error(f"Input file not found at {input_file_path}")
        sys.exit(1)
    except JsonUtilsError as e:
        logging.error(f"Error during JSONL processing: {e}")
        sys.exit(1)
    except Exception as e: # Catch other potential errors
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load data from a JSONL file, convert fact/rule dictionaries to DLV2 strings using asp_formatter, and save the augmented data to a new JSONL file."
    )

    # Input/Output Arguments
    parser.add_argument(
        "--input_path",
        type=str,
        default='datasets/symtex_filter_from_clean_data/20250423_132346_from_2025_04_23_13_15_seed42_n1000_minprob0.5.jsonl', # Use forward slashes
        help="Path to the input JSONL file containing dictionaries to be converted."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None, # Default to None, will generate path in main if not provided
        help="Path to save the output JSONL file. If not provided, defaults to 'datasets/symtex_dlv2/YYYY_MM_DD_HH_MM.jsonl'."
    )
    parser.add_argument(
        "--conceptnet_graph_path",
        type=str,
        default='datasets/conceptnet/bird_graph.graphml', # Default path to the graph
        help="Path to the ConceptNet graph file (GraphML format)."
    )


    # Formatting Arguments
    parser.add_argument(
        "--use_disjunction",
        action='store_true',
        help="Format multi-head rules using '|' in a single string (disjunction)."
    )
    parser.add_argument(
        "--use_random_conceptnet_predicates",
        action='store_true',
        help="Replace default predicate names (e.g., p, q) with unique random ConceptNet terms. Mutually exclusive with --use_related_conceptnet_predicates."
    )
    parser.add_argument(
        "--use_related_conceptnet_predicates",
        action='store_true',
        help="Generate related ConceptNet predicate names based on rule structure using graph traversal. Mutually exclusive with --use_random_conceptnet_predicates."
    )
    parser.add_argument(
        "--check_query_in_answer",
        action='store_true',
        help="Perform consistency check: compare original target_query_in_answerset with DLV2 result."
    )
    parser.add_argument(
        "--negation_removal_prob",
        type=float,
        default=1,
        help="Probability of removing negations ('strong negation', 'default negation') and setting graph data to 1s. (Default: 0.1)"
    )
    parser.add_argument(
        "--noise_removal_prob",
        type=float,
        default=0.3,
        help="Probability of removing noisy facts and rules before processing. (Default: 0.3)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42, # Or set a default like 42
        help="Random seed for reproducibility. (Default: None)"
    )


    args = parser.parse_args()

    # Add freeze_support() for Windows compatibility if creating executables
    # import multiprocessing
    # multiprocessing.freeze_support()

    main(args)
