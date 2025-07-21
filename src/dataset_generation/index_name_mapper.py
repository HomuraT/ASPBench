# -*- coding: utf-8 -*-
"""
Utilities for mapping between indices and ConceptNet names/URIs,
and potentially completing mappings based on graph structure.
"""

import networkx as nx
import logging
import re # Added import
from typing import Dict, List, Set, Optional, Union, Any # Added Any
# Import get_random_node from the correct location
from src.conceptnet_utils.graph_operations import get_random_node # Added import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def complete_uris_by_edge(graph: nx.DiGraph, index_dicts: Union[List[Dict[str, List[int]]], Dict[str, List[int]]], uri_dict: Dict[int, str]) -> None:
    """
    Attempts to find and fill missing ConceptNet URIs in uri_dict for indices listed
    in one or more index_dicts ('head' and 'body') by leveraging existing body -> head
    relationships present in the graph.

    The function iteratively searches for connections within each provided index_dict:
    1. If a body index's URI is missing but a connected head index's URI is known,
       it looks for predecessors of the head URI in the graph to assign as the body URI.
    2. If a head index's URI is missing but a connected body index's URI is known,
       it looks for successors of the body URI in the graph to assign as the head URI.

    The process repeats within each index_dict until no new URIs can be added in a full
    iteration for that specific dictionary. The shared uri_dict is modified in place
    across all dictionaries processed.

    :param graph: The ConceptNet graph (networkx DiGraph).
    :type graph: nx.DiGraph
    :param index_dicts: A single dictionary or a list of dictionaries, each mapping
                        'head' and 'body' to lists of integer indices.
                        Example (single): {'head': [0, 1], 'body': [2, 3]}
                        Example (list): [{'head': [0], 'body': [1]}, {'head': [1], 'body': [2]}]
    :type index_dicts: Union[List[Dict[str, List[int]]], Dict[str, List[int]]]
    :param uri_dict: Dictionary mapping some indices to ConceptNet URIs. This dictionary
                     will be modified in place and shared across the processing of all
                     index_dicts.
                     Example: {0: '/c/en/cat', 2: '/c/en/mammal'}
    :type uri_dict: Dict[int, str]
    :return: None
    :rtype: None
    """
    if not graph:
        logging.debug("Graph is empty or None. Cannot complete URIs.") # Changed to debug
        return
    if not index_dicts:
        logging.debug("index_dicts is empty or None. Cannot complete URIs.") # Changed to debug
        return

    # Ensure index_dicts is always a list for uniform processing
    if isinstance(index_dicts, dict):
        index_dicts_list = [index_dicts]
    else:
        index_dicts_list = index_dicts

    # --- Outer loop to process each dictionary in the list ---
    for i, index_dict in enumerate(index_dicts_list):
        logging.debug(f"Processing index_dict #{i}: {index_dict}")

        if not index_dict or ('head' not in index_dict and 'body' not in index_dict):
            logging.debug(f"Skipping index_dict #{i} as it's empty or missing 'head'/'body' keys.") # Changed to debug
            continue

        head_indices = set(index_dict.get('head', []))
        body_indices = set(index_dict.get('body', []))
        # Consider only indices relevant to the *current* dictionary for missing check
        current_all_indices = head_indices.union(body_indices)

        # --- Inner loop for iterative completion within the current index_dict ---
        uri_added_in_pass = True
        while uri_added_in_pass:
            uri_added_in_pass = False

            # Find indices from the *current* dict that are still missing URIs
            missing_indices = sorted(list(current_all_indices - set(uri_dict.keys())))

            if not missing_indices:
                logging.debug(f"No missing indices found for index_dict #{i} in this pass.")
                break # Exit inner loop if all indices for this dict have URIs

            logging.debug(f"Starting pass for index_dict #{i}. Missing indices: {missing_indices}")

            for missing_idx in missing_indices:
                # Double-check if it was filled in the current pass by a previous iteration
                if missing_idx in uri_dict:
                    continue

                found_uri_for_missing_idx = False

                # --- Try Case A: Find URI for missing_idx assuming it's a BODY index ---
                # Check if missing_idx is a body index *in the current index_dict*
                if missing_idx in body_indices:
                    # Check against head indices *in the current index_dict*
                    for head_idx in head_indices:
                        if head_idx in uri_dict: # Check if the corresponding head URI is known
                            head_uri = uri_dict[head_idx]
                            if graph.has_node(head_uri):
                                # Look for predecessors (potential body nodes)
                                for potential_body_uri in graph.predecessors(head_uri):
                                    # Check if this potential URI is valid and not already assigned *globally*
                                    if graph.has_node(potential_body_uri) and potential_body_uri not in uri_dict.values():
                                        uri_dict[missing_idx] = potential_body_uri
                                        logging.debug(f"[Dict #{i}] Found URI for body index {missing_idx} ('{potential_body_uri}') via known head index {head_idx} ('{head_uri}')") # Changed to debug
                                        uri_added_in_pass = True
                                        found_uri_for_missing_idx = True
                                        break # Stop searching predecessors for this head_idx
                            else:
                                logging.debug(f"[Dict #{i}] Known head URI '{head_uri}' (for index {head_idx}) not found in graph.")

                        if found_uri_for_missing_idx:
                            break # Stop checking other head_idx for this missing_idx

                # --- Try Case B: Find URI for missing_idx assuming it's a HEAD index ---
                # Only try this if Case A didn't find a URI
                # Check if missing_idx is a head index *in the current index_dict*
                if not found_uri_for_missing_idx and missing_idx in head_indices:
                     # Check against body indices *in the current index_dict*
                    for body_idx in body_indices:
                        if body_idx in uri_dict: # Check if the corresponding body URI is known
                            body_uri = uri_dict[body_idx]
                            if graph.has_node(body_uri):
                                # Look for successors (potential head nodes)
                                for potential_head_uri in graph.successors(body_uri):
                                    # Check if this potential URI is valid and not already assigned *globally*
                                    if graph.has_node(potential_head_uri) and potential_head_uri not in uri_dict.values():
                                        uri_dict[missing_idx] = potential_head_uri
                                        logging.debug(f"[Dict #{i}] Found URI for head index {missing_idx} ('{potential_head_uri}') via known body index {body_idx} ('{body_uri}')") # Changed to debug
                                        uri_added_in_pass = True
                                        found_uri_for_missing_idx = True
                                        break # Stop searching successors for this body_idx
                            else:
                                logging.debug(f"[Dict #{i}] Known body URI '{body_uri}' (for index {body_idx}) not found in graph.")

                        if found_uri_for_missing_idx:
                            break # Stop checking other body_idx for this missing_idx

            if not uri_added_in_pass:
                logging.debug(f"No new URIs added for index_dict #{i} in this pass. Inner iteration finished.")
        # --- End of inner while loop ---
    # --- End of outer for loop ---


def extract_concept_from_uri(uri: str) -> Optional[str]:
    """
    Extracts the concept name from a ConceptNet URI.
    e.g., /c/en/bird -> bird
          /c/en/run_fast/v/example -> run_fast
    Handles potential variations. Returns None if extraction fails.

    :param uri: The ConceptNet URI string.
    :type uri: str
    :return: The extracted concept name or None.
    :rtype: Optional[str]
    """
    if not isinstance(uri, str):
        return None
    parts = uri.strip('/').split('/')
    if len(parts) >= 3:
        # Basic check for language code (e.g., 'en', 'fr')
        if len(parts[1]) == 2:
            # Replace underscores with spaces for better readability if needed,
            # but for predicate names, underscores might be preferable.
            # return parts[2].replace('_', ' ')
            return parts[2] # Return the third part as is
    return None # Return None if format is unexpected


def generate_unique_conceptnet_names(graph: nx.DiGraph, indices: List[int], max_retries_per_index: int = 10) -> Dict[int, str]:
    """
    Generates a mapping from integer indices to unique random ConceptNet concept names.
    Ensures that each index gets a unique name extracted from a unique graph node URI.

    :param graph: The ConceptNet graph (nx.DiGraph).
    :type graph: nx.DiGraph
    :param indices: A list of integer indices to map.
    :type indices: List[int]
    :param max_retries_per_index: Max attempts to find a unique node/name for each index.
    :type max_retries_per_index: int
    :return: A dictionary mapping indices to unique concept names. Returns empty dict if graph or indices are empty.
    :rtype: Dict[int, str]
    """
    if not graph or not indices:
        logging.debug("Graph or indices list is empty. Returning empty name map.") # Changed to debug
        return {}

    name_map: Dict[int, str] = {}
    assigned_uris: Set[str] = set()
    assigned_names: Set[str] = set() # Track assigned names for uniqueness

    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        logging.debug("Graph has no nodes. Cannot generate names.") # Changed to debug
        return {}

    max_total_retries = num_nodes * 2 # Limit overall attempts based on graph size

    for index in indices:
        found_unique = False
        for attempt in range(max_retries_per_index):
            # Limit total attempts to prevent excessive looping on dense graphs or difficult constraints
            if len(assigned_uris) >= num_nodes:
                 logging.debug(f"All nodes in the graph seem to be assigned or checked. Cannot find more unique URIs for index {index}.") # Changed to debug
                 break # Break inner loop if all nodes are exhausted

            random_uri = get_random_node(graph)
            if random_uri is None: # Should not happen if graph is not empty and num_nodes > 0
                 logging.debug(f"get_random_node returned None for index {index} despite non-empty graph. Skipping.") # Changed to debug
                 break # Break inner loop for this index

            if random_uri not in assigned_uris:
                concept_name = extract_concept_from_uri(random_uri)
                # Check if concept_name is valid (not None) and unique
                if concept_name and concept_name not in assigned_names:
                    name_map[index] = concept_name
                    assigned_uris.add(random_uri)
                    assigned_names.add(concept_name)
                    found_unique = True
                    logging.debug(f"Assigned index {index} -> name '{concept_name}' (from URI: {random_uri})")
                    break # Found unique name for this index
                # else: # Log why it failed (either bad URI format or name collision)
                #     if not concept_name:
                #         logging.debug(f"Attempt {attempt+1} for index {index}: URI '{random_uri}' failed extraction.")
                #     else:
                #         logging.debug(f"Attempt {attempt+1} for index {index}: Name '{concept_name}' from URI '{random_uri}' already assigned.")


        if not found_unique:
            logging.debug(f"Could not find a unique ConceptNet name for index {index} after {max_retries_per_index} retries. Skipping this index.") # Changed to debug
            # Optionally assign a default placeholder like f"ERR_IDX_{index}" or leave it out

    if len(name_map) != len(indices):
         logging.debug(f"Failed to generate unique names for all indices. Requested: {len(indices)}, Generated: {len(name_map)}") # Changed to debug

    return name_map


# Example Usage (can be added for testing)
if __name__ == '__main__':
    # Create a sample graph
    G = nx.DiGraph()
    G.add_node("/c/en/bird", name="bird")
    G.add_node("/c/en/can_fly", name="can fly")
    G.add_node("/c/en/animal", name="animal")
    G.add_node("/c/en/penguin", name="penguin")
    G.add_node("/c/en/is_a", name="is a") # Relation node example (less common)

    G.add_edge("/c/en/bird", "/c/en/can_fly", relation="/r/CapableOf") # body -> head
    G.add_edge("/c/en/bird", "/c/en/animal", relation="/r/IsA")       # body -> head
    G.add_edge("/c/en/penguin", "/c/en/bird", relation="/r/IsA")      # body -> head

    # Create a sample graph
    G = nx.DiGraph()
    G.add_node("/c/en/bird", name="bird")
    G.add_node("/c/en/can_fly", name="can fly")
    G.add_node("/c/en/animal", name="animal")
    G.add_node("/c/en/penguin", name="penguin")
    G.add_node("/c/en/is_a", name="is a") # Relation node example (less common)
    G.add_node("/c/en/fish", name="fish")
    G.add_node("/c/en/can_swim", name="can swim")

    G.add_edge("/c/en/bird", "/c/en/can_fly", relation="/r/CapableOf") # body -> head
    G.add_edge("/c/en/bird", "/c/en/animal", relation="/r/IsA")       # body -> head
    G.add_edge("/c/en/penguin", "/c/en/bird", relation="/r/IsA")      # body -> head
    G.add_edge("/c/en/fish", "/c/en/can_swim", relation="/r/CapableOf") # body -> head
    G.add_edge("/c/en/fish", "/c/en/animal", relation="/r/IsA")       # body -> head


    # --- Test Case 1: Single dictionary (backward compatibility) ---
    print("--- Test Case 1: Single dictionary (Find missing body) ---")
    indices1 = {'head': [1], 'body': [0]} # Expect 0=bird based on head=1=can_fly
    uris1 = {1: '/c/en/can_fly'}
    print(f"Initial URIs: {uris1}")
    complete_uris_by_edge(G, indices1, uris1)
    print(f"Completed URIs: {uris1}") # Expected: {1: '/c/en/can_fly', 0: '/c/en/bird'}

    # --- Test Case 2: List of dictionaries, sequential completion ---
    print("\n--- Test Case 2: List of dictionaries, sequential completion ---")
    # Dict 1: Find head=1 based on body=0=penguin -> head=1=bird
    # Dict 2: Find head=2 based on body=1 (now bird) -> head=2=can_fly or animal
    indices_list2 = [
        {'head': [1], 'body': [0]}, # Find head 1 from body 0
        {'head': [2], 'body': [1]}  # Find head 2 from body 1 (which should be found above)
    ]
    uris2 = {0: '/c/en/penguin'}
    print(f"Initial URIs: {uris2}")
    complete_uris_by_edge(G, indices_list2, uris2)
    # Expected: {0: '/c/en/penguin', 1: '/c/en/bird', 2: '/c/en/can_fly'} (or /c/en/animal)
    print(f"Completed URIs: {uris2}")

    # --- Test Case 3: List of dictionaries, more complex dependencies ---
    print("\n--- Test Case 3: List of dictionaries, complex dependencies ---")
    # Dict 1: Find body=0 based on head=1=can_fly -> body=0=bird
    # Dict 2: Find head=2 based on body=3=fish -> head=2=can_swim or animal
    # Dict 3: Find head=4 based on body=0 (now bird) -> head=4=can_fly or animal (one already taken)
    indices_list3 = [
        {'head': [1], 'body': [0]}, # Find body 0 from head 1
        {'head': [2], 'body': [3]}, # Find head 2 from body 3
        {'head': [4], 'body': [0]}  # Find head 4 from body 0 (found in first dict)
    ]
    uris3 = {1: '/c/en/can_fly', 3: '/c/en/fish'}
    print(f"Initial URIs: {uris3}")
    complete_uris_by_edge(G, indices_list3, uris3)
    # Expected: {1: '/c/en/can_fly', 3: '/c/en/fish', 0: '/c/en/bird', 2: '/c/en/can_swim', 4: '/c/en/animal'} (or 2=animal, 4=can_fly)
    print(f"Completed URIs: {uris3}")

    # --- Test Case 4: List with empty/invalid dict ---
    print("\n--- Test Case 4: List with empty/invalid dict ---")
    indices_list4 = [
        {'head': [1], 'body': [0]}, # Find body 0 from head 1 = can_fly -> 0=bird
        {},                         # Empty dict, should be skipped
        {'head': [2]}               # Missing body key, should be skipped
    ]
    uris4 = {1: '/c/en/can_fly'}
    print(f"Initial URIs: {uris4}")
    complete_uris_by_edge(G, indices_list4, uris4)
    print(f"Completed URIs: {uris4}") # Expected: {1: '/c/en/can_fly', 0: '/c/en/bird'}

    # --- Test Case 5: Empty graph ---
    print("\n--- Test Case 5: Empty graph ---")
    empty_G = nx.DiGraph()
    indices_list5 = [{'head': [1], 'body': [0]}]
    uris5 = {1: '/c/en/something'}
    print(f"Initial URIs: {uris5}")
    complete_uris_by_edge(empty_G, indices_list5, uris5)
    print(f"Completed URIs: {uris5}") # Expected: {1: '/c/en/something'}

    # --- Test Case 6: Empty list of dicts ---
    print("\n--- Test Case 6: Empty list of dicts ---")
    indices_list6 = []
    uris6 = {0: '/c/en/bird'}
    print(f"Initial URIs: {uris6}")
    complete_uris_by_edge(G, indices_list6, uris6)
    print(f"Completed URIs: {uris6}") # Expected: {0: '/c/en/bird'}
