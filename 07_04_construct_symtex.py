# 07_04_construct_symtex.py
# Loads data from multiple jsonl files in a directory using json_utils,
# deduplicates based on 'id' field by randomly selecting one entry per unique id,
# and saves the result to a new jsonl file.

import argparse
from pathlib import Path
import sys
import os
import json # Keep standard json for potential fallback or specific needs
import logging
from typing import List, Dict, Any, Optional # Ensure Optional is imported
from datetime import datetime
import random
from collections import Counter # Import Counter for counting source types

# --- Add project root to sys.path ---
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
# Adjust if your script lives elsewhere relative to 'src'
# Example: project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#          src_path = os.path.join(project_root, 'src')

print(f"[DEBUG] Detected project root (or script location): {project_root}")
if src_path not in sys.path:
    print(f"[DEBUG] Adding {src_path} to sys.path.")
    sys.path.insert(0, src_path)
else:
    print(f"[DEBUG] {src_path} already in sys.path.")
# --- End sys.path modification ---

# Import utility functions from src.utils.json_utils
try:
    # Specifically import the required functions
    from src.utils.json_utils import read_jsonl_parallel, write_jsonl, JsonUtilsError
    logging.info("Successfully imported read_jsonl_parallel and write_jsonl from utils.json_utils.")
except ImportError as e:
    print(f"Error: Could not import required functions/modules from utils.json_utils: {e}", file=sys.stderr)
    print(f"Please ensure the 'src' directory ({src_path}) is correctly added to sys.path and contains the necessary modules (json_utils.py).", file=sys.stderr)
    sys.exit(1)
except ModuleNotFoundError as e:
    print(f"Error: Module not found, likely 'src' is not in Python path or module missing: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_source_type_from_filename(filename: str) -> Optional[str]:
    """
    Determines the source type based on keywords in the filename.

    Checks for 'P_style', 'related_word', or 'random_word' within the filename string.

    :param filename: The name of the file.
    :type filename: str
    :return: The determined source type ('P_style', 'related_word', 'random_word')
             or None if no keyword is found.
    :rtype: Optional[str]
    """
    if "P_style" in filename:
        return "P_style"
    elif "related_word" in filename:
        return "related_word"
    elif "random_word" in filename:
        return "random_word"
    else:
        logging.debug(f"Filename '{filename}' does not contain known source type keywords.")
        return None # Return None if no specific type is identified

def load_all_jsonl_in_dir(input_dir_path: Path) -> List[Dict[str, Any]]:
    """
    Loads all data from .jsonl files within a specified directory using read_jsonl_parallel.
    Adds a 'source_type' field to each loaded object based on the source filename.

    Iterates through each .jsonl file in the directory, reads its content using
    the parallel utility function, determines the source type from the filename,
    adds it to each object, and aggregates all objects into a single list.

    :param input_dir_path: Path to the directory containing .jsonl files.
    :type input_dir_path: Path
    :return: A list containing all data loaded from the files, with an added 'source_type' field.
             Returns an empty list if no .jsonl files are found or if errors occur during loading.
    :rtype: List[Dict[str, Any]]
    """
    all_data: List[Dict[str, Any]] = []
    try:
        # Use Path.glob to find all .jsonl files
        jsonl_files = sorted(list(input_dir_path.glob('*.jsonl')))
        logging.info(f"Found {len(jsonl_files)} .jsonl files in {input_dir_path}.")
    except Exception as e:
        logging.error(f"Error accessing or listing files in directory {input_dir_path}: {e}", exc_info=True)
        return [] # Return empty list on directory access error

    if not jsonl_files:
        logging.warning(f"No .jsonl files found in {input_dir_path}. Returning empty list.")
        return []

    total_loaded_count = 0
    files_with_errors = 0
    for file_path in jsonl_files:
        logging.info(f"Attempting to load data from: {file_path} using read_jsonl_parallel")
        try:
            # Determine source type from filename
            filename = file_path.name
            source_type = get_source_type_from_filename(filename)
            logging.debug(f"Determined source_type '{source_type}' for file {filename}")

            # Call read_jsonl_parallel for each file
            # Pass the path as a string, as the function might expect str
            data_in_file: List[Dict[str, Any]] = read_jsonl_parallel(str(file_path))

            # Add source_type to each item
            processed_data_in_file: List[Dict[str, Any]] = []
            for item in data_in_file:
                if isinstance(item, dict): # Ensure item is a dictionary before modifying
                    item['source_type'] = source_type # Add/overwrite the source_type field
                    processed_data_in_file.append(item)
                else:
                    # Log if an item in the jsonl is not a dictionary
                    logging.warning(f"Skipping non-dictionary item found in {file_path}: {type(item)}")

            all_data.extend(processed_data_in_file)
            loaded_count = len(processed_data_in_file)
            total_loaded_count += loaded_count
            logging.info(f"Successfully loaded and processed {loaded_count} entries from {file_path}.")

        except JsonUtilsError as e: # Catch specific error from the utility
            logging.error(f"Failed to load data from {file_path} using json_utils: {e}")
            files_with_errors += 1
        except FileNotFoundError: # Should be caught by read_jsonl_parallel, but good practice
             logging.error(f"File not found during loading (unexpected): {file_path}.")
             files_with_errors += 1
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
            files_with_errors += 1
            # Decide whether to continue or exit based on requirements
            # For now, we log and continue to the next file

    logging.info(f"Finished loading files. Total entries loaded: {total_loaded_count}.")
    if files_with_errors > 0:
        logging.warning(f"Encountered errors while loading {files_with_errors} file(s).")

    return all_data


def deduplicate_data_by_id(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates a list of data dictionaries based on the 'id' field.

    Groups items by 'id' and randomly selects one entry for each unique ID.
    Items without an 'id' key are logged as warnings and excluded.
    The selected item retains all its original fields, including 'source_type' if added previously.

    :param data_list: The list of data dictionaries to deduplicate (potentially including 'source_type').
    :type data_list: List[Dict[str, Any]]
    :return: A list containing only unique data entries based on 'id', with one
             randomly chosen representative for each original ID. Each item retains its fields.
    :rtype: List[Dict[str, Any]]
    """
    unique_data_map: Dict[str, List[Dict[str, Any]]] = {}
    items_without_id_count = 0

    logging.info(f"Starting deduplication process on {len(data_list)} items...")
    for item in data_list:
        item_id = item.get('id') # Use .get() for safer access than item['id']
        if item_id is not None:
            # Ensure id is string for consistent dictionary keys, handle potential non-string IDs
            item_id_str = str(item_id)
            if item_id_str not in unique_data_map:
                unique_data_map[item_id_str] = []
            unique_data_map[item_id_str].append(item)
        else:
            items_without_id_count += 1
            # Log only once per N items or summarize at the end to avoid log spam
            if items_without_id_count <= 10: # Log first few occurrences
                 logging.warning(f"Item found without an 'id' key (occurrence {items_without_id_count}): {str(item)[:200]}...") # Log truncated item

    if items_without_id_count > 0:
        logging.warning(f"Found a total of {items_without_id_count} items without an 'id' key. These items will be excluded from the output.")

    final_data: List[Dict[str, Any]] = []
    num_unique_ids = len(unique_data_map)
    logging.info(f"Found {num_unique_ids} unique IDs.")

    processed_ids = 0
    for item_id, duplicates in unique_data_map.items():
        if duplicates: # Ensure the list is not empty
            try:
                selected_item = random.choice(duplicates)
                final_data.append(selected_item)
            except IndexError: # Should not happen if duplicates is not empty, but safety check
                 logging.warning(f"IndexError choosing from duplicates for ID '{item_id}'. List was likely empty unexpectedly.")
            except Exception as choice_err:
                 logging.error(f"Error during random choice for ID '{item_id}': {choice_err}", exc_info=True)
        else:
             # This case should ideally not be reached if items are added correctly
             logging.warning(f"Found unique ID '{item_id}' but its duplicate list was empty. Skipping.")

        processed_ids += 1
        if processed_ids % 1000 == 0: # Log progress periodically
            logging.info(f"Processed {processed_ids}/{num_unique_ids} unique IDs for selection.")


    logging.info(f"Finished deduplication. Selected {len(final_data)} unique items from {num_unique_ids} unique IDs.")
    return final_data


def main(args: argparse.Namespace) -> None:
    """
    Main execution function for the script.

    Handles argument parsing, orchestrates loading (using json_utils, now adding source_type),
    deduplication (retaining source_type), counting source types, and saving of the SymTex data.

    :param args: Command-line arguments parsed by argparse.
    :type args: argparse.Namespace
    :return: None
    :rtype: None
    """
    # --- Set Random Seed ---
    if args.seed is not None:
        try:
            seed_value = int(args.seed)
            random.seed(seed_value)
            logging.info(f"Random seed set to: {seed_value}")
        except ValueError:
            logging.warning(f"Invalid seed value '{args.seed}'. Using default random state.")
    else:
        logging.info("No random seed provided. Using default random state.")

    input_dir_path: Path = Path(args.input_dir)
    output_dir_path: Path = Path(args.output_dir)

    # --- Validate Input Path ---
    if not input_dir_path.is_dir():
        logging.error(f"Input directory not found or is not a directory: {input_dir_path}")
        sys.exit(1)
    logging.info(f"Input data directory: {input_dir_path}")

    # --- Ensure Output Directory Exists ---
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory ensured: {output_dir_path}")
    except OSError as e:
        logging.error(f"Failed to create output directory {output_dir_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error creating output directory {output_dir_path}: {e}", exc_info=True)
        sys.exit(1)


    # --- Load Data using the updated function ---
    all_loaded_data = load_all_jsonl_in_dir(input_dir_path)

    if not all_loaded_data:
        logging.warning("No data was loaded from the input directory. Exiting.")
        sys.exit(0) # Exit gracefully if no data

    # --- Deduplicate Data ---
    final_unique_data = deduplicate_data_by_id(all_loaded_data)

    if not final_unique_data:
        logging.warning("No unique data found after deduplication (possibly all items lacked 'id' or input was empty after loading). Exiting.")
        sys.exit(0) # Exit gracefully if no data to save

    # --- Determine Output Filename ---
    try:
        input_dir_name: str = input_dir_path.name # Gets the last component of the path (folder name)
        if not input_dir_name: # Handle edge case like root directory "/"
             input_dir_name = "root_input"
             logging.warning("Input directory path seems unusual (e.g., root). Using 'root_input' for filename.")

        timestamp: str = datetime.now().strftime("%Y_%m_%d_%H_%M") # Format: YYYY_MM_DD_HH_MM
        output_filename: str = f"{input_dir_name}_{timestamp}.jsonl"
        output_file_path: Path = output_dir_path / output_filename
        logging.info(f"Output data file will be: {output_file_path}")
    except Exception as e:
        logging.error(f"Error determining output filename: {e}", exc_info=True)
        sys.exit(1)


    # --- Save Processed Data using write_jsonl from json_utils ---
    logging.info(f"Attempting to save {len(final_unique_data)} unique entries to: {output_file_path}")
    try:
        # Use the imported utility function write_jsonl
        # Ensure the utility function handles Path objects or convert to string
        write_jsonl(final_unique_data, str(output_file_path))
        logging.info(f"Successfully saved data to {output_file_path}.")
    except JsonUtilsError as e: # Catch specific error from the utility
        logging.error(f"Failed to save data using write_jsonl utility: {e}")
        sys.exit(1)
    except NameError:
         # This shouldn't happen with the explicit import, but good fallback check
         logging.error("Failed to save processed data: 'write_jsonl' function not found. Check import from utils.json_utils.")
         sys.exit(1)
    except IOError as e:
         logging.error(f"IOError saving data to {output_file_path}: {e}")
         sys.exit(1)
    except Exception as e: # Catch other potential errors during saving
        logging.error(f"An unexpected error occurred during saving: {e}", exc_info=True)
        sys.exit(1)

    # --- Count and Log Source Types ---
    try:
        source_type_counts = Counter(item.get('source_type', 'Unknown') for item in final_unique_data if isinstance(item, dict))
        logging.info("--- Final Dataset Source Type Distribution ---")
        if source_type_counts:
            for source_type, count in source_type_counts.items():
                print(f"  {source_type}: {count}")
        else:
            logging.info("  No source types found or counted in the final dataset.")
        logging.info("---------------------------------------------")
    except Exception as count_err:
        logging.error(f"An error occurred while counting source types: {count_err}", exc_info=True)
        # Continue even if counting fails, as saving is more critical

    # --- Count and Log Boolean Field Statistics ---
    logging.info("--- Boolean Field Statistics ---")
    boolean_keys_to_count = [
        'is_positive',
        'is_tight',
        'is_disjunctive_stratified',
        'is_stratified'
    ]
    boolean_stats = {key: {'True': 0, 'False': 0, 'Missing': 0} for key in boolean_keys_to_count}

    for item in final_unique_data:
        if isinstance(item, dict):
            for key in boolean_keys_to_count:
                value = item.get(key) # Safely get the value
                if value is True:
                    boolean_stats[key]['True'] += 1
                elif value is False:
                    boolean_stats[key]['False'] += 1
                else:
                    # Count missing or non-boolean values
                    boolean_stats[key]['Missing'] += 1
                    if value is not None: # Log if value exists but isn't True/False
                        logging.debug(f"Item ID {item.get('id', 'N/A')} has non-boolean value '{value}' for key '{key}'. Counted as Missing.")


    # Print the collected statistics
    for key, counts in boolean_stats.items():
        total = counts['True'] + counts['False'] + counts['Missing']
        print(f"  {key}:")
        print(f"    True: {counts['True']}")
        print(f"    False: {counts['False']}")
        # Optionally print missing count if it's useful
        if counts['Missing'] > 0:
             print(f"    Missing/Non-Boolean: {counts['Missing']}")
        print(f"    (Total items checked for this key: {total})") # Should match len(final_unique_data)

    logging.info("---------------------------------")
    # --- End Boolean Field Statistics ---


    logging.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Constructs the final SymTex dataset by loading data from multiple JSONL files in a directory (using json_utils), adding a 'source_type' based on filename, deduplicating based on 'id' by random selection (retaining source_type), saving the unique entries to a new JSONL file, and logging the final source type distribution.", # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # Input/Output Arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        default='datasets/a_symtex_task_fact_state_querying',
        help="Path to the input directory containing source .jsonl files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./datasets/symtex_final',
        help="Directory to save the final deduplicated output JSONL file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None, # Default to None, meaning no fixed seed unless specified
        help="Optional random seed for reproducibility of the random selection during deduplication."
    )

    args = parser.parse_args()
    main(args)
