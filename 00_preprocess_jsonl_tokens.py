import argparse
import json
import os
from pathlib import Path
import tiktoken # Ensure this is importable
from src.utils.json_utils import write_jsonl # Added import

# --- Constants for new/updated fields ---
FIELD_TOTAL_COST = "total_cost"
FIELD_TOTAL_TOKENS = "total_tokens"
FIELD_PROMPT_TOKENS = "prompt_tokens"
FIELD_PROMPT_TOKENS_CACHED = "prompt_tokens_cached"
FIELD_COMPLETION_TOKENS = "completion_tokens"
FIELD_REASONING_TOKENS = "reasoning_tokens"
FIELD_SUCCESSFUL_REQUESTS = "successful_requests"

# --- Default values for placeholder fields ---
DEFAULT_VALUE_TOTAL_COST0 = 0
DEFAULT_VALUE_PROMPT_TOKENS_CACHED0 = 0
DEFAULT_VALUE_REASONING_TOKENS0 = 0
DEFAULT_VALUE_SUCCESSFUL_REQUESTS1 = 1


def calculate_and_update_tokens_for_item(item_dict: dict, encoder: tiktoken.Encoding) -> None:
    """
    Calculates token counts for 'input' and 'response' fields in item_dict
    and updates/adds the specified token-related fields.
    Modifications are made in-place to item_dict.

    :param item_dict: The dictionary (an item from llm_input_and_response list) to process.
    :type item_dict: dict
    :param encoder: The tiktoken encoder to use.
    :type encoder: tiktoken.Encoding
    :return: None
    :rtype: None
    """
    if not isinstance(item_dict, dict):
        print(f"Warning: Expected a dictionary for token calculation, got {type(item_dict)}. Skipping this item.")
        return

    # Check if token fields already exist
    if (FIELD_PROMPT_TOKENS in item_dict or
            FIELD_COMPLETION_TOKENS in item_dict or
            FIELD_TOTAL_TOKENS in item_dict):
        # print(f"Warning: Token fields already exist in an item. Skipping token calculation for this item: {item_dict}")
        return # Skip processing this item as it seems to have token data already

    # Process 'input' field
    input_value = item_dict.get("input")
    prompt_tokens = 0
    if isinstance(input_value, str):
        if input_value:  # Tiktoken on non-empty string
            try:
                prompt_tokens = len(encoder.encode(input_value))
            except Exception as e:
                print(f"Warning: Tiktoken failed to encode input: '{input_value[:50]}...'. Error: {e}. Setting prompt_tokens to 0.")
                # prompt_tokens remains 0
        # else: input_value is an empty string "", len(encoder.encode("")) is 0, so prompt_tokens is correctly 0
    elif input_value is not None:  # Field exists but is not a string (and not None)
        print(f"Warning: 'input' field found but is not a string (type: {type(input_value)}). Setting prompt_tokens to 0.")
    # If input_value is None (field entirely missing), prompt_tokens remains 0.

    # Process 'response' field
    response_value = item_dict.get("response")
    completion_tokens = 0
    if isinstance(response_value, str):
        if response_value:  # Tiktoken on non-empty string
            try:
                completion_tokens = len(encoder.encode(response_value))
            except Exception as e:
                print(f"Warning: Tiktoken failed to encode response: '{response_value[:50]}...'. Error: {e}. Setting completion_tokens to 0.")
                # completion_tokens remains 0
        # else: response_value is an empty string "", len(encoder.encode("")) is 0, so completion_tokens is correctly 0
    elif response_value is not None:  # Field exists but is not a string (and not None)
        print(f"Warning: 'response' field found but is not a string (type: {type(response_value)}). Setting completion_tokens to 0.")
    # If response_value is None (field entirely missing), completion_tokens remains 0.

    item_dict[FIELD_PROMPT_TOKENS] = prompt_tokens
    item_dict[FIELD_COMPLETION_TOKENS] = completion_tokens
    item_dict[FIELD_TOTAL_TOKENS] = prompt_tokens + completion_tokens
    
    item_dict[FIELD_TOTAL_COST] = DEFAULT_VALUE_TOTAL_COST0
    item_dict[FIELD_PROMPT_TOKENS_CACHED] = DEFAULT_VALUE_PROMPT_TOKENS_CACHED0
    item_dict[FIELD_REASONING_TOKENS] = DEFAULT_VALUE_REASONING_TOKENS0
    item_dict[FIELD_SUCCESSFUL_REQUESTS] = DEFAULT_VALUE_SUCCESSFUL_REQUESTS1


def process_jsonl_file(file_path: Path, encoder: tiktoken.Encoding) -> None:
    """
    Reads a JSONL file, processes each line to calculate and add token counts
    to items in 'llm_input_and_response' lists, and overwrites the original file
    with only the valid, processed JSON objects.

    :param file_path: Path to the .jsonl file.
    :type file_path: Path
    :param encoder: The tiktoken encoder to use.
    :type encoder: tiktoken.Encoding
    :return: None
    :rtype: None
    """
    print(f"Processing file: {file_path}...")
    
    try:
        with file_path.open('r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}. Skipping this file.")
        return

    if not lines:
        print(f"File {file_path} is empty. Skipping.")
        return

    valid_json_data_to_write = [] # Store only valid JSON objects
    file_had_updatable_content = False # To track if any content was actually modified

    for i, line_content in enumerate(lines):
        line_strip = line_content.strip()
        if not line_strip: # Skip empty or whitespace-only lines
            continue

        try:
            data_obj = json.loads(line_strip)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from line {i+1} in {file_path}. Line: '{line_strip[:100]}...'. This line will be skipped in the output.")
            continue # Skip lines that are not valid JSON
        
        llm_input_output_list = data_obj.get("llm_input_and_response")
        
        item_was_updated_in_this_data_obj = False
        if isinstance(llm_input_output_list, list):
            for llm_item_index, llm_item in enumerate(llm_input_output_list):
                if isinstance(llm_item, dict):
                    # Store initial state or a specific field to check for changes if needed
                    # For now, calculate_and_update_tokens_for_item directly modifies llm_item
                    # and sets global flags like DEFAULT_VALUE_SUCCESSFUL_REQUESTS1 etc.
                    # We are checking if any item was *eligible* for update by calculate_and_update_tokens_for_item
                    # The function calculate_and_update_tokens_for_item itself doesn't return a status
                    # So, we rely on file_had_updatable_content which is set inside.
                    # Let's assume if we call it, it's an attempt to update.
                    # The actual check for *newly added tokens* vs *pre-existing tokens* is inside calculate_and_update_tokens_for_item
                    calculate_and_update_tokens_for_item(llm_item, encoder)
                    item_was_updated_in_this_data_obj = True # Mark that processing happened for this data_obj
                else:
                    print(f"Warning: Item at index {llm_item_index} in 'llm_input_and_response' is not a dictionary on line {i+1} in {file_path}. Item: {str(llm_item)[:100]}.")
        elif llm_input_output_list is not None: # It exists but is not a list
             print(f"Warning: 'llm_input_and_response' is not a list on line {i+1} in {file_path} (type: {type(llm_input_output_list)}). Token calculation skipped for this field.")

        if item_was_updated_in_this_data_obj: # If any llm_item was processed
            file_had_updatable_content = True

        valid_json_data_to_write.append(data_obj) # Add the (potentially) modified Python dict

    # Write back to file using json_utils.write_jsonl
    try:
        write_jsonl(valid_json_data_to_write, str(file_path)) # Ensure file_path is string

        num_written_objects = len(valid_json_data_to_write)
        if num_written_objects > 0:
            if file_had_updatable_content: # If any item within any valid JSON object was processed by calculate_...
                print(f"Finished processing and updated {file_path}. {num_written_objects} valid JSON object(s) written.")
            else: # Valid JSON objects written, but none of them triggered the "update" condition for tokens
                print(f"Finished processing {file_path}. {num_written_objects} valid JSON object(s) written. No items required token calculations or updates (e.g., tokens already present or no applicable fields).")
        else: # No valid JSON objects were found to write (original file might have been empty or all non-JSON)
            print(f"Finished processing {file_path}. No valid JSON objects found to write. The file may now be empty if it previously contained only non-JSON lines or was empty.")
            
    except Exception as e:
        print(f"Error writing processed data back to {file_path} using json_utils: {e}")


def main():
    """
    Main function to parse arguments and orchestrate the processing.
    """
    parser = argparse.ArgumentParser(
        description="Scans a folder for .jsonl files, calculates token counts for 'input' and 'response' "
                    "fields within each item of 'llm_input_and_response' lists (which are expected to be lists of dictionaries), "
                    "and updates the files in-place. Adds specified token-related fields to these dictionaries.",
        formatter_class=argparse.RawTextHelpFormatter # Allows for better formatting of help text
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="experiments/answer_set_generation/w_few_shot",
        help="Path to the folder containing .jsonl files to process."
    )
    parser.add_argument(
        "--tiktoken_encoding",
        type=str,
        default="cl100k_base",
        help="The tiktoken encoding to use (e.g., 'cl100k_base', 'p50k_base'). Default: cl100k_base."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, search for .jsonl files recursively in subdirectories."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    if not input_dir.is_dir():
        print(f"Error: Input folder '{input_dir}' does not exist or is not a directory.")
        return

    try:
        encoder = tiktoken.get_encoding(args.tiktoken_encoding)
        print(f"Using tiktoken encoding: {args.tiktoken_encoding}")
    except ImportError:
        print("Error: Tiktoken library not found. Please install it using 'pip install tiktoken'.")
        return
    except Exception as e:
        print(f"Error: Could not initialize tiktoken encoder '{args.tiktoken_encoding}'. Ensure tiktoken is installed and the encoding name is correct. Error: {e}")
        return

    print(f"\\nWARNING: This script will attempt to modify .jsonl files in-place in the folder '{input_dir}' {'recursively' if args.recursive else ''}.")
    print("It is STRONGLY recommended to back up your data before proceeding.")
    
    # User confirmation
    while True:
        confirm = input("Do you want to continue? (yes/no): ").strip().lower()
        if confirm == 'yes':
            break
        elif confirm == 'no':
            print("Operation cancelled by the user.")
            return
        else:
            print("Please type 'yes' or 'no'.")
    print("-" * 30)

    file_pattern = "*.jsonl"
    if args.recursive:
        print(f"Recursively searching for '{file_pattern}' files in '{input_dir}' and its subdirectories...")
        jsonl_files = list(input_dir.rglob(file_pattern))
    else:
        print(f"Searching for '{file_pattern}' files in '{input_dir}' (non-recursively)...")
        jsonl_files = list(input_dir.glob(file_pattern))

    if not jsonl_files:
        print(f"No .jsonl files found matching '{file_pattern}' in the specified location.")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s) to process.")
    processed_count = 0
    for file_path in jsonl_files:
        process_jsonl_file(file_path, encoder)
        processed_count +=1
        print("-" * 30)

    print(f"All {processed_count} targeted file(s) have been processed.")

if __name__ == "__main__":
    main() 