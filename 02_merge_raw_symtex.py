# 02_merge_raw_symtex.py
import argparse
import os
import glob
import logging
from datetime import datetime
import concurrent.futures
from typing import List, Any, Optional
from tqdm import tqdm # Import tqdm
from src.utils import json_utils # Assuming src is in the python path or relative import works

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_single_jsonl(file_path: str) -> Optional[List[Any]]:
    """Reads a single JSONL file and handles errors."""
    try:
        logging.debug(f"Reading file: {file_path}...")
        data = json_utils.read_jsonl(file_path)
        logging.debug(f"Successfully read {len(data)} records from {os.path.basename(file_path)}.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found during processing: {file_path}. Skipping.")
    except json_utils.json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in file {file_path}: {e}. Skipping.")
    except json_utils.JsonUtilsError as e:
        logging.error(f"Error reading file {file_path} using json_utils: {e}. Skipping.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading {file_path}: {e}. Skipping.")
    return None # Return None on error

def merge_jsonl_files(input_dir: str, output_dir: str, num_workers: Optional[int] = None):
    """
    Merges all .jsonl files from an input directory into a single timestamped
    .jsonl file in the output directory.

    Args:
        input_dir: The directory containing the .jsonl files to merge.
        output_dir: The directory where the merged file will be saved.
    """
    logging.info(f"Starting merge process for directory: {input_dir}")

    # Validate input directory
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory not found or is not a directory: {input_dir}")
        return

    # Find all .jsonl files in the input directory
    jsonl_files = glob.glob(os.path.join(input_dir, '*.jsonl'))

    if not jsonl_files:
        logging.warning(f"No .jsonl files found in directory: {input_dir}")
        return

    logging.info(f"Found {len(jsonl_files)} .jsonl files to merge.")

    merged_data = []
    total_records = 0

    # Use ThreadPoolExecutor for parallel reading
    # If num_workers is None, ThreadPoolExecutor defaults to a reasonable number (often os.cpu_count() * 5)
    logging.info(f"Using up to {num_workers or 'default'} worker threads for reading.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit read tasks for each file
        future_to_file = {executor.submit(read_single_jsonl, fp): fp for fp in jsonl_files}

        # Wrap as_completed with tqdm for progress bar
        logging.info("Processing files...")
        progress_bar = tqdm(concurrent.futures.as_completed(future_to_file), total=len(jsonl_files), desc="Reading files", unit="file")

        for future in progress_bar:
            file_path = future_to_file[future]
            # Update progress bar description (optional)
            # progress_bar.set_description(f"Processing {os.path.basename(file_path)}")
            try:
                data = future.result()
                if data is not None: # Check if reading was successful
                    merged_data.extend(data)
                    records_in_file = len(data)
                    total_records += records_in_file
                    # Logging can be reduced when using tqdm, or kept for detailed logs
                    # logging.info(f"Successfully processed {os.path.basename(file_path)} ({records_in_file} records).")
                else:
                    # Log skipped files if needed
                    logging.warning(f"Skipped file due to read errors: {file_path}")
            except Exception as exc:
                logging.error(f'{file_path} generated an exception during future processing: {exc}')

    if not merged_data:
        logging.warning("No data was successfully merged after processing all files. Exiting.")
        return

    logging.info(f"Total records merged: {total_records}")

    # Prepare output file path
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    output_filename = f"{timestamp}.jsonl"
    output_filepath = os.path.join(output_dir, output_filename)

    # Write the merged data to the output file
    try:
        logging.info(f"Writing merged data to: {output_filepath}...")
        # write_jsonl handles directory creation
        json_utils.write_jsonl(merged_data, output_filepath, ensure_ascii=False)
        logging.info(f"Successfully merged {total_records} records into {output_filepath}")
    except json_utils.JsonUtilsError as e:
        logging.error(f"Error writing merged file using json_utils: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing the output file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Merge all .jsonl files from a specified directory into a single timestamped .jsonl file using multiple threads.")
    parser.add_argument('--input_dir', type=str, default='datasets/symtex_batch_runs/auto_generated_files',
                        help='Directory containing the .jsonl files to merge.')
    parser.add_argument('--num_workers', type=int, default=4, # Default to None, ThreadPoolExecutor will choose
                        help='Number of worker threads for reading files. Defaults to ThreadPoolExecutor default.')

    args = parser.parse_args()

    # Define the fixed output directory relative to the script's location or CWD
    # Assuming the script is run from the project root (where datasets/ exists)
    output_directory = "datasets/symtex_merged_raw_dataset"

    merge_jsonl_files(args.input_dir, output_directory, args.num_workers)

if __name__ == "__main__":
    main()
