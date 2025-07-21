# src/utils/json_utils.py
import json
from typing import Any, List, Union, Generator
import os
import sys
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import os # Added for debugger check (though sys is primary)

# --- Try importing orjson for faster parsing ---
try:
    import orjson  # Use orjson if available
    _JSON_LIB = orjson
    # print("Using orjson for JSON operations.", file=sys.stderr)
except ImportError:
    _JSON_LIB = json # Fallback to standard json
    # print("Warning: orjson not found, falling back to standard json library. Install orjson for faster parsing.", file=sys.stderr)
# --- End orjson import ---

class JsonUtilsError(Exception):
    """Custom exception for JSON utility errors."""
    pass

# --- Internal helper for parallel parsing ---
def _parse_json_line(line: Union[str, bytes]) -> Any:
    """
    Parses a single JSON line using the selected library (_JSON_LIB), handles errors.
    Internal helper for read_jsonl_parallel.
    """
    try:
        # orjson loads directly from bytes or string, json loads from string
        return _JSON_LIB.loads(line)
    except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError): # Catch errors from both libs
        # Log or handle the error appropriately, here we skip the line and return None
        # print(f"Warning: Skipping invalid JSON line: {line[:100]}...", file=sys.stderr)
        return None
    except Exception as e:
        # print(f"Warning: Unexpected error parsing line: {e}", file=sys.stderr)
        return None
# --- End internal helper ---

def read_json(file_path: str) -> Any:
    """
    Reads data from a JSON file using the standard json library.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The loaded JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        JsonUtilsError: For other file reading issues.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Standard json library is sufficient here, orjson might be overkill for single files
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: JSON file not found at {file_path}")
    except json.JSONDecodeError as e:
        # Provide more context in the error message
        raise json.JSONDecodeError(f"Error decoding JSON from {file_path}: {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise JsonUtilsError(f"An unexpected error occurred while reading {file_path}: {e}")

def write_json(data: Any, file_path: str, indent: int = 4, ensure_ascii: bool = False) -> None:
    """
    Writes data to a JSON file using the standard json library.
    Creates parent directories if they don't exist.

    Args:
        data: The Python object to serialize.
        file_path: Path to the output JSON file.
        indent: Indentation level for pretty printing. Defaults to 4.
        ensure_ascii: If False, allows writing non-ASCII characters directly. Defaults to False.

    Raises:
        JsonUtilsError: For file writing issues.
    """
    try:
        # Ensure parent directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name: # Only create if path includes a directory
            os.makedirs(dir_name, exist_ok=True)

        # Use standard json for writing with indentation
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    except TypeError as e:
        raise JsonUtilsError(f"Error serializing data to JSON for {file_path}: {e}")
    except Exception as e:
        raise JsonUtilsError(f"An unexpected error occurred while writing to {file_path}: {e}")

def read_jsonl(file_path: str) -> List[Any]:
    """
    Reads all lines from a JSON Lines file sequentially using the selected JSON library.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        A list of loaded JSON objects from each line.

    Raises:
        FileNotFoundError: If the file does not exist.
        _JSON_LIB.JSONDecodeError: If a line contains invalid JSON.
        JsonUtilsError: For other file reading issues.
    """
    data = []
    try:
        # Use 'rb' mode if orjson is used, 'r' otherwise
        read_mode = 'rb' if _JSON_LIB == orjson else 'r'
        encoding = None if _JSON_LIB == orjson else 'utf-8' # orjson handles bytes directly

        with open(file_path, read_mode, encoding=encoding) as f:
            for i, line in enumerate(f):
                line_stripped = line.strip()
                if line_stripped: # Skip empty lines
                    try:
                        data.append(_JSON_LIB.loads(line_stripped))
                    except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
                         # Provide more context in the error message
                         # Note: json.JSONDecodeError might be needed if fallback occurs mid-operation (unlikely here)
                         raise _JSON_LIB.JSONDecodeError(f"Error decoding JSON on line {i+1} in {file_path}: {e.msg}", e.doc, e.pos)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: JSONL file not found at {file_path}")
    except Exception as e:
        raise JsonUtilsError(f"An unexpected error occurred while reading {file_path}: {e}")

def read_jsonl_lazy(file_path: str) -> Generator[Any, None, None]:
    """
    Reads a JSON Lines file line by line lazily using a generator and the selected JSON library.
    Useful for very large files.

    Args:
        file_path: Path to the JSONL file.

    Yields:
        Loaded JSON objects from each line.

    Raises:
        FileNotFoundError: If the file does not exist.
        _JSON_LIB.JSONDecodeError: If a line contains invalid JSON (raised during iteration).
        JsonUtilsError: For other file reading issues (raised during iteration).
    """
    try:
        # Use 'rb' mode if orjson is used, 'r' otherwise
        read_mode = 'rb' if _JSON_LIB == orjson else 'r'
        encoding = None if _JSON_LIB == orjson else 'utf-8'

        with open(file_path, read_mode, encoding=encoding) as f:
            for i, line in enumerate(f):
                line_stripped = line.strip()
                if line_stripped:
                    try:
                        yield _JSON_LIB.loads(line_stripped)
                    except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
                        # Raise error during iteration when the problematic line is encountered
                        raise _JSON_LIB.JSONDecodeError(f"Error decoding JSON on line {i+1} in {file_path}: {e.msg}", e.doc, e.pos)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: JSONL file not found at {file_path}")
    except Exception as e:
        # Catch potential errors during file opening or initial reading
        raise JsonUtilsError(f"An unexpected error occurred while reading {file_path}: {e}")

def read_jsonl_parallel(file_path: Union[str, Path], num_processes: int = None, chunksize: int = 1000) -> List[Any]:
    """
    Reads a JSON Lines file in parallel using multiprocessing and the selected JSON library (_JSON_LIB).
    Optimized for large files. Uses tqdm for progress indication.
    Automatically falls back to sequential reading if a debugger is detected.

    Args:
        file_path (Union[str, Path]): Path to the JSONL file.
        num_processes (int, optional): Number of worker processes. Defaults to cpu_count() - 1.
        chunksize (int, optional): Number of lines sent to each worker process at a time. Defaults to 1000.

    Returns:
        List[Any]: A list containing all successfully parsed JSON objects from the file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        JsonUtilsError: For other processing errors.
    """
    # --- Debugger Check ---
    # sys.gettrace() is not None when a debugger is active (like PyCharm's)
    # Also check for PYDEVD_USE_FRAME_EVAL env var which PyCharm debugger might set
    is_debugging = sys.gettrace() is not None or os.getenv('PYDEVD_USE_FRAME_EVAL') == 'YES'

    if is_debugging:
        print("Debugger detected. Falling back to sequential JSONL reading.", file=sys.stderr)
        # Call the sequential version instead
        try:
            # Use tqdm for sequential reading as well for consistency
            data = []
            total_lines = 0
            try:
                # Count lines for tqdm
                with open(file_path, 'rb') as f:
                    buf_gen = iter(lambda: f.read(1024*1024), b'')
                    total_lines = sum(buf.count(b'\n') for buf in buf_gen)
            except Exception:
                total_lines = None # Ignore if counting fails

            read_mode = 'rb' if _JSON_LIB == orjson else 'r'
            encoding = None if _JSON_LIB == orjson else 'utf-8'
            with open(file_path, read_mode, encoding=encoding) as f:
                 for line in tqdm(f, total=total_lines, desc="Parsing JSONL (Sequential)"):
                     parsed = _parse_json_line(line)
                     if parsed is not None:
                         data.append(parsed)
            print(f"Successfully parsed {len(data)} entries sequentially.")
            return data
        except Exception as e:
             raise JsonUtilsError(f"An error occurred during sequential fallback reading of {file_path}: {e}")
    # --- End Debugger Check ---


    # --- Original Parallel Logic ---
    input_file = Path(file_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Error: Input JSONL file not found at {input_file}")

    # --- Determine number of lines for tqdm ---
    total_lines = 0
    print(f"Counting lines in {input_file} for progress bar...")
    try:
        # Efficiently count lines
        with open(input_file, 'rb') as f: # Read as bytes for speed
             buf_gen = iter(lambda: f.read(1024*1024), b'')
             total_lines = sum(buf.count(b'\n') for buf in buf_gen)
        print(f"Found {total_lines} lines.")
    except Exception as e:
        print(f"Warning: Could not count lines accurately ({e}), progress bar might be inaccurate.", file=sys.stderr)
        total_lines = None # Set to None if counting fails

    # --- Setup multiprocessing ---
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Reading and parsing data from {input_file} using {num_processes} processes...")

    pool = None
    all_data = []
    try:
        pool = multiprocessing.Pool(processes=num_processes)

        # Use 'rb' mode if orjson is used, 'r' otherwise
        read_mode = 'rb' if _JSON_LIB == orjson else 'r'
        encoding = None if _JSON_LIB == orjson else 'utf-8'

        with open(input_file, read_mode, encoding=encoding) as f:
            # Use imap_unordered for potentially faster processing when order doesn't matter
            results_iterator = pool.imap_unordered(_parse_json_line, f, chunksize=chunksize)

            # Process results using tqdm for progress
            all_data = [result for result in tqdm(results_iterator, total=total_lines, desc="Parsing JSONL") if result is not None]

        print(f"Successfully parsed {len(all_data)} entries.")

    except Exception as e:
        raise JsonUtilsError(f"An unexpected error occurred during parallel processing of {input_file}: {e}")
    finally:
        # Ensure the pool is always closed and joined
        if pool:
            pool.close()
            pool.join()
            print("Process pool closed.")

    return all_data


def write_jsonl(data: List[Any], file_path: str, ensure_ascii: bool = False, use_orjson: bool = True) -> None:
    """
    Writes a list of Python objects to a JSON Lines file (overwrites existing file).
    Creates parent directories if they don't exist. Uses orjson by default if available for speed.

    Args:
        data: A list of Python objects to serialize.
        file_path: Path to the output JSONL file.
        ensure_ascii: If False, allows writing non-ASCII characters directly. Defaults to False.
                      (Note: orjson handles UTF-8 efficiently by default).
        use_orjson: If True and orjson is available, use it for writing. Defaults to True.

    Raises:
        JsonUtilsError: For file writing issues or if input is not a list.
    """
    if not isinstance(data, list):
        raise JsonUtilsError("Error: Input data for write_jsonl must be a list.")

    writer_lib = _JSON_LIB if use_orjson and _JSON_LIB == orjson else json
    write_mode = 'wb' if writer_lib == orjson else 'w'
    encoding = None if writer_lib == orjson else 'utf-8'
    newline = b'\n' if writer_lib == orjson else '\n'

    try:
        # Ensure parent directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, write_mode, encoding=encoding) as f:
            for item in data:
                try:
                    if writer_lib == orjson:
                        # orjson dumps to bytes, ensure_ascii is not a direct option but handles UTF-8
                        # Use OPT_APPEND_NEWLINE? No, add manually.
                        json_bytes = writer_lib.dumps(item)
                        f.write(json_bytes + newline)
                    else:
                        # standard json dumps to string
                        json_string = writer_lib.dumps(item, ensure_ascii=ensure_ascii, separators=(',', ':'))
                        f.write(json_string + newline)
                except TypeError as e:
                    # Provide more context about the failing item
                    raise JsonUtilsError(f"Error serializing item to JSON for {file_path} using {writer_lib.__name__}: {e}. Item: {str(item)[:100]}...")
    except Exception as e:
        raise JsonUtilsError(f"An unexpected error occurred while writing to {file_path}: {e}")

def append_jsonl(data: Union[Any, List[Any]], file_path: str, ensure_ascii: bool = False, use_orjson: bool = True) -> None:
    """
    Appends one or more Python objects to a JSON Lines file.
    Creates the file and parent directories if they don't exist. Uses orjson by default if available.

    Args:
        data: A single Python object or a list of objects to append.
        file_path: Path to the JSONL file.
        ensure_ascii: If False, allows writing non-ASCII characters directly. Defaults to False.
        use_orjson: If True and orjson is available, use it for writing. Defaults to True.

    Raises:
        JsonUtilsError: For file writing issues.
    """
    if not isinstance(data, list):
        items_to_append = [data] # Treat single item as a list with one element
    else:
        items_to_append = data

    writer_lib = _JSON_LIB if use_orjson and _JSON_LIB == orjson else json
    # Append mode needs careful handling with bytes vs text
    write_mode = 'ab' if writer_lib == orjson else 'a'
    encoding = None if writer_lib == orjson else 'utf-8'
    newline = b'\n' if writer_lib == orjson else '\n'

    try:
        # Ensure parent directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, write_mode, encoding=encoding) as f: # Use append mode
            for item in items_to_append:
                 try:
                    if writer_lib == orjson:
                        json_bytes = writer_lib.dumps(item, option=orjson.OPT_NON_STR_KEYS)
                        f.write(json_bytes + newline)
                    else:
                        json_string = writer_lib.dumps(item, ensure_ascii=ensure_ascii, separators=(',', ':'))
                        f.write(json_string + newline)
                 except TypeError as e:
                    raise JsonUtilsError(f"Error serializing item to JSON for appending to {file_path} using {writer_lib.__name__}: {e}. Item: {str(item)[:100]}...")
    except Exception as e:
        raise JsonUtilsError(f"An unexpected error occurred while appending to {file_path}: {e}")

# Example Usage (Optional - can be removed or kept for testing)
if __name__ == '__main__':
    # Create a dummy directory for testing relative to this script's location
    script_dir = os.path.dirname(__file__)
    test_dir = os.path.join(script_dir, 'temp_json_test')
    os.makedirs(test_dir, exist_ok=True)
    print(f"Using test directory: {test_dir}")

    json_file = os.path.join(test_dir, 'test.json')
    jsonl_file = os.path.join(test_dir, 'test.jsonl')
    jsonl_parallel_file = os.path.join(test_dir, 'test_parallel.jsonl')

    # --- Test JSON ---
    print(f"\n--- Testing JSON ---")
    print(f"Writing JSON to {json_file}...")
    my_dict = {"name": "测试", "value": 123, "nested": {"a": True, "b": None}}
    try:
        write_json(my_dict, json_file, ensure_ascii=False)
        print(f"Reading JSON from {json_file}...")
        read_data_json = read_json(json_file)
        print(f"Read JSON data: {read_data_json}")
        assert read_data_json == my_dict
        print("JSON Read/Write Test: PASSED")
    except (JsonUtilsError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"JSON Read/Write Test: FAILED - {e}")

    # --- Test JSONL ---
    print(f"\n--- Testing JSONL ---")
    print(f"Writing JSONL to {jsonl_file}...")
    my_list = [{"id": 1, "text": "第一行"}, {"id": 2, "text": "second line"}, {"id": 3, "valid": None}]
    try:
        # Test both writers if orjson is available
        write_jsonl(my_list, jsonl_file, ensure_ascii=False, use_orjson=(_JSON_LIB == orjson))
        print(f"Reading JSONL from {jsonl_file}...")
        read_data_jsonl = read_jsonl(jsonl_file)
        print(f"Read JSONL data: {read_data_jsonl}")
        assert read_data_jsonl == my_list
        print("JSONL Write/Read Test: PASSED")

        # Test standard json writer explicitly
        jsonl_std_file = os.path.join(test_dir, 'test_std.jsonl')
        write_jsonl(my_list, jsonl_std_file, ensure_ascii=False, use_orjson=False)
        read_data_std = read_jsonl(jsonl_std_file) # Read with default lib
        assert read_data_std == my_list
        print("JSONL Standard Write/Default Read Test: PASSED")

    except (JsonUtilsError, FileNotFoundError, _JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
        print(f"JSONL Write/Read Test: FAILED - {e}")


    # --- Test JSONL Append ---
    print(f"\n--- Testing JSONL Append ---")
    print(f"Appending to JSONL file {jsonl_file}...")
    append_item = {"id": 4, "new": True}
    append_list = [{"id": 5, "extra": "数据"}, {"id": 6}]
    try:
        append_jsonl(append_item, jsonl_file, ensure_ascii=False, use_orjson=(_JSON_LIB == orjson))
        append_jsonl(append_list, jsonl_file, ensure_ascii=False, use_orjson=(_JSON_LIB == orjson))
        print(f"Reading appended JSONL from {jsonl_file}...")
        read_data_appended = read_jsonl(jsonl_file)
        print(f"Read appended JSONL data: {read_data_appended}")
        expected_appended = my_list + [append_item] + append_list
        assert read_data_appended == expected_appended
        print("JSONL Append Test: PASSED")
    except (JsonUtilsError, FileNotFoundError, _JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
        print(f"JSONL Append Test: FAILED - {e}")

    # --- Test Lazy Reading ---
    print(f"\n--- Testing Lazy JSONL Reading ---")
    print(f"Lazy reading JSONL from {jsonl_file}...")
    read_lazy_data = []
    try:
        for item in read_jsonl_lazy(jsonl_file):
            print(f"  Read item: {item}")
            read_lazy_data.append(item)
        assert read_lazy_data == expected_appended
        print("JSONL Lazy Read Test: PASSED")
    except (JsonUtilsError, FileNotFoundError, _JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
        print(f"JSONL Lazy Read Test: FAILED - {e}")

    # --- Test Parallel Reading ---
    print(f"\n--- Testing Parallel JSONL Reading ---")
    print(f"Writing data for parallel test to {jsonl_parallel_file}...")
    # Use the appended data for a slightly larger test set
    try:
        write_jsonl(expected_appended, jsonl_parallel_file, use_orjson=(_JSON_LIB == orjson))
        print(f"Parallel reading JSONL from {jsonl_parallel_file}...")
        # Use fewer processes for small test file to avoid overhead issues
        num_test_processes = min(2, max(1, multiprocessing.cpu_count() - 1))
        read_parallel_data = read_jsonl_parallel(jsonl_parallel_file, num_processes=num_test_processes)
        # Order doesn't matter for comparison, sort both lists by id
        read_parallel_data.sort(key=lambda x: x.get('id', float('inf')))
        expected_appended.sort(key=lambda x: x.get('id', float('inf')))
        print(f"Read {len(read_parallel_data)} items in parallel.")
        assert read_parallel_data == expected_appended
        print("JSONL Parallel Read Test: PASSED")
    except (JsonUtilsError, FileNotFoundError) as e:
         print(f"JSONL Parallel Read Test: FAILED - {e}")
    except Exception as e: # Catch any other unexpected errors
         print(f"JSONL Parallel Read Test: FAILED - Unexpected error: {e}")


    # --- Test Error Handling ---
    print(f"\n--- Testing Error Handling ---")
    non_existent_file = os.path.join(test_dir, 'non_existent.json')
    invalid_json_file = os.path.join(test_dir, 'invalid.json')
    invalid_jsonl_file = os.path.join(test_dir, 'invalid.jsonl')
    with open(invalid_json_file, 'w') as f:
        f.write('{"key": "value",') # Invalid JSON
    with open(invalid_jsonl_file, 'w') as f:
        f.write('{"id": 1}\n')
        f.write('this is not json\n')
        f.write('{"id": 3}\n')


    try:
        read_json(non_existent_file)
    except FileNotFoundError:
        print(f"Caught expected FileNotFoundError for {non_existent_file}: PASSED")
    except Exception as e:
        print(f"Caught unexpected error for non-existent file: {e}: FAILED")

    try:
        read_json(invalid_json_file)
    except json.JSONDecodeError: # Standard json lib used for read_json
        print(f"Caught expected JSONDecodeError for {invalid_json_file}: PASSED")
    except Exception as e:
        print(f"Caught unexpected error for invalid JSON file: {e}: FAILED")

    # Test error in read_jsonl
    try:
        read_jsonl(invalid_jsonl_file)
        print(f"Read invalid JSONL without error: FAILED")
    except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError):
         print(f"Caught expected JSONDecodeError for invalid line in {invalid_jsonl_file}: PASSED")
    except Exception as e:
        print(f"Caught unexpected error reading invalid JSONL: {e}: FAILED")

    # Test error in read_jsonl_lazy (error during iteration)
    try:
        invalid_items = []
        for item in read_jsonl_lazy(invalid_jsonl_file):
            invalid_items.append(item)
        print(f"Lazy read invalid JSONL without error: FAILED")
    except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError):
        print(f"Caught expected JSONDecodeError during lazy read of {invalid_jsonl_file}: PASSED")
    except Exception as e:
        print(f"Caught unexpected error during lazy read of invalid JSONL: {e}: FAILED")

    # Test error in read_jsonl_parallel (should skip bad lines)
    try:
        # Should read lines 1 and 3, skipping line 2
        parallel_invalid_data = read_jsonl_parallel(invalid_jsonl_file, num_processes=num_test_processes)
        expected_invalid_parallel = [{"id": 1}, {"id": 3}]
        # Sort for comparison
        parallel_invalid_data.sort(key=lambda x: x.get('id', float('inf')))
        expected_invalid_parallel.sort(key=lambda x: x.get('id', float('inf')))
        assert parallel_invalid_data == expected_invalid_parallel
        print(f"Parallel read skipped invalid line in {invalid_jsonl_file} correctly: PASSED")
    except Exception as e:
        print(f"Caught unexpected error during parallel read of invalid JSONL: {e}: FAILED")


    # Clean up test files/directory (optional but recommended)
    import shutil
    try:
        print(f"\nCleaning up {test_dir}...")
        shutil.rmtree(test_dir)
        print("Cleanup successful.")
    except Exception as e:
        print(f"Warning: Failed to clean up test directory {test_dir}: {e}")

    print("\nJSON Utils tests completed.")
