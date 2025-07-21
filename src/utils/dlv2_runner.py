import subprocess
import tempfile
import os
import yaml
import re
from typing import List, Dict, Optional, Any

class Dlv2RunnerError(Exception):
    """Custom exception for Dlv2Runner errors."""
    pass

class Dlv2Runner:
    """
    A wrapper class to execute ASP programs using dlv2 and parse the results.

    Reads the dlv2 executable path from a YAML configuration file.
    Handles temporary file creation for ASP programs, executes dlv2,
    parses the output for answer sets or errors, and ensures cleanup.
    """
    def __init__(self, config_path: str = "configs/dlv2.yaml"):
        """
        Initializes the Dlv2Runner.

        Args:
            config_path: Path to the YAML configuration file containing 'dlv2_path'.

        Raises:
            Dlv2RunnerError: If the config file is not found, invalid,
                             or the dlv2 path is not executable (basic check).
        """
        self.dlv2_path: str = self._load_config(config_path)
        self._check_dlv2_executable()

    def _load_config(self, config_path: str) -> str:
        """Loads dlv2 path from the YAML configuration file."""
        if not os.path.exists(config_path):
            raise Dlv2RunnerError(f"Configuration file not found: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict) or 'dlv2_path' not in config:
                raise Dlv2RunnerError(f"Invalid configuration format in {config_path}. Expected a dictionary with 'dlv2_path'.")
            dlv2_path = config['dlv2_path']
            if not isinstance(dlv2_path, str):
                 raise Dlv2RunnerError(f"Invalid 'dlv2_path' in {config_path}. Expected a string.")
            if not dlv2_path.strip():
                 raise Dlv2RunnerError(f"'dlv2_path' in {config_path} cannot be empty.")
            return dlv2_path
        except yaml.YAMLError as e:
            raise Dlv2RunnerError(f"Error parsing YAML configuration file {config_path}: {e}")
        except Exception as e:
            raise Dlv2RunnerError(f"Error reading configuration file {config_path}: {e}")

    def _check_dlv2_executable(self) -> None:
        """Checks if the configured dlv2 path points to an existing file."""
        if not os.path.exists(self.dlv2_path):
             raise Dlv2RunnerError(f"dlv2 executable not found at path specified in config: {self.dlv2_path}")
        if not os.path.isfile(self.dlv2_path):
             raise Dlv2RunnerError(f"dlv2 path specified in config is not a file: {self.dlv2_path}")


    def run(self, asp_program: str, timeout: Optional[float] = None, num_answer_sets: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs the given ASP program string using dlv2.
        (Raises ValueError if num_answer_sets is invalid)
        """
        temp_file_path = None
        stdout = ""
        stderr = ""

        # Validate num_answer_sets before entering the main try block
        if num_answer_sets is not None:
            if not isinstance(num_answer_sets, int) or num_answer_sets < 0:
                 raise ValueError("num_answer_sets must be a non-negative integer.")

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".dlv", delete=False, encoding='utf-8') as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(asp_program)
                temp_file.flush()
                try:
                    os.fsync(temp_file.fileno())
                except OSError:
                    pass

            cmd = [self.dlv2_path, temp_file_path]
            # Append -n option if num_answer_sets is valid (already checked)
            if num_answer_sets is not None:
                cmd.append(f"-n={num_answer_sets}")

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )
            stdout = process.stdout
            stderr = process.stderr

        except subprocess.TimeoutExpired:
            # Timeout is a failure condition, return directly
            return {
                "success": False,
                "answer_sets": None,
                "error_message": f"dlv2 execution timed out after {timeout} seconds.",
                "raw_output": f"Timeout occurred after {timeout}s.\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}".strip()
            }
        except FileNotFoundError:
             return {
                "success": False, "answer_sets": None,
                "error_message": f"dlv2 executable not found or failed to execute at: {self.dlv2_path}",
                "raw_output": f"Error: dlv2 not found at {self.dlv2_path}"
            }
        except PermissionError:
             return {
                "success": False, "answer_sets": None,
                "error_message": f"Permission denied to execute dlv2 at: {self.dlv2_path}",
                 "raw_output": f"Error: Permission denied for {self.dlv2_path}"
            }
        # Let ValueError for num_answer_sets propagate
        except OSError as e:
             raise Dlv2RunnerError(f"OS error during dlv2 execution or file handling: {e}") from e
        except Exception as e:
             # Catch other unexpected errors during subprocess execution
             print(f"Unexpected error during Dlv2Runner.run: {type(e).__name__}: {e}")
             return {
                "success": False, "answer_sets": None,
                "error_message": f"An unexpected error occurred during execution: {type(e).__name__}",
                "raw_output": f"Unexpected Error: {e}\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}".strip()
            }
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    print(f"Warning: Failed to delete temporary file {temp_file_path}: {e}")

        # If no exception occurred (including Timeout), parse the output
        parsed_result = self._parse_output(stdout, stderr)
        raw_output = f"--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}"
        parsed_result["raw_output"] = raw_output.strip()
        return parsed_result


    def _parse_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """
        Parses the stdout and stderr from dlv2 execution.
        Prioritizes errors in stderr, then non-success messages in stdout, then parses answer sets.
        """
        # --- Step 1: Prioritize stderr ---
        # If stderr has any content, assume failure immediately.
        if stderr:
            stderr_lower = stderr.lower()
            # Try to find specific error types for better messages
            if "syntax error" in stderr_lower:
                error_line = next((line for line in stderr.splitlines() if "syntax error" in line.lower()), stderr.strip())
                return {"success": False, "answer_sets": None, "error_message": f"Syntax Error: {error_line}"}
            if "safety error" in stderr_lower:
                 error_line = next((line for line in stderr.splitlines() if "safety error" in line.lower()), stderr.strip())
                 match = re.search(r"Safety Error:\s*(.*)", stderr, re.IGNORECASE | re.DOTALL)
                 error_detail = match.group(1).strip() if match else error_line
                 return {"success": False, "answer_sets": None, "error_message": f"Safety Error: {error_detail}"}
            if "aborting due to parser errors" in stderr_lower:
                 return {"success": False, "answer_sets": None, "error_message": "Execution aborted due to parser errors."}
            # If stderr has content but doesn't match known patterns, return generic stderr error
            return {"success": False, "answer_sets": None, "error_message": f"dlv2 Error (stderr): {stderr.strip()}"}

        # --- Step 2: Check stdout for known non-success messages (only if stderr was empty) ---
        stdout_content_upper = stdout.upper()
        if "UNSATISFIABLE" in stdout_content_upper:
             return {"success": False, "answer_sets": None, "error_message": "UNSATISFIABLE"}
        if "INCOHERENT" in stdout_content_upper:
             return {"success": False, "answer_sets": None, "error_message": "INCOHERENT"}

        # --- Step 3: If no errors detected so far, attempt to parse answer sets ---
        answer_sets: List[List[str]] = []
        # Regex to find lines exactly matching {content} or {}
        # Removed re.DOTALL as '.' matching newline might be problematic. re.MULTILINE is sufficient.
        answer_set_matches = re.findall(r"^{(.*)}$", stdout, re.MULTILINE)

        for content in answer_set_matches:
            content = content.strip()
            if content: # Non-empty set like {a, b}
                # Use regex to handle predicates with arguments containing commas
                predicate_pattern = r'-?\w+\([^)]*\)|-?\w+'
                predicates = [p.strip() for p in re.findall(predicate_pattern, content) if p.strip()]
            else: # Empty set {}
                predicates = []
            answer_sets.append(predicates)

        # --- Step 4: Determine final success state ---
        # If we found any answer sets via regex, it's definitely a success.
        if answer_set_matches:
            return {"success": True, "answer_sets": answer_sets, "error_message": None}
        else:
            # No answer sets found. Check if stdout was otherwise empty (ignoring header).
            relevant_stdout = stdout.strip()
            relevant_stdout = re.sub(r"^DLV \d+\.\d+\.\d+\s*", "", relevant_stdout, count=1).strip()
            if not relevant_stdout:
                # Empty relevant output means success with zero answer sets (e.g., comments only)
                return {"success": True, "answer_sets": [], "error_message": None}
            else:
                # No sets found, stdout not empty, and no known errors -> Unknown/Failure
                return {"success": False, "answer_sets": None, "error_message": f"Unknown or unexpected dlv2 output format in stdout (no errors/sets): {relevant_stdout[:200]}..."}


# Example Usage Block (useful for direct testing of this file)
if __name__ == "__main__":
    print("Running Dlv2Runner example...")
    try:
        # Ensure the config file exists for the example
        config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
        config_file = os.path.join(config_dir, 'dlv2.yaml')
        if not os.path.exists(config_file):
             print(f"Warning: Config file {config_file} not found. Creating a dummy one for testing.")
             os.makedirs(config_dir, exist_ok=True)
             dummy_dlv2_path = '/path/to/your/dlv2' # <--- EDIT THIS IF NEEDED FOR LOCAL TEST
             with open(config_file, 'w', encoding='utf-8') as f:
                 yaml.dump({'dlv2_path': dummy_dlv2_path}, f)
             print(f"Created dummy config with path: {dummy_dlv2_path}")
             print("--- Please ensure this path is correct for dlv2! ---")

        runner = Dlv2Runner()

        print(f"\nUsing dlv2 path: {runner.dlv2_path}")
        print("\n--- Test Case 1: Simple Satisfiable Program ---")
        asp_code_1 = "p(a). q(X) :- p(X)."
        result_1 = runner.run(asp_code_1)
        print(f"Result: {result_1}")

        print("\n--- Test Case 2: Syntax Error ---")
        asp_code_2 = "a :- b." # Missing dot
        result_2 = runner.run(asp_code_2)
        print(f"Result: {result_2}")

        print("\n--- Test Case 3: Unsatisfiable Program ---")
        asp_code_3 = "a. :- a."
        result_3 = runner.run(asp_code_3)
        print(f"Result: {result_3}")

        print("\n--- Test Case 4: Multiple Answer Sets (-n=0) ---")
        asp_code_4 = "a v b."
        result_4 = runner.run(asp_code_4, num_answer_sets=0) # Get all
        print(f"Result: {result_4}")

        print("\n--- Test Case 5: Empty Answer Set ---")
        asp_code_5 = ":- not a. a."
        result_5 = runner.run(asp_code_5)
        print(f"Result: {result_5}")

        print("\n--- Test Case 6: Only Comments ---")
        asp_code_6 = "% comment"
        result_6 = runner.run(asp_code_6)
        print(f"Result: {result_6}")

        print("\n--- Test Case 7: Timeout ---")
        # Use a potentially longer running program for timeout test
        asp_code_7 = """
        num(1..20).
        p(X) :- num(X), p(X-1).
        p(0).
        :- not p(20).
        """
        try:
             result_7 = runner.run(asp_code_7, timeout=0.01) # Very short timeout
             print(f"Result: {result_7}")
        except ValueError as e:
             print(f"Caught ValueError during timeout test (unexpected): {e}")
        except Dlv2RunnerError as e:
             print(f"Caught Dlv2RunnerError (OS error): {e}")
        except Exception as e:
             print(f"Caught other Exception during timeout test: {e}")


        print("\n--- Test Case 8: Safety Error ---")
        asp_code_8 = "P10(V0,V1):-not -P14(V0,V1)."
        result_8 = runner.run(asp_code_8)
        print(f"Result: {result_8}")

        print("\nExample execution finished.")

    except Dlv2RunnerError as e:
        print(f"\nError initializing or running Dlv2Runner: {e}")
    except FileNotFoundError:
         print(f"\nError: The dlv2 executable specified in the config was not found.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the example run: {type(e).__name__}: {e}")
