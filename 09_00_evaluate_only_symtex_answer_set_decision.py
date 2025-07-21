import argparse
import os
import sys
import logging
import time
import json # Added for potentially saving metrics

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# Add src to sys.path if needed, assuming the script is run from the root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# --- Setup Logging ---
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
# Use a specific log file name for this evaluation script
log_file = os.path.join(log_directory, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # Keep console output
    ]
)
logger = logging.getLogger(__name__) # Get logger for this script


# --- Import necessary evaluation module ---
try:
    # Import the evaluation function for Answer Set Decision
    from src.symtex_evaluation.evaluate_answer_set_decision import run_evaluation_answer_set
except ImportError as e:
    logger.critical(f"Failed to import run_evaluation_answer_set function. Ensure 'src/symtex_evaluation/evaluate_answer_set_decision.py' exists and is accessible. Error: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during module imports: {e}", exc_info=True)
    sys.exit(1)


def main() -> None:
    """
    Main function to run the evaluation for Answer Set Decision results.

    Parses command-line arguments, constructs the path to the results file based on
    the base path and API name, checks if the file exists, runs the evaluation,
    and optionally saves the resulting metrics.
    """
    parser = argparse.ArgumentParser(description="Run Evaluation for Answer Set Decision Results.")
    parser.add_argument("--api_name",
                        default='local_qwen3_8b',
                        type=str,
                        help="Name of the model/API configuration used to generate the results (e.g., 'local_qwen2_5_7b').")
    parser.add_argument("--results_base_file",
                        default='experiments/answer_set_decision/w_few_shot/answerset_selection_textual.jsonl', # Default base input for this task
                        type=str,
                        help="Base path for the input JSONL file (e.g., 'experiments/answer_set_decision/answerset_selection_symbolic.jsonl'). The script will append '_{api_name}.jsonl' to this base path to find the actual results file.")
    parser.add_argument("--metrics_output_file",
                        type=str,
                        default=None, # Optional: Path to save the evaluation metrics as JSON
                        help="Optional: Path to save the evaluation metrics dictionary as a JSON file. If not provided, metrics will be logged but not saved to a separate file.")

    args = parser.parse_args()

    # --- Construct Final Results Path ---
    try:
        results_dir = os.path.dirname(args.results_base_file)
        base_name_without_ext = os.path.splitext(os.path.basename(args.results_base_file))[0]
        # Construct the expected results filename based on the base and api_name
        final_results_filename = f"{base_name_without_ext}_{args.api_name}.jsonl"
        final_results_path = os.path.join(results_dir, final_results_filename)

    except Exception as e:
        logger.critical(f"Error constructing the results file path from base '{args.results_base_file}' and api_name '{args.api_name}': {e}", exc_info=True)
        sys.exit(1)


    logger.info("--- Starting Answer Set Decision Evaluation Only ---")
    logger.info(f"API Name: {args.api_name}")
    logger.info(f"Results Base Path: {args.results_base_file}")
    logger.info(f"Constructed Results File to Evaluate: {final_results_path}")
    if args.metrics_output_file:
        # Construct the full metrics path if a base is given, append _{api_name}
        metrics_dir = os.path.dirname(args.metrics_output_file)
        metrics_base_name = os.path.splitext(os.path.basename(args.metrics_output_file))[0]
        final_metrics_filename = f"{metrics_base_name}_{args.api_name}.json" # Metrics file as .json
        final_metrics_path = os.path.join(metrics_dir, final_metrics_filename)
        logger.info(f"Metrics Output File: {final_metrics_path}")
    else:
        final_metrics_path = None
        logger.info("Metrics will not be saved to a separate file.")


    # --- Ensure results file exists before proceeding ---
    if not os.path.exists(final_results_path):
        logger.critical(f"Constructed results file does not exist: {final_results_path}. Aborting evaluation.")
        sys.exit(1)


    # --- Run Evaluation ---
    evaluation_start_time = time.time()
    logger.info(f"Starting evaluation of results file: {final_results_path}")
    try:
        # Call the imported evaluation function for Answer Set Decision
        evaluation_metrics = run_evaluation_answer_set(results_file_path=final_results_path)

        if evaluation_metrics:
            logger.info("Evaluation completed successfully (details logged by evaluation script).")
            # Save metrics if an output path is provided
            if final_metrics_path:
                try:
                    # Ensure the output directory exists
                    metrics_output_dir = os.path.dirname(final_metrics_path)
                    if metrics_output_dir:
                         os.makedirs(metrics_output_dir, exist_ok=True)

                    with open(final_metrics_path, 'w', encoding='utf-8') as f: # Use utf-8 encoding
                        json.dump(evaluation_metrics, f, indent=4, ensure_ascii=False) # ensure_ascii=False for non-latin chars
                    logger.info(f"Evaluation metrics saved to {final_metrics_path}")
                except IOError as e:
                    logger.error(f"Failed to save metrics to {final_metrics_path}: {e}", exc_info=True)
                except TypeError as e:
                    logger.error(f"Failed to serialize evaluation metrics to JSON: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"An unexpected error occurred while saving metrics: {e}", exc_info=True)
        else:
            logger.error("Evaluation script reported failure or returned no metrics (check logs above and evaluation script logs).")

        evaluation_duration = time.time() - evaluation_start_time
        logger.info(f"Evaluation finished. Time taken: {evaluation_duration:.2f} seconds.")

    except FileNotFoundError:
         # This might be redundant now due to the check above, but kept for safety
         logger.error(f"Results file {final_results_path} not found during evaluation step (should have been caught earlier).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the evaluation step: {e}", exc_info=True)

    finally:
        logger.info(f"--- Evaluation Script Finished ---")
        logging.shutdown() # Ensure all logs are flushed


if __name__ == "__main__":
    main() 