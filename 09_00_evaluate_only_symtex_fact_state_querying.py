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
    # Import the evaluation function
    from src.symtex_evaluation.evaluate_fact_state_querying import run_evaluation
except ImportError as e:
    logger.critical(f"Failed to import run_evaluation function. Ensure 'src/symtex_evaluation/evaluate_fact_state_querying.py' exists and is accessible. Error: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during module imports: {e}", exc_info=True)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation for Fact State Querying Results.")
    # parser.add_argument("--results_file",
    #                     required=True, # Make this mandatory as it's the core input
    #                     type=str,
    #                     help="Path to the JSONL file containing the results to be evaluated.")
    parser.add_argument("--api_name",
                        default='local_qwen2_5_7b',
                        type=str,
                        help="Name of the model/API configuration used to generate the results (e.g., 'local_qwen2_5_7b').")
    parser.add_argument("--output_file",
                        default='experiments/fact_state_querying/w_few_shot/fact_state_querying_textual.jsonl',
                        # default='experiments/fact_state_querying/fact_state_querying_textual.jsonl',
                        type=str,
                        help="Base path for the output JSONL file (e.g., 'experiments/fact_state_querying/FSQ'). The script will append '_{api_name}.jsonl'.")
    parser.add_argument("--metrics_output_file",
                        type=str,
                        default=None, # Optional: Path to save the evaluation metrics as JSON
                        help="Optional: Path to save the evaluation metrics dictionary as a JSON file.")

    args = parser.parse_args()

    # --- Construct Final Results Path ---
    output_dir = os.path.dirname(args.output_file)
    base_name = os.path.basename(args.output_file)
    name_part, ext_part = os.path.splitext(base_name)
    # Ensure the extension is .jsonl, handling cases where base_name might already have one
    final_results_filename = f"{name_part}_{args.api_name}.jsonl"
    final_results_path = os.path.join(output_dir, final_results_filename)


    logger.info("--- Starting Fact State Querying Evaluation Only ---")
    logger.info(f"API Name: {args.api_name}")
    logger.info(f"Output Base Path: {args.output_file}")
    logger.info(f"Constructed Results File to Evaluate: {final_results_path}")
    if args.metrics_output_file:
        logger.info(f"Metrics Output File: {args.metrics_output_file}")

    # --- Ensure results file exists before proceeding ---
    if not os.path.exists(final_results_path):
        logger.critical(f"Constructed results file does not exist: {final_results_path}. Aborting evaluation.")
        sys.exit(1)


    # --- Run Evaluation ---
    evaluation_start_time = time.time()
    logger.info(f"Starting evaluation of results file: {final_results_path}")
    try:
        # Call the imported evaluation function
        evaluation_metrics = run_evaluation(results_file_path=final_results_path)

        if evaluation_metrics:
            logger.info("Evaluation completed successfully (details logged by evaluation script).")
            # Save metrics if an output path is provided
            if args.metrics_output_file:
                try:
                    # Ensure the output directory exists
                    metrics_output_dir = os.path.dirname(args.metrics_output_file)
                    if metrics_output_dir:
                         os.makedirs(metrics_output_dir, exist_ok=True)

                    with open(args.metrics_output_file, 'w') as f:
                        json.dump(evaluation_metrics, f, indent=4)
                    logger.info(f"Evaluation metrics saved to {args.metrics_output_file}")
                except Exception as e:
                    logger.error(f"Failed to save metrics to {args.metrics_output_file}: {e}", exc_info=True)
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