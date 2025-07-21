# 0900_evaluate_only_symtex_answer_set_generation.py
import argparse
import os
import sys
import logging
import time
import json

# Ensure the src directory is in the Python path
# Assume the script is run from the root directory containing 'src'
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# Add src to sys.path if needed
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
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler() # Keep console output
    ]
)
logger = logging.getLogger(__name__) # Get logger for this script


# --- Import necessary evaluation module --- 
try:
    # Import the evaluation function for Answer Set Generation
    from src.symtex_evaluation.evaluate_answer_set_generation import run_evaluation_answer_set_generation
except ImportError as e:
    logger.critical(f"Failed to import run_evaluation_answer_set_generation function. Ensure 'src/symtex_evaluation/evaluate_answer_set_generation.py' exists and is accessible. Error: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during module imports: {e}", exc_info=True)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Only for Answer Set Generation Results.")
    parser.add_argument("--api_name",
                        default='mmmfz_gpt_4o_mini', # Default model name used for generation
                        type=str,
                        help="Name of the model/API configuration used to generate the results (e.g., 'local_qwen2_5_7b').")
    parser.add_argument("--output_file",
                        # Default input base for generation task (used to construct the results file name)
                        # default='experiments/answer_set_generation/w_few_shot/answerset_generation_symbolic.jsonl',
                        default='experiments/openkg/full/openkg_ASC.jsonl',
                        type=str,
                        help="Base path of the input file used for generation (e.g., 'datasets/.../input.jsonl'). The script will derive the results file name from this and the api_name.")
    parser.add_argument("--data_file",
                        # Default input base for generation task (used to construct the results file name)
                        # default='experiments/answer_set_generation/w_few_shot/answerset_generation_symbolic.jsonl',
                        default='datasets/openkg_subset/openkg_ASC.jsonl',
                        type=str,
                        help="Base path of the input file used for generation (e.g., 'datasets/.../input.jsonl'). The script will derive the results file name from this and the api_name.")
    parser.add_argument("--metrics_output_file",
                        type=str,
                        default=None, # Optional: Path to save the evaluation metrics as JSON
                        help="Optional: Path to save the evaluation metrics dictionary as a JSON file.")

    args = parser.parse_args()

    # --- Construct Final Results Path to Evaluate ---
    try:
        output_dir = os.path.dirname(args.output_file)
        base_name = os.path.basename(args.output_file)
        name_part, _ = os.path.splitext(base_name)
        # Construct the expected results filename (matching the pattern in 09_03... script)
        final_results_filename = f"{name_part}_{args.api_name}_generated.jsonl"
        final_results_path = os.path.join(output_dir, final_results_filename)
    except Exception as e:
        logger.critical(f"Error constructing the results file path from --output_file '{args.output_file}': {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- Starting Answer Set Generation Evaluation Only ---")
    logger.info(f"API Name used for generation: {args.api_name}")
    logger.info(f"Base Input Path used for deriving results filename: {args.output_file}")
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
    evaluation_metrics = None # Initialize
    try:
        # Call the imported evaluation function
        evaluation_metrics = run_evaluation_answer_set_generation(results_file_path=final_results_path, original_dataset_file_path=args.data_file)

        if evaluation_metrics:
            logger.info("Evaluation completed successfully (details logged by evaluation script).")
            logger.info(f"Evaluation summary: {json.dumps(evaluation_metrics, indent=2)}")

            # Save metrics if an output path is provided
            if args.metrics_output_file:
                try:
                    # Ensure the output directory exists
                    metrics_output_dir = os.path.dirname(args.metrics_output_file)
                    if metrics_output_dir:
                         os.makedirs(metrics_output_dir, exist_ok=True)

                    with open(args.metrics_output_file, 'w', encoding='utf-8') as f:
                        json.dump(evaluation_metrics, f, indent=4, ensure_ascii=False)
                    logger.info(f"Evaluation metrics saved to {args.metrics_output_file}")
                except IOError as e:
                    logger.error(f"Failed to save metrics to {args.metrics_output_file}: {e}")
                except TypeError as e:
                    logger.error(f"Failed to serialize evaluation metrics to JSON: {e}")
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
        # Ensure the script signals completion
        status = "successfully" if evaluation_metrics else "with errors"
        logger.info(f"--- Evaluation Script Finished {status} ---")
        logging.shutdown() # Ensure all logs are flushed


if __name__ == "__main__":
    main() 