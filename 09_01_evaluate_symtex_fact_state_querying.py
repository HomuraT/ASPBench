import argparse
import os
import sys
import logging
import time

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# Add src to sys.path if needed, assuming the script is run from the root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# --- Setup Logging ---
# It's good practice to set up logging early.
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
# Use a more specific log file name for this main script
main_log_file = os.path.join(log_directory, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(main_log_file),
        logging.StreamHandler() # Keep console output
    ]
)
logger = logging.getLogger(__name__) # Get logger for this main script


# --- Import necessary modules after path setup ---
try:
    from src.symtex_evaluation.fact_state_querying import FactStateQueryingFramework
    # Import the refactored evaluation function
    from src.symtex_evaluation.evaluate_fact_state_querying import run_evaluation
except ImportError as e:
    logger.critical(f"Failed to import necessary modules. Ensure 'src' directory is accessible and contains the required scripts. Error: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during module imports: {e}", exc_info=True)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Fact State Querying and Evaluation Pipeline.")
    parser.add_argument("--input_file",
                        default="datasets/SymTex/fact_state_querying_textual.jsonl",
                        type=str,
                        help="Path to the input JSONL dataset file.")
    parser.add_argument("--api_name",
                        default="ppinfra_deepseek_v3",
                        type=str,
                        help="Name of the model/API configuration to use (e.g., 'mmm_gpt_4o_mini').")
    parser.add_argument("--json_parsing_model_name",
                        default="mmmfz_gpt_4o_mini",
                        type=str,
                        help="Name of the model/API used specifically for JSON parsing.")
    parser.add_argument("--output_dir", # Renamed for clarity, though variable name remains output_file internally for now
                        default="experiments/fact_state_querying/w_few_shot/",
                        type=str,
                        help="Path to the output directory for the results JSONL file.")
    # Optional arguments for more control, matching fact_state_querying.py environment variables
    parser.add_argument("--threads", type=int, default=1, help="Number of worker threads for processing.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save results every N items.")

    args = parser.parse_args()

    logger.info("--- Starting Fact State Querying Pipeline ---")
    logger.info(f"Input File: {args.input_file}")
    logger.info(f"Model API Name: {args.api_name}")
    logger.info(f"JSON Parsing Model Name: {args.json_parsing_model_name}")
    logger.info(f"Output Directory: {args.output_dir}") # Updated log message
    logger.info(f"Worker Threads: {args.threads}")
    logger.info(f"Save Interval: {args.save_interval}")

    # --- 1. Construct Final Output Path ---
    # Get the base name of the input file (without extension)
    input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
    # Construct the output filename using input basename and api_name
    final_output_filename = f"{input_basename}_{args.api_name}.jsonl"
    # Join the output directory and the filename
    final_output_path = os.path.join(args.output_dir, final_output_filename)
    logger.info(f"Final Results Path: {final_output_path}")

    # Create output directory if it doesn't exist
    output_directory = args.output_dir # Use the provided directory path directly
    if output_directory: # Check if the path is not empty
        try:
            os.makedirs(output_directory, exist_ok=True)
            logger.info(f"Ensured output directory exists: {output_directory}")
        except OSError as e:
            logger.error(f"Could not create output directory {output_directory}: {e}. Aborting.")
            return # Exit main function
    else:
        # Handle case where output_directory might be empty if default is changed or user provides empty string
        logger.warning("Output directory path is empty. Results will be saved in the current working directory.")
        # In this case, final_output_path will just be the filename, saving to CWD.

    # --- 2. Run Fact State Querying Processing ---
    processing_start_time = time.time()
    try:
        logger.info(f"Initializing FactStateQueryingFramework with model: {args.api_name} and JSON parser model: {args.json_parsing_model_name}...")
        framework = FactStateQueryingFramework(
            model_name=args.api_name,
            json_parsing_model_name=args.json_parsing_model_name
        )
        logger.info("Framework initialized successfully.")

        logger.info(f"Starting dataset processing for {args.input_file} -> {final_output_path}...")
        framework.process_dataset(
            input_file_path=args.input_file,
            output_file_path=final_output_path,
            num_threads=args.threads,
            save_interval=args.save_interval
        )
        processing_duration = time.time() - processing_start_time
        logger.info(f"Dataset processing finished. Time taken: {processing_duration:.2f} seconds.")

    except FileNotFoundError as e:
         logger.critical(f"Input file not found during processing step: {e}")
         return # Stop if input file not found by the framework
    except RuntimeError as e:
        logger.critical(f"Framework initialization or processing failed: {e}. Exiting.", exc_info=True)
        return # Stop if framework has issues
    except Exception as e:
         logger.critical(f"An unexpected error occurred during dataset processing: {e}", exc_info=True)
         return # Stop on other critical errors

    # --- 3. Run Evaluation ---
    evaluation_start_time = time.time()
    logger.info(f"Starting evaluation of results file: {final_output_path}")
    try:
        # Call the imported evaluation function
        evaluation_metrics = run_evaluation(results_file_path=final_output_path)

        if evaluation_metrics:
            logger.info("Evaluation completed successfully (details logged by evaluation script).")
            # Optionally, you could save the metrics dictionary here if needed
            # e.g., metrics_output_path = os.path.splitext(final_output_path)[0] + "_metrics.json"
            # with open(metrics_output_path, 'w') as f:
            #     json.dump(evaluation_metrics, f, indent=4)
            # logger.info(f"Evaluation metrics saved to {metrics_output_path}")
        else:
            logger.error("Evaluation script reported failure (check logs above and evaluation script logs).")

        evaluation_duration = time.time() - evaluation_start_time
        logger.info(f"Evaluation finished. Time taken: {evaluation_duration:.2f} seconds.")

    except FileNotFoundError:
        # This case should ideally be caught within run_evaluation, but handle defensively
         logger.error(f"Results file {final_output_path} not found for evaluation step.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during the evaluation step: {e}", exc_info=True)

    finally:
        total_duration = time.time() - processing_start_time # Total time includes processing and eval
        logger.info(f"--- Pipeline Finished --- Total time: {total_duration:.2f} seconds.")
        logging.shutdown() # Ensure all logs are flushed


if __name__ == "__main__":
    main()
