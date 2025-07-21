# -*- coding: utf-8 -*-
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
# Use a specific log file name for this script
main_log_file = os.path.join(log_directory, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(main_log_file, encoding='utf-8'), # Ensure UTF-8 for file handler
        logging.StreamHandler() # Keep console output
    ]
)
logger = logging.getLogger(__name__) # Get logger for this main script

# --- Import necessary modules after path setup ---
try:
    # Import the framework for Answer Set Generation
    from src.symtex_evaluation.answer_set_generation import AnswerSetGenerationFramework
    # Import the evaluation function for Answer Set Generation
    from src.symtex_evaluation.evaluate_answer_set_generation import run_evaluation_answer_set_generation
    logger.info("Framework and evaluation script imported.")
except ImportError as e:
    logger.critical(f"Failed to import necessary modules. Ensure 'src' directory is accessible and contains required files. Error: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"An unexpected error occurred during module imports: {e}", exc_info=True)
    sys.exit(1)


def main():
    """
    Main function to run the Answer Set Generation processing and (optional) evaluation pipeline.
    """
    start_pipeline_time = time.time() # Time the entire pipeline
    pipeline_successful = False # Flag to track success

    parser = argparse.ArgumentParser(description="Run Answer Set Generation and Evaluation Pipeline.") # Updated description
    parser.add_argument("--input_file",
                        default="datasets/SymTex/answerset_selection_classic_symbolic.jsonl", # Default input for generation task (CHANGE IF NEEDED)
                        type=str,
                        help="Path to the input JSONL dataset file for answer set generation.")
    parser.add_argument("--api_name",
                        default="mmm_gpt_4o_mini", # Default model, can be changed
                        type=str,
                        help="Name of the model/API configuration to use.")
    parser.add_argument("--json_parsing_model_name",
                        default="mmm_gpt_4o_mini", # 设置默认值
                        type=str,
                        help="Name of the model/API used specifically for JSON parsing.")
    parser.add_argument("--output_dir",
                        default="experiments/manual/answer_set_generation/w_few_shot/", # Default output dir for generation task
                        type=str,
                        help="Path to the output directory for the results JSONL file.")
    parser.add_argument("--threads", type=int, default=1, help="Number of worker threads for processing.") # Adjusted default threads
    parser.add_argument("--save_interval", type=int, default=1, help="Save results every N items.") # Adjusted default save interval
    parser.add_argument("--skip_evaluation", action='store_true', help="Skip the evaluation step.") # Option to skip evaluation

    args = parser.parse_args()

    logger.info("--- Starting Answer Set Generation Pipeline ---") # Updated task name
    logger.info(f"Input File: {args.input_file}")
    logger.info(f"Model API Name: {args.api_name}")
    logger.info(f"JSON Parsing Model Name: {args.json_parsing_model_name}") # 记录新参数
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Worker Threads: {args.threads}")
    logger.info(f"Save Interval: {args.save_interval}")
    logger.info(f"Skip Evaluation: {args.skip_evaluation}")

    # --- Construct Final Output Path ---
    try:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        final_output_filename = f"{input_basename}_{args.api_name}_generated.jsonl" # Added '_generated' suffix
        final_output_path = os.path.join(args.output_dir, final_output_filename)
        logger.info(f"Final Results Path: {final_output_path}")

        # Create output directory if it doesn't exist
        output_directory = args.output_dir
        if output_directory:
            try:
                os.makedirs(output_directory, exist_ok=True)
                logger.info(f"Ensured output directory exists: {output_directory}")
            except OSError as e:
                logger.error(f"Could not create output directory {output_directory}: {e}. Aborting.")
                return
        else:
            logger.warning("Output directory path is empty. Results will be saved in the current working directory.")

    except OSError as e:
        logger.error(f"Could not create output directory or access paths: {e}. Aborting.")
        return
    except Exception as e:
        logger.critical(f"Error during path setup: {e}. Aborting.", exc_info=True)
        return

    # --- Wrap Core Logic in Try/Except/Finally ---
    try:
        # --- 1. Run Answer Set Generation Processing --- (Renumbered step)
        processing_start_time = time.time()
        try:
            logger.info(f"Initializing AnswerSetGenerationFramework with model: {args.api_name} and JSON parser model: {args.json_parsing_model_name}...") # Updated framework name
            framework = AnswerSetGenerationFramework(
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
            # Assume processing is successful if it completes without exceptions
            processing_successful = True

        except FileNotFoundError as e:
             logger.critical(f"Input file not found during processing step: {e}")
             processing_successful = False
             raise # Reraise to be caught by outer block
        except RuntimeError as e:
            logger.critical(f"Framework initialization or processing failed: {e}. Exiting.", exc_info=True)
            processing_successful = False
            raise # Reraise
        except Exception as e:
             logger.critical(f"An unexpected error occurred during dataset processing: {e}", exc_info=True)
             processing_successful = False
             raise # Reraise

        # --- 2. Run Evaluation (Optional) --- (Renumbered step)
        evaluation_metrics = None
        if processing_successful and not args.skip_evaluation:
            evaluation_start_time = time.time()
            logger.info(f"Starting evaluation of results file: {final_output_path}")
            try:
                # Call the evaluation function
                evaluation_metrics = run_evaluation_answer_set_generation(results_file_path=final_output_path, original_dataset_file_path=args.input_file)

                if evaluation_metrics:
                    logger.info("Evaluation completed successfully (details logged by evaluation script).")
                    # Optional: Save metrics
                    metrics_output_path = os.path.splitext(final_output_path)[0] + "_metrics.json"
                    try:
                        with open(metrics_output_path, 'w', encoding='utf-8') as f:
                             json.dump(evaluation_metrics, f, indent=4, ensure_ascii=False)
                        logger.info(f"Evaluation metrics saved to {metrics_output_path}")
                    except IOError as e:
                        logger.error(f"Failed to save evaluation metrics to {metrics_output_path}: {e}")
                    except TypeError as e:
                         logger.error(f"Failed to serialize evaluation metrics to JSON: {e}")
                else:
                    logger.error("Evaluation did not return metrics (or was skipped/not implemented). Check logs.")

                evaluation_duration = time.time() - evaluation_start_time
                logger.info(f"Evaluation step finished. Time taken: {evaluation_duration:.2f} seconds.")
                # Consider pipeline successful if processing succeeded AND evaluation (if run) produced metrics
                if evaluation_metrics:
                    pipeline_successful = True
                elif args.skip_evaluation:
                     pipeline_successful = True # Successful if processing done and eval skipped

            except ImportError:
                logger.error(f"Could not import evaluation function 'run_evaluation_answer_set_generation'. Evaluation skipped.")
            except FileNotFoundError:
                 logger.error(f"Results file {final_output_path} not found for evaluation step.")
                 raise # Reraise
            except Exception as e:
                logger.critical(f"An unexpected error occurred during the evaluation step: {e}", exc_info=True)
                raise # Reraise
        elif not processing_successful:
            logger.error("Skipping evaluation because the processing step failed.")
        else:
            logger.info("Skipping evaluation step as requested by --skip_evaluation flag.")
            pipeline_successful = True # Successful if processing done and eval skipped

    except Exception as e:
        # Catch errors raised from the processing or evaluation blocks
        logger.critical(f"A critical error occurred in the main pipeline execution: {e}", exc_info=True)
        pipeline_successful = False

    finally:
        # This block executes regardless of success or failure
        total_duration = time.time() - start_pipeline_time
        status = "Successfully" if pipeline_successful else "with ERRORS"
        logger.info(f"--- Pipeline Finished {status} --- Total time: {total_duration:.2f} seconds.")
        logging.shutdown()


if __name__ == "__main__":
    main() 