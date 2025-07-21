# src/symtex_evaluation/evaluate_answer_set_decision.py

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import argparse # Import argparse for main block

# Import metrics calculation utilities
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import pandas as pd # For better confusion matrix display

# Ensure utils are importable if run directly (adjust path as needed)
import sys
project_root_eval = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root_eval not in sys.path:
    sys.path.insert(0, project_root_eval)

try:
    from utils.json_utils import read_jsonl, JsonUtilsError, _JSON_LIB # Import necessary components
    from src.symtex_evaluation.common_metrics import compute_classification_metrics # Import new common function
except ImportError:
    # Fallback for direct execution if path setup fails, less robust
    print("Warning: Could not import from utils.json_utils or common_metrics via adjusted path. Ensure the script is run in an environment where 'src' is importable or add it to PYTHONPATH.")
    # Define a basic fallback or raise error if json_utils is critical
    def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Basic fallback read_jsonl if import fails."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Log or handle individual line errors if necessary
                        print(f"Error decoding JSON line in {file_path}: {e} - Line: {line.strip()}")
                        # Depending on requirements, you might skip the line or raise an error
                        # raise JsonUtilsError(f"JSON decode error in {file_path}") from e
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred reading {file_path}: {e}")
            raise
    # Define JsonUtilsError and _JSON_LIB minimally if needed for except blocks
    class JsonUtilsError(Exception): pass
    _JSON_LIB = json # Use standard json as fallback lib

# Setup logging for the evaluation script
log_directory_eval = os.path.join(project_root_eval, "logs") # Place logs in root logs directory
os.makedirs(log_directory_eval, exist_ok=True)
eval_log_file = os.path.join(log_directory_eval, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(eval_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_labels_and_predictions(file_path: str) -> Optional[Tuple[List[str], List[str], List[Dict[str, Any]]]]:
    """
    Loads labels and predictions from a JSONL result file for Answer Set Decision.

    :param file_path: Path to the JSONL file.
    :type file_path: str
    :return: A tuple containing (labels, predictions, mismatched_items), or None on failure.
    :rtype: Optional[Tuple[List[str], List[str], List[Dict[str, Any]]]]
    """
    labels = []
    predictions = []
    mismatched_items = []
    skipped_count = 0
    processed_count = 0

    logger.info(f"Attempting to load evaluation data from: {file_path}")
    try:
        results_data = read_jsonl(file_path)
        logger.info(f"Successfully loaded {len(results_data)} raw items from {file_path}.")

        if not results_data:
            logger.warning("Results file is empty.")
            return [], [], [] # Return empty lists if file is empty

        for i, item in enumerate(results_data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping item #{i+1} as it's not a dictionary.")
                skipped_count += 1
                continue

            prediction = item.get('prediction')
            label = item.get('label')
            item_id = item.get('id', f'index_{i+1}') # Get ID or use index

            if prediction is None or label is None:
                logger.warning(f"Skipping item {item_id}: Missing 'prediction' ('{prediction}') or 'label' ('{label}') field.")
                skipped_count += 1
                continue

            # Normalize expected values "Correct" and "Error" (e.g., to lower case)
            # Adjust if the actual values differ
            norm_label = str(label).strip() # Basic normalization
            norm_prediction = str(prediction).strip()

            # Treat any label that is not 'Correct' as 'Error'
            if norm_label != 'Correct':
                norm_label = 'Error'

            # Commenting out the original warning for unexpected labels as we now force them to Error
            # # Add check for expected values, could warn if unexpected values appear
            # expected_values = {"Correct", "Error"}
            # if norm_label not in expected_values:
            #     logger.warning(f"Item {item_id}: Unexpected label value '{label}'. Treating as is.")
            # if norm_prediction not in expected_values:
            #      logger.warning(f"Item {item_id}: Unexpected prediction value '{prediction}'. Treating as is.")

            labels.append(norm_label)
            predictions.append(norm_prediction)
            processed_count += 1

            if norm_label != norm_prediction:
                mismatched_items.append({
                    'id': item_id,
                    'label': norm_label,
                    'prediction': norm_prediction
                })

        if skipped_count > 0:
             logger.warning(f"Skipped {skipped_count} items due to missing data or incorrect format.")
        logger.info(f"Successfully extracted {processed_count} valid label/prediction pairs.")

        return labels, predictions, mismatched_items

    except FileNotFoundError:
        logger.error(f"Evaluation failed: Results file not found at {file_path}")
        return None
    except (JsonUtilsError, _JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
        logger.error(f"Evaluation failed: Error reading or decoding JSON file {file_path}. Error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: An unexpected error occurred loading {file_path}. Error: {e}", exc_info=True)
        return None


def calculate_and_log_metrics(labels: List[str], predictions: List[str], mismatched_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Calculates and logs evaluation metrics for binary classification.

    :param labels: List of true labels.
    :type labels: List[str]
    :param predictions: List of predicted labels.
    :type predictions: List[str]
    :param mismatched_items: List of items where prediction did not match the label.
    :type mismatched_items: List[Dict[str, Any]]
    :return: Dictionary containing calculated metrics, or None on failure.
    :rtype: Optional[Dict[str, Any]]
    """
    if not labels or not predictions:
        logger.warning("Cannot calculate metrics: Label or prediction list is empty after filtering.")
        return {
            'accuracy': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0,
            'total_items': 0, 'correct_items': 0, 'mismatched_count': 0,
            'classification_report': "N/A - No data",
            'confusion_matrix_string': "N/A - No data"
        }

    if len(labels) != len(predictions):
         logger.error(f"Label count ({len(labels)}) and prediction count ({len(predictions)}) mismatch. Cannot calculate metrics accurately.")
         return None # Critical error, should not happen if load function is correct

    # Determine the unique labels present (should be 'Correct', 'Error')
    # Use labels found in the actual data for robustness
    unique_labels_for_report = sorted(list(set(labels) | set(predictions)))
    logger.info(f"Unique labels identified for metrics calculation: {unique_labels_for_report}")

    # Define positive label if needed for specific metrics, assuming 'Correct' is positive
    pos_label = 'Correct' # For binary_f1, if desired later via common function

    try:
        # Use the common metrics function
        # Pass unique_labels_for_report to ensure consistent reporting for all classes present.
        # target_names_for_report can also be unique_labels_for_report if no special names are needed.
        classification_metrics = compute_classification_metrics(
            labels=labels, 
            predictions=predictions, 
            labels_for_report=unique_labels_for_report, 
            target_names_for_report=unique_labels_for_report,
            pos_label_for_binary_f1=pos_label
        )

        # Extract values from the returned dictionary
        accuracy = classification_metrics['accuracy']
        macro_f1 = classification_metrics['macro_f1']
        micro_f1 = classification_metrics['micro_f1'] # This is often same as accuracy
        report = classification_metrics['classification_report']
        cm_string = classification_metrics['confusion_matrix_string']
        # cm_df = classification_metrics['confusion_matrix_df'] # Available if needed

        total_items = len(labels)
        correct_items = int(accuracy * total_items) # Calculate correct count from accuracy
        mismatched_count = len(mismatched_items)

        # --- Log Metrics ---
        logger.info("--- Answer Set Decision Evaluation Results ---")
        logger.info(f"Total items evaluated: {total_items}")
        logger.info(f"Correct predictions: {correct_items}")
        logger.info(f"Mismatched predictions: {mismatched_count}")
        logger.info(f"Accuracy: {accuracy*100:.2f}%")
        logger.info(f"Micro-F1 Score: {micro_f1:.4f}")
        logger.info(f"Macro-F1 Score: {macro_f1:.4f}")
        logger.info("\nClassification Report:\n" + report)
        logger.info("\nConfusion Matrix:\n" + cm_string)
        if mismatched_items:
            logger.warning(f"First few mismatched items: {mismatched_items[:5]}")
        logger.info("---------------------------------------------")

        metrics = {
            'accuracy': round(accuracy * 100, 2),
            'macro_f1': round(macro_f1, 4),
            'micro_f1': round(micro_f1, 4),
            'total_items': total_items,
            'correct_items': correct_items,
            'mismatched_count': mismatched_count,
            'classification_report': report,
            'confusion_matrix_string': cm_string,
            # Keep the dataframe if needed elsewhere, but maybe not in the final returned dict for JSON saving
            # 'confusion_matrix_df': cm_df,
            # 'mismatched_items_details': mismatched_items # Optionally include all mismatches
        }
        return metrics

    except Exception as e:
        logger.error(f"Error calculating or logging metrics: {e}", exc_info=True)
        return None


def run_evaluation_answer_set(results_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Evaluates the results of the Answer Set Decision task using multiple metrics.

    :param results_file_path: Path to the JSONL file containing the results.
    :type results_file_path: str
    :return: A dictionary containing evaluation metrics, or None if evaluation fails.
    :rtype: Optional[Dict[str, Any]]
    """
    logger.info(f"Starting evaluation pipeline for Answer Set Decision task using file: {results_file_path}")

    load_result = load_labels_and_predictions(results_file_path)

    if load_result is None:
        # Error already logged in load_labels_and_predictions
        return None

    labels, predictions, mismatched_items = load_result

    # Calculate and log metrics
    metrics = calculate_and_log_metrics(labels, predictions, mismatched_items)

    logger.info(f"Evaluation pipeline for {results_file_path} finished.")
    return metrics # Return the dictionary of metrics or None if calculation failed

# Example of how to run this script directly (optional)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Answer Set Decision Evaluation")
    parser.add_argument("results_file", type=str, help="Path to the results JSONL file.")
    args = parser.parse_args()

    logger.info(f"Running evaluation directly for file: {args.results_file}")
    evaluation_results = run_evaluation_answer_set(args.results_file)

    if evaluation_results:
        # Log the summary dictionary returned by the function
        logger.info(f"Direct evaluation results summary:\n{json.dumps(evaluation_results, indent=2)}")
        # Detailed metrics are already logged within calculate_and_log_metrics
    else:
        logger.error("Direct evaluation run failed.")
        sys.exit(1) 