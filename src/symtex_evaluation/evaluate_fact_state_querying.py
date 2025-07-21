import argparse
import logging
import os
import json
from typing import List, Tuple, Dict, Any
import sys

from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd # For better confusion matrix display

# Import utility for reading jsonl safely
# This now relies strictly on json_utils being available
try:
    from src.utils.json_utils import read_jsonl, JsonUtilsError, _JSON_LIB # Import specific components
    from src.symtex_evaluation.common_metrics import compute_classification_metrics # Import new common function
except ImportError as e:
    # If json_utils cannot be imported, log a critical error and exit
    # It's better to fail early if a core dependency is missing
    print(f"Critical Error: Failed to import required module 'src.utils.json_utils' or 'src.symtex_evaluation.common_metrics'. Please ensure it exists and is accessible. Details: {e}", file=sys.stderr)
    # Optionally log this as well if logging is configured before this point
    # logger.critical(f"Failed to import required module 'src.utils.json_utils': {e}", exc_info=True)
    sys.exit(1) # Exit the script

# Setup basic logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_results(predictions_file_path: str, ground_truth_dataset_path: str) -> Tuple[List[str], List[str]]:
    """
    Loads predictions from the predictions file and labels from the ground truth dataset file.

    It matches predictions to labels using the 'id' field.

    :param predictions_file_path: Path to the JSONL file containing predictions (must have 'id' and 'prediction').
    :type predictions_file_path: str
    :param ground_truth_dataset_path: Path to the JSONL file containing ground truth labels (must have 'id' and 'label').
    :type ground_truth_dataset_path: str
    :return: A tuple containing two lists: true labels and predictions.
    :rtype: Tuple[List[str], List[str]]
    :raises FileNotFoundError: If either file does not exist.
    :raises JsonUtilsError: If there is an error reading the files using json_utils.
    :raises json.JSONDecodeError: If there is an error decoding JSON.
    :raises Exception: For any other unexpected loading errors.
    """
    labels = []
    predictions = []
    ground_truth_map: Dict[str, str] = {}

    try:
        # 1. Load ground truth labels into a map
        logger.info(f"Loading ground truth labels from: {ground_truth_dataset_path}")
        ground_truth_data = read_jsonl(ground_truth_dataset_path)
        skipped_gt_count = 0
        for i, item in enumerate(ground_truth_data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping item #{i+1} in ground truth file {ground_truth_dataset_path} as it's not a dictionary.")
                skipped_gt_count += 1
                continue
            item_id = item.get('id')
            label_value = item.get('label')
            if item_id is None or label_value is None:
                logger.warning(f"Skipping item #{i+1} in ground truth file {ground_truth_dataset_path} due to missing 'id' or 'label'. ID: {item_id}, Label: {label_value}")
                skipped_gt_count += 1
                continue
            ground_truth_map[str(item_id)] = str(label_value)
        if skipped_gt_count > 0:
            logger.warning(f"Skipped {skipped_gt_count} items from ground truth file {ground_truth_dataset_path} due to missing data or incorrect format.")
        logger.info(f"Successfully loaded {len(ground_truth_map)} ground truth ID-label pairs from {ground_truth_dataset_path}.")

        # 2. Load predictions and match with ground truth labels
        logger.info(f"Loading predictions from: {predictions_file_path}")
        predictions_data = read_jsonl(predictions_file_path)
        logger.info(f"Successfully loaded {len(predictions_data)} prediction items from {predictions_file_path} using json_utils.")

        skipped_pred_count = 0
        matched_count = 0
        for i, pred_item in enumerate(predictions_data):
            if not isinstance(pred_item, dict):
                logger.warning(f"Skipping prediction item #{i+1} from {predictions_file_path} as it's not a dictionary.")
                skipped_pred_count += 1
                continue

            pred_id = pred_item.get('id')
            prediction = pred_item.get('prediction')

            if pred_id is None or prediction is None:
                logger.warning(f"Skipping prediction item #{i+1} (ID: {pred_id if pred_id else 'N/A'}) from {predictions_file_path} due to missing 'id' or 'prediction'. Prediction: {prediction}")
                skipped_pred_count += 1
                continue
            
            pred_id_str = str(pred_id)
            label = ground_truth_map.get(pred_id_str)

            if label is None:
                logger.warning(f"No ground truth label found for ID '{pred_id_str}' from prediction file {predictions_file_path}. Skipping this prediction.")
                skipped_pred_count += 1
                continue

            # Basic normalization (consistent with previous logic)
            label = str(label).lower()
            prediction = str(prediction).lower()

            # Ensure consistent values (e.g., map 'true'/'false' if they appear)
            if label == "true": label = "positive"
            if label == "false": label = "negative"
            if prediction == "true": prediction = "positive"
            if prediction == "false": prediction = "negative"

            labels.append(label)
            predictions.append(prediction)
            matched_count +=1

        if skipped_pred_count > 0:
             logger.warning(f"Skipped {skipped_pred_count} prediction items due to missing data, incorrect format, or no matching ground truth label.")
        logger.info(f"Extracted {len(labels)} valid label/prediction pairs for evaluation after matching. Matched {matched_count} predictions with ground truth.")

        return labels, predictions

    except FileNotFoundError as e: # This will catch if either file is not found
        logger.error(f"Error: A required input file was not found. Details: {e}")
        raise
    except JsonUtilsError as e:
        logger.error(f"An error occurred reading a JSONL file using json_utils: {e}", exc_info=True)
        raise
    except (_JSON_LIB.JSONDecodeError, json.JSONDecodeError) as e:
        logger.error(f"Error decoding JSON in one of the input files: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading results: {e}", exc_info=True)
        raise


def calculate_metrics(labels: List[str], predictions: List[str]) -> Dict[str, Any] | None:
    """
    Calculates evaluation metrics (Macro-F1, Micro-F1, classification report, confusion matrix).

    :param labels: List of true labels.
    :type labels: List[str]
    :param predictions: List of predicted labels.
    :type predictions: List[str]
    :return: Dictionary containing the calculated metrics, or None if calculation fails.
    :rtype: Dict[str, Any] | None
    """
    if not labels or not predictions:
        logger.error("Cannot calculate metrics: Label or prediction list is empty.")
        return None
    if len(labels) != len(predictions):
         logger.error(f"Label count ({len(labels)}) and prediction count ({len(predictions)}) mismatch. Cannot calculate metrics.")
         return None

    # Determine the unique labels present in both true and predicted sets for reporting
    unique_labels_for_report = sorted(list(set(labels) | set(predictions)))
    logger.info(f"Unique labels found for evaluation: {unique_labels_for_report}")
    pos_label = "positive" # Assuming 'positive' is the positive class for binary F1 if needed

    try:
        # Use the common metrics function
        classification_metrics = compute_classification_metrics(
            labels=labels,
            predictions=predictions,
            labels_for_report=unique_labels_for_report,
            target_names_for_report=unique_labels_for_report, # Use unique_labels as target names
        )
        
        # The common function returns a dictionary with various metrics.
        # We can directly use this or extract specific parts if needed for the old structure.
        # For example, if the old structure expected just these specific keys:
        metrics_to_return = {
            "macro_f1": classification_metrics['macro_f1'],
            "micro_f1": classification_metrics['micro_f1'], # or classification_metrics['accuracy']
            "classification_report": classification_metrics['classification_report'],
            "confusion_matrix_df": classification_metrics['confusion_matrix_df'],
            "confusion_matrix_string": classification_metrics['confusion_matrix_string']
            # Add other metrics from classification_metrics if needed by the caller
        }
        return metrics_to_return

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        return None

def run_evaluation(results_file_path: str) -> Dict[str, Any] | None:
    """
    Runs the full evaluation pipeline for a given predictions file.
    It determines the ground truth dataset path based on the predictions file name.

    Loads data, calculates metrics, logs results, and returns the metrics dictionary.

    :param results_file_path: Path to the JSONL file containing prediction results.
    :type results_file_path: str
    :return: Dictionary containing evaluation metrics, or None if evaluation fails.
    :rtype: Dict[str, Any] | None
    """
    logger.info(f"Starting evaluation for predictions file: {results_file_path}")

    if not os.path.exists(results_file_path):
        logger.error(f"Predictions input file not found: {results_file_path}")
        return None

    # Determine ground truth dataset path based on predictions file name
    predictions_filename = os.path.basename(results_file_path)
    ground_truth_dataset_path = ""

    if predictions_filename.startswith("fact_state_querying_symbolic_"):
        ground_truth_dataset_path = os.path.join("datasets", "SymTex", "fact_state_querying_symbolic.jsonl")
    elif predictions_filename.startswith("fact_state_querying_textual_"):
        ground_truth_dataset_path = os.path.join("datasets", "SymTex", "fact_state_querying_textual.jsonl")
    else:
        logger.error(
            f"Cannot determine ground truth dataset for predictions file: {results_file_path}. "
            f"Filename must start with 'fact_state_querying_symbolic_' or 'fact_state_querying_textual_'."
        )
        return None
    
    logger.info(f"Determined ground truth dataset path: {ground_truth_dataset_path}")

    # Check if the determined ground truth dataset path actually exists
    if not os.path.exists(ground_truth_dataset_path):
        logger.error(f"Determined ground truth dataset file not found at: {ground_truth_dataset_path}")
        return None

    try:
        # Call the updated load_results with both file paths
        labels, predictions = load_results(results_file_path, ground_truth_dataset_path)

        if not labels or not predictions: # This might be true if no items could be matched or files were empty
             logger.warning("No valid data loaded or matched for evaluation. Evaluation cannot proceed.")
             return None

        metrics = calculate_metrics(labels, predictions)

        if metrics:
            logger.info("--- Evaluation Results ---")
            logger.info(f"Predictions file evaluated: {results_file_path}")
            logger.info(f"Ground truth data from: {ground_truth_dataset_path}") # Log the GT path
            logger.info(f"Micro-F1 Score: {metrics['micro_f1']:.4f}")
            logger.info(f"Macro-F1 Score: {metrics['macro_f1']:.4f}")
            logger.info("\nClassification Report:\n" + metrics['classification_report'])
            logger.info("\nConfusion Matrix:\n" + metrics['confusion_matrix_string'])
            logger.info("--------------------------")
            return metrics
        else:
            logger.error("Failed to calculate metrics.")
            return None

    except Exception as e:
        logger.critical(f"A critical error occurred during the evaluation process for {results_file_path} using GT {ground_truth_dataset_path}: {e}", exc_info=True)
        return None
    finally:
        logger.info(f"Evaluation process for {results_file_path} (using GT {ground_truth_dataset_path}) finished.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fact state querying results from a JSONL file using a separate ground truth dataset.")
    parser.add_argument("input_file", help="Path to the JSONL predictions file (e.g., fact_state_querying_symbolic_results.jsonl). Ground truth dataset path will be inferred based on this filename.")
    # Optional: Add argument for output file if needed later to save metrics
    # parser.add_argument("-o", "--output", help="Path to save evaluation metrics (e.g., JSON).")

    args = parser.parse_args()

    metrics_results = run_evaluation(args.input_file)

    if metrics_results is None:
        logger.error("Evaluation failed or produced no metrics.")
    else:
        logger.info("Evaluation completed successfully (results logged above).")

    logging.shutdown()

if __name__ == "__main__":
    main() 