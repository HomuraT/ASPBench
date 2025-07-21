import os
import sys
import json
import logging
import time
import math
import argparse
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
from collections import defaultdict

# --- Project Setup ---
# Assuming this script is in the project root
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import API_NAMES and json_utils
try:
    from notebooks.vars import API_NAMES
except ImportError:
    print("Error: Could not import API_NAMES from notebooks.vars. Make sure PYTHONPATH is set correctly or the script is run from the project root.")
    API_NAMES = [] # Default to empty list to allow script to be parsed, but it won't run meaningfully.

try:
    from src.utils.json_utils import read_jsonl
    # write_json from json_utils typically dumps a whole list as one JSON array,
    # not suitable for JSONL. We will define our own write_jsonl.
except ImportError:
    print("Error: Could not import read_jsonl from src.utils.json_utils. Basic fallback will be used if possible, or script may fail.")
    # Fallback basic read_jsonl if not found (copied from analysis scripts)
    def read_jsonl(file_path: Union[str, Path]) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {file_path}: {line.strip()}")
                    continue
        return data

# --- Constants ---
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR_BASE = PROJECT_ROOT / "datasets" / "SymTex_hard"

# Supported task names (used for inferring from filenames)
TASK_ANSWER_SET_DECISION = "answer_set_decision"
TASK_ANSWER_SET_GENERATION = "answer_set_generation"
TASK_FACT_STATE_QUERYING = "fact_state_querying"

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- Helper Functions (copied/adapted from evaluation scripts) ---

def _normalize_atom_string_for_asg(atom_str: str) -> str:
    """
    Normalizes a string representation of a logical atom for ASG tasks.
    (Copied and slightly adapted from evaluate_answer_set_generation.py)

    :param atom_str: The atom string to normalize.
    :type atom_str: str
    :return: The normalized atom string.
    :rtype: str
    """
    if not isinstance(atom_str, str):
        # logger.warning(f"ASG Norm: Attempted to normalize non-string: {atom_str}, type: {type(atom_str)}")
        return str(atom_str) # Ensure string type
    s = str(atom_str).strip()
    if s.endswith('.'):
        s = s[:-1]
    s = s.replace('"', '')   # Remove all internal quotes
    s = s.replace(' ', '') # Remove all spaces for a compact form, typical in some ASP outputs
    # Consider more sophisticated normalization if needed, matching evaluate_answer_set_generation.py's intent
    # Original from evaluate_answer_set_generation.py was more detailed:
    # s = s.replace(" (", "(").replace("( ", "(")
    # s = s.replace(" )", ")").replace(") ", ")")
    # s = s.replace(" ,", ",").replace(", ", ",")
    # if s.startswith("- ") and len(s) > 2: s = "-" + s[2:]
    # if s.startswith("+ ") and len(s) > 2: s = "+" + s[2:]
    return s.lower() # Ensure lowercase comparison, common for ASP atoms

def calculate_f1_for_asg(predicted_set: set, golden_set: set) -> Tuple[float, float, float]:
    """Calculates Precision, Recall, and F1 score between two sets of strings for ASG.
       (Copied from evaluate_answer_set_generation.py)
    :param predicted_set: A set of predicted atom strings.
    :type predicted_set: set
    :param golden_set: A set of golden atom strings.
    :type golden_set: set
    :return: Tuple (precision, recall, f1).
    :rtype: Tuple[float, float, float]
    """
    if not isinstance(predicted_set, set): predicted_set = set(predicted_set)
    if not isinstance(golden_set, set): golden_set = set(golden_set)

    true_positives = len(predicted_set.intersection(golden_set))
    predicted_positives = len(predicted_set)
    actual_positives = len(golden_set)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def write_jsonl(data_list: List[Dict], file_path: Union[str, Path], ensure_ascii: bool = False) -> None:
    """
    Writes a list of dictionaries to a JSONL file, one JSON object per line.
    Creates parent directories if they don't exist.

    :param data_list: List of Python dictionaries to serialize.
    :type data_list: List[Dict]
    :param file_path: Path to the output JSONL file.
    :type file_path: Union[str, Path]
    :param ensure_ascii: If False, allows writing non-ASCII characters directly. Defaults to False.
    :type ensure_ascii: bool
    :return: None
    :rtype: None
    """
    try:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Using newline='' to prevent Python from translating \n to \r\n on Windows etc.
        # We want to explicitly write LF (\n).
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            for item_index, item_data in enumerate(data_list):
                try:
                    json_string = json.dumps(item_data, ensure_ascii=ensure_ascii)
                    f.write(json_string)
                    f.write('\n') # Always write a newline character after each JSON object
                except TypeError as te_dump: # More specific exception for dump errors
                    item_id = item_data.get('id', f'unknown_id_at_index_{item_index}')
                    logger.error(f"TypeError dumping item '{item_id}' to JSON: {te_dump}. Item data: {str(item_data)[:200]}...", exc_info=False) # Log part of data
                except Exception as e_dump:
                    item_id = item_data.get('id', f'unknown_id_at_index_{item_index}')
                    logger.error(f"Error dumping item '{item_id}' to JSON: {e_dump}", exc_info=True)
                    
        logger.info(f"Finished writing {len(data_list)} items to {output_path}")
    except IOError as e_io: # More specific for file I/O issues during open/write
        logger.error(f"IOError occurred while writing to {file_path}: {e_io}", exc_info=True)
        raise
    except Exception as e_general: # Catch-all for other unexpected issues in the function
        logger.error(f"An unexpected error occurred in write_jsonl for {file_path}: {e_general}", exc_info=True)
        raise

def _normalize_atom_string(atom_str: str) -> str:
    """
    Normalizes an atom string by removing spaces, quotes, and lowercasing.
    (Copied from analysis_answer_set_generation.py)

    :param atom_str: The atom string to normalize.
    :type atom_str: str
    :return: The normalized atom string.
    :rtype: str
    """
    if not isinstance(atom_str, str):
        logger.warning(f"Attempted to normalize a non-string value: {atom_str} (type: {type(atom_str)}). Returning as is.")
        return str(atom_str) # Ensure it's a string if not already
    return atom_str.replace(" ", "").replace("'", "").replace("\"", "").replace(".", "").lower()

def _is_exact_match_for_item(predicted_atoms: Optional[List[str]], golden_sets_atoms_list: Optional[List[List[str]]]) -> bool:
    """
    Checks if the predicted_atoms exactly match any of the golden_sets_atoms_list.
    Handles normalization. (Adapted from analysis_answer_set_generation.py)

    :param predicted_atoms: A list of strings for the predicted answer set.
    :type predicted_atoms: Optional[List[str]]
    :param golden_sets_atoms_list: A list of lists of strings for golden answer sets.
    :type golden_sets_atoms_list: Optional[List[List[str]]]
    :return: True if an exact match is found, False otherwise.
    :rtype: bool
    """
    s_predicted: set
    if predicted_atoms is None:
        s_predicted = set()
    elif not isinstance(predicted_atoms, list):
        logger.debug(f"In _is_exact_match_for_item, predicted_atoms is not a list (type: {type(predicted_atoms)}). Treating as non-match.")
        return False
    else:
        str_predicted_atoms = [str(atom) for atom in predicted_atoms]
        s_predicted = set(_normalize_atom_string(atom) for atom in str_predicted_atoms)

    if not golden_sets_atoms_list: # No golden sets provided
        return not s_predicted # Match if predicted is also empty

    if not isinstance(golden_sets_atoms_list, list):
        logger.debug(f"golden_sets_atoms_list is not a list: {golden_sets_atoms_list}. Treating as non-match.")
        return False

    for golden_set_atoms in golden_sets_atoms_list:
        if not isinstance(golden_set_atoms, list):
            logger.debug(f"A golden set within golden_sets_atoms_list is not a list: {golden_set_atoms}. Skipping this golden set.")
            continue
        str_golden_set_atoms = [str(atom) for atom in golden_set_atoms]
        s_golden = set(_normalize_atom_string(atom) for atom in str_golden_set_atoms)
        if s_predicted == s_golden:
            return True
    return False

def determine_task_and_type(original_dataset_filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Determines the task name and result type from the original dataset filename.

    :param original_dataset_filename: The filename of the original dataset (e.g., "fact_state_querying_textual.jsonl").
    :type original_dataset_filename: str
    :return: A tuple (task_name, result_type) or (None, None) if not determinable.
    :rtype: Tuple[Optional[str], Optional[str]]
    """
    name_lower = original_dataset_filename.lower()
    task_name = None
    result_type = None

    # Determine task based on unique prefixes/substrings
    if "fact_state_querying" in name_lower:
        task_name = TASK_FACT_STATE_QUERYING
    elif "answerset_selection" in name_lower: # Catches "answerset_selection_textual.jsonl" etc.
        task_name = TASK_ANSWER_SET_DECISION
    elif "answerset_generation" in name_lower: # Catches "answerset_generation_symbolic.jsonl" etc.
        task_name = TASK_ANSWER_SET_GENERATION
    # Add more specific task checks if other naming conventions exist

    # Determine result type
    if "textual" in name_lower:
        result_type = "textual"
    elif "symbolic" in name_lower:
        result_type = "symbolic"

    if task_name and result_type:
        logger.info(f"Determined task: {task_name}, type: {result_type} from filename: {original_dataset_filename}")
        return task_name, result_type
    else:
        if not task_name:
            logger.warning(f"Could not determine task from filename: {original_dataset_filename}")
        if not result_type:
            logger.warning(f"Could not determine result_type from filename: {original_dataset_filename}")
        # More specific overall warning message
        logger.warning(f"Overall determination failed for filename: {original_dataset_filename}. Task resolved to '{task_name}', Type resolved to '{result_type}'.")
        return None, None


def get_model_prediction_file_path(task_name: str, result_type: str, api_name: str, category: str = "w_few_shot") -> Path:
    """
    Constructs the path to a model's prediction file based on task, type, and API name.
    Adjusts for known filename variations.

    :param task_name: The name of the task (e.g., 'fact_state_querying').
    :type task_name: str
    :param result_type: The result type ('textual' or 'symbolic').
    :type result_type: str
    :param api_name: The name of the API/model.
    :type api_name: str
    :param category: The category (default: 'w_few_shot').
    :type category: str
    :return: The Path object to the model's prediction file.
    :rtype: Path
    """
    # Base filename pattern: {task_name}_{result_type}_{api_name}.jsonl
    # For answer_set_decision: experiments/answer_set_decision/w_few_shot/answerset_selection_textual_API_NAME.jsonl
    # For answer_set_generation: experiments/answer_set_generation/w_few_shot/answerset_generation_symbolic_API_NAME_generated.jsonl

    filename_base = f"{task_name}_{result_type}"
    suffix = f"{api_name}.jsonl"

    if task_name == TASK_ANSWER_SET_DECISION:
        # Example from analysis: experiments/answer_set_decision/w_few_shot/answerset_selection_textual_API_NAME.jsonl
        # Original dataset: datasets/SymTex/answerset_selection_textual.jsonl
        filename_base = f"answerset_selection_{result_type}" # Matches the dataset_base_name used in analysis
    elif task_name == TASK_ANSWER_SET_GENERATION:
        # Example: experiments/answer_set_generation/w_few_shot/answerset_generation_symbolic_API_NAME_generated.jsonl
        filename_base = f"answerset_generation_{result_type}" # Matches dataset_base_name
        suffix = f"{api_name}_generated.jsonl" # Note the "_generated" part

    return EXPERIMENTS_DIR / task_name / category / f"{filename_base}_{suffix}"


def is_item_prediction_correct(
    original_item_data: Dict,
    model_prediction_item: Dict,
    task_name: str
) -> Union[bool, float]:
    """
    Determines if a model's prediction for an item is correct, based on the task.
    For FSQ and ASD, returns a boolean.
    For ASG, returns a float (max_f1_inferred_score).

    :param original_item_data: The dictionary for the item from the original dataset.
    :type original_item_data: Dict
    :param model_prediction_item: The dictionary for the item from the model's prediction file.
    :type model_prediction_item: Dict
    :param task_name: The name of the task.
    :type task_name: str
    :return: True if the prediction is correct, False otherwise.
    :rtype: Union[bool, float]
    """
    if task_name == TASK_FACT_STATE_QUERYING:
        true_label_raw = original_item_data.get('label')
        model_pred_raw = model_prediction_item.get('prediction')

        if true_label_raw is None or model_pred_raw is None:
            logger.debug(f"FSQ: Missing label or prediction for item ID {original_item_data.get('id', 'N/A')}. Treating as incorrect.")
            return False

        true_label = str(true_label_raw).lower().strip()
        model_pred = str(model_pred_raw).lower().strip()

        # Normalize to "positive"/"negative" as in analysis_fact_state_querying.py
        if true_label == "true": true_label = "positive"
        if true_label == "false": true_label = "negative"
        if model_pred == "true": model_pred = "positive"
        if model_pred == "false": model_pred = "negative"
        
        return true_label == model_pred

    elif task_name == TASK_ANSWER_SET_DECISION:
        true_label_raw = original_item_data.get('answer_set_decision')['type'] # e.g., "Correct" or "Error: something"
        model_pred_raw = model_prediction_item.get('prediction')

        if true_label_raw is None or model_pred_raw is None:
            logger.debug(f"ASD: Missing label or prediction for item ID {original_item_data.get('id', 'N/A')}. Treating as incorrect.")
            return False

        # Aligning with src/symtex_evaluation/evaluate_answer_set_decision.py
        # 1. Process true label
        processed_true_label = str(true_label_raw).strip()
        if processed_true_label != 'Correct': # Case-sensitive
            processed_true_label = 'Error'
        
        # 2. Process model prediction
        processed_model_pred = str(model_pred_raw).strip() # Expected to be "Correct" or "Error"

        is_correct = (processed_true_label == processed_model_pred)
        
        # Detailed debug log for ASD to understand previous 0% accuracy issues
        # This will log for every ASD item, consider reducing verbosity once diagnosed
        logger.debug(
            f"ASD Check ID {original_item_data.get('id', 'N/A')}: "
            f"Original Label='{str(true_label_raw)[:50]}', Model Pred='{str(model_pred_raw)[:50]}' | "
            f"Processed True='{processed_true_label}', Processed Pred='{processed_model_pred}' | "
            f"Match: {is_correct}"
        )
            
        return is_correct

    elif task_name == TASK_ANSWER_SET_GENERATION:
        # Get predicted answer set from model prediction item
        predicted_set_raw = model_prediction_item.get("aligned_answer_set")
        if predicted_set_raw is None or not isinstance(predicted_set_raw, list):
            logger.debug(f"ASG F1: Missing or invalid 'predicted_answer_set' in model data for ID {original_item_data.get('id', 'N/A')}. Defaulting F1 to 0.0.")
            return 0.0
        predicted_set_normalized = set(_normalize_atom_string_for_asg(atom) for atom in predicted_set_raw)

        # Get golden answer sets (prioritize original_item_data)
        golden_sets_raw = original_item_data.get('answer_sets') # Common key in original data
        if golden_sets_raw is None:
            golden_sets_raw = original_item_data.get('golden_answer_sets') # Alternative key
        if golden_sets_raw is None: # Fallback to model_prediction_item if not in original
            golden_sets_raw = model_prediction_item.get('golden_answer_sets')
            if golden_sets_raw is None:
                 golden_sets_raw = model_prediction_item.get('golden_answet_sets') # Typo check

        if golden_sets_raw is None or not isinstance(golden_sets_raw, list) or not all(isinstance(s, list) for s in golden_sets_raw):
            logger.debug(f"ASG F1: Missing or invalid golden sets for ID {original_item_data.get('id', 'N/A')}. Defaulting F1 to 0.0.")
            return 0.0
        
        golden_sets_normalized = [set(_normalize_atom_string_for_asg(atom) for atom in gs) for gs in golden_sets_raw]

        # Get facts from original_item_data
        facts_raw = original_item_data.get("facts", [])
        if not isinstance(facts_raw, list):
            logger.warning(f"ASG F1: 'facts' in original data for ID {original_item_data.get('id', 'N/A')} is not a list. Treating as empty.")
            facts_raw = []
        fact_set_normalized = set(_normalize_atom_string_for_asg(fact) for fact in facts_raw)

        # Calculate max F1 for inferred conclusions
        item_f1_inferred_scores = []
        predicted_set_inferred = predicted_set_normalized - fact_set_normalized

        if not golden_sets_normalized: # Should have been caught by earlier check, but as safeguard
            return 0.0
            
        for golden_s_norm in golden_sets_normalized:
            golden_set_inferred = golden_s_norm - fact_set_normalized
            
            current_precision, current_recall, current_f1 = 0.0, 0.0, 0.0
            if not predicted_set_inferred and not golden_set_inferred: # Both empty inferred sets
                current_precision, current_recall, current_f1 = 1.0, 1.0, 1.0
            elif fact_set_normalized == predicted_set_normalized and fact_set_normalized == golden_s_norm: # Only facts, no inferred part
                current_precision, current_recall, current_f1 = 1.0, 1.0, 1.0
            elif not golden_set_inferred and len(predicted_set_inferred) > 0 : # Golden inferred is empty, predicted inferred is not
                 current_precision, current_recall, current_f1 = 0.0, 0.0, 0.0
            # elif not predicted_set_inferred and len(golden_set_inferred) > 0: # Predicted inferred is empty, golden inferred is not (This case is implicitly handled by calculate_f1)
            #    current_precision, current_recall, current_f1 = 0.0, 0.0, 0.0
            else:
                _, _, current_f1 = calculate_f1_for_asg(predicted_set_inferred, golden_set_inferred)
            item_f1_inferred_scores.append(current_f1)
        
        max_f1_inferred = max(item_f1_inferred_scores) if item_f1_inferred_scores else 0.0
        
        # logger.debug(f"ASG F1 ID {original_item_data.get('id', 'N/A')}: PredRaw: {str(predicted_set_raw)[:50]}, Facts: {str(facts_raw)[:50]}, GoldenRaw: {str(golden_sets_raw)[:50]} -> MaxF1_inf: {max_f1_inferred:.4f}")
        return max_f1_inferred

    else:
        logger.warning(f"Unknown task name '{task_name}' for correctness check. Treating as incorrect.")
        return False


def extract_hardest_samples(original_dataset_file_path_str: str, api_names_list: List[str]):
    """
    Main function to extract the 10% hardest samples from a dataset based on average model performance.

    :param original_dataset_file_path_str: Path string to the original dataset JSONL file.
    :type original_dataset_file_path_str: str
    :param api_names_list: List of API names to consider for performance evaluation.
    :type api_names_list: List[str]
    :return: None
    :rtype: None
    """
    original_dataset_file_path = Path(original_dataset_file_path_str)
    if not original_dataset_file_path.is_file():
        logger.error(f"Original dataset file not found: {original_dataset_file_path}")
        return

    if not api_names_list:
        logger.error("API_NAMES list is empty. Cannot evaluate model performance.")
        return

    original_filename = original_dataset_file_path.name
    task_name, result_type = determine_task_and_type(original_filename)

    if not task_name or not result_type:
        logger.error(f"Could not determine task/type for {original_filename}. Skipping.")
        return

    logger.info(f"Processing dataset: {original_filename} (Task: {task_name}, Type: {result_type})")

    try:
        original_dataset_items = read_jsonl(original_dataset_file_path)
    except Exception as e:
        logger.error(f"Failed to read original dataset {original_dataset_file_path}: {e}", exc_info=True)
        return

    if not original_dataset_items:
        logger.info(f"Original dataset {original_filename} is empty. No samples to process.")
        return

    original_dataset_map = {item['id']: item for item in original_dataset_items if 'id' in item}
    if not original_dataset_map:
        logger.info(f"No items with 'id' found in {original_filename}. Cannot process.")
        return
    
    original_sample_ids = list(original_dataset_map.keys())
    # sample_model_performance = defaultdict(lambda: {'correct_model_names': [], 'incorrect_model_names': []}) # Old structure

    # New structure to store raw scores (bool for FSQ/ASD, float for ASG)
    sample_scores_by_id_then_api = defaultdict(dict)


    logger.info(f"Evaluating {len(original_sample_ids)} samples against {len(api_names_list)} models: {api_names_list}")

    for api_name in api_names_list:
        model_pred_file = get_model_prediction_file_path(task_name, result_type, api_name)
        logger.debug(f"Processing predictions for model {api_name} from {model_pred_file}")

        model_predictions_available_for_this_api = True
        model_item_predictions_map = {}

        if not model_pred_file.is_file():
            logger.warning(f"Prediction file for model {api_name} not found at {model_pred_file}. This model will yield default scores (False/0.0) for all samples.")
            model_predictions_available_for_this_api = False
        
        else:
            try:
                model_prediction_data = read_jsonl(model_pred_file)
                model_item_predictions_map = {item['id']: item for item in model_prediction_data if 'id' in item}
                logger.info(f"Loaded {len(model_item_predictions_map)} predictions for model {api_name}.")
            except Exception as e:
                logger.error(f"Failed to read or process prediction file {model_pred_file} for model {api_name}: {e}", exc_info=True)
                model_predictions_available_for_this_api = False

        for sample_id in original_sample_ids:
            original_item_data = original_dataset_map.get(sample_id) # Should always exist

            default_score_on_error = 0.0 if task_name == TASK_ANSWER_SET_GENERATION else False

            if not model_predictions_available_for_this_api:
                sample_scores_by_id_then_api[sample_id][api_name] = default_score_on_error
                continue

            model_prediction_item = model_item_predictions_map.get(sample_id)

            if original_item_data and model_prediction_item:
                try:
                    current_score = is_item_prediction_correct(original_item_data, model_prediction_item, task_name)
                    sample_scores_by_id_then_api[sample_id][api_name] = current_score
                except Exception as e_check:
                    logger.error(f"Error checking correctness for sample {sample_id}, model {api_name}: {e_check}", exc_info=True)
                    sample_scores_by_id_then_api[sample_id][api_name] = default_score_on_error
            elif original_item_data and not model_prediction_item:
                 logger.debug(f"Sample ID {sample_id} not found in predictions for model {api_name}. Assigning default score.")
                 sample_scores_by_id_then_api[sample_id][api_name] = default_score_on_error
            elif not original_item_data: # Should not happen as we iterate keys of original_dataset_map
                 logger.error(f"Critical: Original data for sample ID {sample_id} inexplicably missing. Assigning default score for model {api_name}.")
                 sample_scores_by_id_then_api[sample_id][api_name] = default_score_on_error


    # Calculate final metric for each sample based on aggregated model scores
    samples_with_computed_metric = []
    num_models = len(api_names_list)
    
    logger.debug(f"Calculating final metrics for {len(original_sample_ids)} samples.")

    for sample_id in original_sample_ids:
        model_specific_raw_scores = sample_scores_by_id_then_api[sample_id]
        
        # Ensure scores for all models are present for consistent averaging, defaulting if necessary
        scores_for_aggregation = []
        for api_n in api_names_list:
            if api_n in model_specific_raw_scores:
                scores_for_aggregation.append(model_specific_raw_scores[api_n])
            else:
                # This case should ideally be covered by the loops above, which iterate all api_names for each sample_id
                default_score = 0.0 if task_name == TASK_ANSWER_SET_GENERATION else False
                scores_for_aggregation.append(default_score)
                logger.warning(f"Score for model {api_n} on sample {sample_id} was unexpectedly missing from raw scores. Defaulting to {default_score}.")

        if task_name == TASK_ANSWER_SET_GENERATION:
            # scores_for_aggregation contains F1 floats
            if num_models == 0: 
                avg_metric_val = 0.0
            else:
                avg_metric_val = sum(scores_for_aggregation) / num_models
            
            # Store individual scores directly from model_specific_raw_scores, ensuring all APIs are represented
            individual_scores_dict = {api: model_specific_raw_scores.get(api, 0.0) for api in api_names_list}

            samples_with_computed_metric.append({
                'id': sample_id,
                'avg_score_to_sort_by': avg_metric_val, # Primary metric for sorting
                'metric_type': 'average_max_f1_inferred',
                'individual_model_scores': individual_scores_dict,
                'total_models_evaluated_against': num_models
            })
        else: # FSQ, ASD
            # scores_for_aggregation contains booleans
            correct_count = sum(1 for score_val in scores_for_aggregation if score_val is True)
            if num_models == 0:
                avg_metric_val = 0.0
            else:
                avg_metric_val = correct_count / num_models
            
            correct_model_names_list = [api for api in api_names_list if model_specific_raw_scores.get(api) is True]
            # incorrect_model_names_list includes those explicitly False AND those defaulted due to missing data/errors
            incorrect_model_names_list = [api for api in api_names_list if model_specific_raw_scores.get(api, False) is False]


            # Sanity check for FSQ/ASD regarding model name lists
            if len(correct_model_names_list) + len(incorrect_model_names_list) != num_models:
                 # This might happen if a score is neither True nor False (e.g. None, though current logic tries to avoid this)
                 # Recalculate incorrect_model_names_list to be more robust: all not in correct_model_names_list
                 all_apis_set = set(api_names_list)
                 correct_apis_set = set(correct_model_names_list)
                 incorrect_model_names_list = list(all_apis_set - correct_apis_set)

                 if len(correct_model_names_list) + len(incorrect_model_names_list) != num_models:
                     logger.warning(
                        f"FSQ/ASD Model count mismatch for sample {sample_id} AFTER recalculation: "
                        f"Correct ({len(correct_model_names_list)}), Incorrect ({len(incorrect_model_names_list)}). "
                        f"Total models: {num_models}. Scores: {model_specific_raw_scores}"
                     )


            samples_with_computed_metric.append({
                'id': sample_id,
                'avg_score_to_sort_by': avg_metric_val, # Primary metric for sorting
                'metric_type': 'average_accuracy',
                'correct_predictions_count': correct_count,
                'total_models_evaluated_against': num_models,
                'correct_model_names': correct_model_names_list,
                'incorrect_model_names': incorrect_model_names_list
            })

    # Filter out samples with 0% effective score for FSQ and ASD tasks (where 0% means all models failed)
    # For ASG, a score of 0.0 is a valid (worst) F1 score, so we don't filter it unless explicitly asked.
    # The existing filter was: samples_with_accuracy = [s for s in samples_with_accuracy if s['avg_accuracy'] > 0.0]
    # Now it should use 'avg_score_to_sort_by'
    if task_name in [TASK_FACT_STATE_QUERYING, TASK_ANSWER_SET_DECISION]:
        logger.info(f"Applying 0%-metric filter for task: {task_name}")
        original_sample_count = len(samples_with_computed_metric)
        # Filter if avg_score_to_sort_by is effectively zero (e.g. <= 0.0 for safety with floats, though it's accuracy here)
        samples_with_computed_metric = [s for s in samples_with_computed_metric if s['avg_score_to_sort_by'] > 1e-9] # Use small epsilon for float comparison
        
        filtered_sample_count = len(samples_with_computed_metric)
        if original_sample_count > filtered_sample_count:
            logger.info(
                f"Task '{task_name}': Filtered out {original_sample_count - filtered_sample_count} samples "
                f"where all models were incorrect (0% effective score). Now {filtered_sample_count} samples remain."
            )
        else:
            logger.info(f"Task '{task_name}': No samples were filtered out for 0% effective score (remains {filtered_sample_count} samples).")
    else:
        logger.info(f"Skipping 0%-metric filter for task: {task_name} (ASG or other).")


    # Sort samples by the primary metric (ascending, lower is harder), then by ID for stable sort
    samples_with_computed_metric.sort(key=lambda x: (x['avg_score_to_sort_by'], x['id']))

    # Determine number of samples to extract (10%)
    if not samples_with_computed_metric:
        logger.info("No samples processed or all filtered out. Exiting.")
        return
        
    num_to_extract = math.ceil(len(samples_with_computed_metric) * 0.10)
    if num_to_extract == 0 and len(samples_with_computed_metric) > 0: # Ensure at least one sample
        num_to_extract = 1
    
    logger.info(f"Total samples after any filtering: {len(samples_with_computed_metric)}. Extracting hardest {num_to_extract} samples (10%).")

    hardest_samples_info_list = samples_with_computed_metric[:num_to_extract]
    hardest_sample_ids = [s['id'] for s in hardest_samples_info_list]

    output_data = []
    hardest_samples_info_map = {info['id']: info for info in hardest_samples_info_list}

    for sample_id in hardest_sample_ids:
        if sample_id in original_dataset_map:
            original_sample_dict = original_dataset_map[sample_id].copy()
            metrics_for_sample = hardest_samples_info_map.get(sample_id)
            
            if metrics_for_sample:
                # Adapt extraction_metrics based on task type / metric_type
                if metrics_for_sample['metric_type'] == 'average_max_f1_inferred': # ASG
                    original_sample_dict["extraction_metrics"] = {
                        "average_max_f1_inferred_across_models": round(metrics_for_sample['avg_score_to_sort_by'], 4),
                        "individual_model_max_f1_inferred_scores": {k: round(v, 4) if isinstance(v, float) else v for k,v in metrics_for_sample['individual_model_scores'].items()},
                        "total_models_evaluated_against": metrics_for_sample['total_models_evaluated_against']
                    }
                else: # FSQ, ASD (average_accuracy)
                    original_sample_dict["extraction_metrics"] = {
                        "average_accuracy_across_models": round(metrics_for_sample['avg_score_to_sort_by'], 4),
                        "correct_predictions_count": metrics_for_sample['correct_predictions_count'],
                        "total_models_evaluated_against": metrics_for_sample['total_models_evaluated_against'],
                        "correct_model_names": metrics_for_sample['correct_model_names'],
                        "incorrect_model_names": metrics_for_sample['incorrect_model_names']
                    }
            else:
                original_sample_dict["extraction_metrics"] = None 
                logger.warning(f"Metrics for hardest sample ID '{sample_id}' not found. This is unexpected.")
            
            output_data.append(original_sample_dict)
        else:
            logger.warning(f"Original data for hardest sample ID '{sample_id}' not found. Skipping from output.")

    if not output_data:
        logger.info("No hardest samples identified or original data missing. Nothing to write.")
        return

    output_dir = OUTPUT_DIR_BASE
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / original_filename

    logger.info(f"Identified {len(output_data)} hardest samples. Saving to: {output_file_path}")
    if hardest_samples_info_list:
        logger.info("Top 5 hardest samples (ID, Sort_Metric_Value, More_Info):")
        for s_info in hardest_samples_info_list[:5]:
            sort_val = s_info['avg_score_to_sort_by']
            metric_type = s_info['metric_type']
            details = ""
            if metric_type == 'average_max_f1_inferred':
                details = f"AvgMaxF1Inferred: {sort_val:.4f}"
            else: # average_accuracy for FSQ/ASD
                correct_count = s_info['correct_predictions_count']
                total_models = s_info['total_models_evaluated_against']
                details = f"AvgAcc: {sort_val:.4f} ({correct_count}/{total_models})"
            logger.info(f"  - ID: {s_info['id']}, MetricValue: {sort_val:.4f} ({metric_type}), Details: [{details}]")
    
    try:
        write_jsonl(output_data, output_file_path)
    except Exception as e:
        logger.error(f"Failed to write hardest samples to {output_file_path}: {e}", exc_info=True)

    logger.info("Script finished.")


if __name__ == "__main__":
    # List of dataset files to process
    target_dataset_files = [
        "datasets/SymTex/answerset_generation_symbolic.jsonl",
        "datasets/SymTex/answerset_selection_textual.jsonl",
        "datasets/SymTex/fact_state_querying_textual.jsonl",
        "datasets/SymTex/answerset_generation_textual.jsonl",
        "datasets/SymTex/answerset_selection_symbolic.jsonl",
        "datasets/SymTex/fact_state_querying_symbolic.jsonl"
    ]

    if not API_NAMES:
        logger.error("API_NAMES is not defined or empty. Please check 'notebooks.vars.py'. Cannot proceed.")
        sys.exit(1)
        
    total_start_time = time.time()
    logger.info(f"Starting batch processing for {len(target_dataset_files)} dataset files...")

    for dataset_file_str in target_dataset_files:
        logger.info(f"--- Processing file: {dataset_file_str} ---")
        file_start_time = time.time()
        try:
            # Construct absolute path if needed, or ensure relative paths work from script location
            # Assuming PROJECT_ROOT is correctly defined for relative paths like "datasets/SymTex/..."
            full_path_to_dataset = PROJECT_ROOT / dataset_file_str
            extract_hardest_samples(str(full_path_to_dataset), API_NAMES)
        except Exception as e:
            logger.error(f"An error occurred while processing {dataset_file_str}: {e}", exc_info=True)
        file_end_time = time.time()
        logger.info(f"--- Finished processing {dataset_file_str} in {file_end_time - file_start_time:.2f} seconds ---")

    total_end_time = time.time()
    logger.info(f"Batch processing finished. Total execution time: {total_end_time - total_start_time:.2f} seconds.") 