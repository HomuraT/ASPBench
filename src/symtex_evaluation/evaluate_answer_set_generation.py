# src/symtex_evaluation/evaluate_answer_set_generation.py

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Set
import argparse
import sys
import pandas as pd # Added pandas import

# --- Project Root Setup ---
# Assuming the script might be run directly or imported
try:
    # Attempt to find the project root assuming structure like /path/to/SymTex/src/symtex_evaluation
    project_root_eval = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # __file__ is not defined, possibly running in an interactive environment
    # Fallback or default path, adjust as needed
    project_root_eval = os.path.abspath(os.path.join(os.getcwd(), "..", "..")) # Example fallback

if project_root_eval not in sys.path:
    sys.path.insert(0, project_root_eval)
# Add src to sys.path for utility imports
src_path_eval = os.path.join(project_root_eval, 'src')
if src_path_eval not in sys.path:
     sys.path.insert(0, src_path_eval)

# --- Utility Imports ---
try:
    # Import JSONL reading utility
    from utils.json_utils import read_jsonl, JsonUtilsError
    # Import common metrics utilities
    from src.symtex_evaluation.common_metrics import normalize_atom_string, calculate_set_prf1
except ImportError:
    print("Warning: Could not import from utils.json_utils or symtex_evaluation.common_metrics. Using basic fallbacks.")
    # Define basic fallback read_jsonl if import fails.
    def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Basic fallback read_jsonl if import fails."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON line in {file_path}: {e} - Line: {line.strip()}")
                        # Decide how to handle errors: skip line or raise
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred reading {file_path}: {e}")
            raise
    class JsonUtilsError(Exception): pass # Minimal definition for except block

# --- Logging Setup ---
log_directory_eval = os.path.join(project_root_eval, "logs")
os.makedirs(log_directory_eval, exist_ok=True)
eval_log_file = os.path.join(log_directory_eval, f"{os.path.splitext(os.path.basename(__file__))[0]}.log") # Use specific name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(eval_log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper function for string normalization ---
# def _normalize_atom_string(atom_str: str) -> str:
#     """
#     Normalizes a string representation of a logical atom.
#     Removes leading/trailing whitespace, trailing period, all internal quotes,
#     and normalizes spaces around parentheses, commas, and leading signs.
#     """
#     s = str(atom_str).strip()
#     s = s.removesuffix('.')  # Python 3.9+
#     s = s.replace('"', '')   # Remove all internal quotes
# 
#     # Normalize spaces around parentheses and commas
#     s = s.replace(" (", "(").replace("( ", "(")
#     s = s.replace(" )", ")").replace(") ", ")")
#     s = s.replace(" ,", ",").replace(", ", ",")
#     
#     # Normalize spaces around ':-' if it's a rule (less common for facts but good for general utility)
#     s = s.replace(" :-", ":-").replace(":- ", ":-")
# 
#     # Handle space after leading sign, e.g., "- predicate" -> "-predicate"
#     if s.startswith("- ") and len(s) > 2:
#         s = "-" + s[2:]
#     # '+' is often implicit but handle if explicitly used with a space
#     elif s.startswith("+ ") and len(s) > 2:
#         s = "+" + s[2:]
#     
#     return s

# --- Core Evaluation Logic ---

# def calculate_f1(predicted_set: Set[str], golden_set: Set[str]) -> Tuple[float, float, float]:
#     """Calculates Precision, Recall, and F1 score between two sets."""
#     if not isinstance(predicted_set, set): predicted_set = set(predicted_set)
#     if not isinstance(golden_set, set): golden_set = set(golden_set)
# 
#     true_positives = len(predicted_set.intersection(golden_set))
#     predicted_positives = len(predicted_set)
#     actual_positives = len(golden_set)
# 
#     precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
#     recall = true_positives / actual_positives if actual_positives > 0 else 0.0
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
# 
#     return precision, recall, f1

def load_and_process_generation_data(
    predictions_file_path: str, 
    original_dataset_file_path: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Loads and processes data from the JSONL results file for generation evaluation,
    optionally merging with an original dataset for facts, golden truths, and source_type.

    :param predictions_file_path: Path to the JSONL file containing model predictions.
    :type predictions_file_path: str
    :param original_dataset_file_path: Optional path to the original JSONL dataset file.
    :type original_dataset_file_path: Optional[str]
    :return: List of processed items with extracted sets and source_type, or None on failure.
    :rtype: Optional[List[Dict[str, Any]]]
    """
    processed_data = []
    skipped_count = 0
    skipped_missing_id_count = 0  # 新增：跟踪因ID不存在而跳过的项目数
    logger.info(f"Attempting to load prediction data from: {predictions_file_path}")
    if original_dataset_file_path:
        logger.info(f"Attempting to load original dataset from: {original_dataset_file_path}")

    original_data_map: Dict[str, Dict[str, Any]] = {}
    if original_dataset_file_path:
        try:
            original_raw_data = read_jsonl(original_dataset_file_path)
            for item in original_raw_data:
                item_id = item.get('id')
                if item_id:
                    original_data_map[str(item_id)] = item
                else:
                    logger.warning(f"Original dataset item in {original_dataset_file_path} missing 'id'. Skipping.")
            logger.info(f"Successfully loaded {len(original_data_map)} items from original dataset {original_dataset_file_path}.")
        except FileNotFoundError:
            logger.error(f"Original dataset file not found at {original_dataset_file_path}. Proceeding without it.")
            original_data_map = {} # Ensure it's an empty dict
        except (JsonUtilsError, json.JSONDecodeError) as e:
            logger.error(f"Error reading or decoding original dataset {original_dataset_file_path}. Error: {e}. Proceeding without it.", exc_info=True)
            original_data_map = {}
        except Exception as e:
            logger.error(f"Unexpected error loading original dataset {original_dataset_file_path}. Error: {e}. Proceeding without it.", exc_info=True)
            original_data_map = {}
            
    try:
        predictions_raw_data = read_jsonl(predictions_file_path)
        logger.info(f"Successfully loaded {len(predictions_raw_data)} raw prediction items from {predictions_file_path}.")

        if not predictions_raw_data:
            logger.warning("Predictions file is empty.")
            return []

        for i, pred_item in enumerate(predictions_raw_data):
            item_id_str = str(pred_item.get('id', f'pred_index_{i+1}'))
            original_item_data = original_data_map.get(item_id_str)

            # --- ID过滤：只处理在原始数据集中存在的样本 ---
            if original_dataset_file_path and original_item_data is None:
                logger.debug(f"Skipping item {item_id_str}: ID not found in original dataset {original_dataset_file_path}.")
                skipped_count += 1
                skipped_missing_id_count += 1
                continue

            # --- Extract source_type ---
            # source_type MUST come from the original dataset
            item_source_type: Optional[str] = None
            valid_source_types = {"P_style", "related_word", "random_word"}

            if original_item_data:
                raw_source_type = original_item_data.get('source_type')
                if isinstance(raw_source_type, str) and raw_source_type in valid_source_types:
                    item_source_type = raw_source_type
                elif raw_source_type is not None: # Found source_type but it's not one of the valid ones
                    logger.warning(f"Item {item_id_str}: Unexpected 'source_type' value '{raw_source_type}' in original data. Expected one of {valid_source_types}. Item will be processed without a specific source_type or an error might occur later if source_type is strictly required.")
                    # Depending on strictness, you might assign a default or skip. For now, let it be None.
                    # item_source_type = "unknown_source_type" # Or handle as error
                else: # source_type field is missing or not a string
                    logger.warning(f"Item {item_id_str}: Missing or invalid 'source_type' field in original_item_data. Original data provided: {original_item_data is not None}. Item will be processed without specific source_type.")
            else:
                logger.warning(f"Item {item_id_str}: No corresponding original_item_data found to extract 'source_type'. Cannot assign a source_type.")
            # --- End Extract source_type ---

            # Extract predicted answer set (aligned) - Must come from predictions
            predicted_set_list = pred_item.get('aligned_answer_set')
            if predicted_set_list is None:
                 logger.warning(f"Skipping item {item_id_str}: Missing 'aligned_answer_set' in prediction item.")
                 skipped_count += 1
                 continue
            if not isinstance(predicted_set_list, list):
                logger.warning(f"Skipping item {item_id_str}: 'aligned_answer_set' in prediction item is not a list (found type {type(predicted_set_list)}).")
                skipped_count += 1
                continue
            predicted_set = set(normalize_atom_string(str(x)) for x in predicted_set_list if x is not None)

            # --- Extract token usage information ---
            token_info = {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'non_prompt_tokens': 0
            }
            
            llm_input_and_response = pred_item.get('llm_input_and_response', [])
            if isinstance(llm_input_and_response, list) and len(llm_input_and_response) > 0:
                first_response = llm_input_and_response[0]
                if isinstance(first_response, dict):
                    token_info['total_tokens'] = first_response.get('total_tokens', 0)
                    token_info['prompt_tokens'] = first_response.get('prompt_tokens', 0)
                    token_info['completion_tokens'] = first_response.get('completion_tokens', 0)
                    # 计算非prompt_tokens数量 (total_tokens - prompt_tokens)
                    token_info['non_prompt_tokens'] = max(0, token_info['total_tokens'] - token_info['prompt_tokens'])
                    logger.debug(f"Item {item_id_str}: Extracted token info - total: {token_info['total_tokens']}, prompt: {token_info['prompt_tokens']}, completion: {token_info['completion_tokens']}, non_prompt: {token_info['non_prompt_tokens']}")
                else:
                    logger.warning(f"Item {item_id_str}: First element of llm_input_and_response is not a dict.")
            else:
                logger.warning(f"Item {item_id_str}: Missing or empty 'llm_input_and_response' field.")
            # --- End Extract token usage information ---

            # --- Determine facts and golden sets ---
            input_facts_list = []
            golden_sets_list = []

            if original_item_data:
                # Prioritize facts from original dataset
                input_facts_list = original_item_data.get('facts', [])
                if not input_facts_list and 'facts' not in original_item_data: # If 'facts' key exists but list is empty, that's fine
                     logger.debug(f"Item {item_id_str}: 'facts' field missing or empty in original data. Trying prediction item.")
                     input_facts_list = pred_item.get('facts', []) # Fallback to pred_item if not in original
                
                # Prioritize golden sets from original dataset (key 'answer_sets' or 'golden_answer_sets')
                golden_sets_list = original_item_data.get('answer_sets') 
                if golden_sets_list is None: # Try 'golden_answer_sets' if 'answer_sets' is not found
                    golden_sets_list = original_item_data.get('golden_answer_sets')
                
                if golden_sets_list is None: # If still None, try from prediction item
                    logger.debug(f"Item {item_id_str}: Golden sets not found in original data. Trying prediction item.")
                    golden_sets_list = pred_item.get('golden_answet_sets') # Typo key
                    if golden_sets_list is None:
                        golden_sets_list = pred_item.get('golden_answer_sets') # Correct key
            else:
                # logger.warning(f"Item {item_id_str}: No corresponding item found in original dataset. Using data from prediction item only for facts and golden sets.")
                input_facts_list = pred_item.get('facts', [])
                golden_sets_list = pred_item.get('golden_answet_sets') # Typo key
                if golden_sets_list is None:
                    golden_sets_list = pred_item.get('golden_answer_sets') # Correct key
            
            # Validate and process facts
            if not isinstance(input_facts_list, list):
                logger.warning(f"Item {item_id_str}: 'facts' field is not a list (found type {type(input_facts_list)}). Treating as empty facts.")
                input_facts_list = []
            fact_set = set(normalize_atom_string(str(x)) for x in input_facts_list if x is not None)

            # Validate and process golden sets
            if golden_sets_list is None:
                logger.warning(f"Skipping item {item_id_str}: Missing 'golden_answer_sets' (or similar) in both original and prediction data.")
                skipped_count += 1
                continue
            if not isinstance(golden_sets_list, list) or not all(isinstance(s, list) for s in golden_sets_list):
                logger.warning(f"Skipping item {item_id_str}: Golden sets field is not a list of lists.")
                skipped_count += 1
                continue
            golden_sets = [set(normalize_atom_string(str(x)) for x in gold_list if x is not None) for gold_list in golden_sets_list]
            
            processed_data.append({
                'id': item_id_str,
                'predicted_set': predicted_set,
                'golden_sets': golden_sets,
                'fact_set': fact_set,
                'source_type': item_source_type, # Add the determined source_type
                'token_info': token_info # Add token usage information
            })

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} items due to missing/invalid data during processing.")
            if skipped_missing_id_count > 0:
                logger.info(f"  - {skipped_missing_id_count} items were skipped because their IDs were not found in the original dataset.")
            logger.info(f"  - {skipped_count - skipped_missing_id_count} items were skipped due to other data validation issues.")
        
        if original_dataset_file_path and skipped_missing_id_count > 0:
            logger.info(f"ID Filtering Summary: Processed {len(processed_data)} items, skipped {skipped_missing_id_count} items (IDs not in dataset) out of {len(predictions_raw_data)} total predictions.")
        
        logger.info(f"Successfully processed {len(processed_data)} items for evaluation.")
        return processed_data

    except FileNotFoundError:
        logger.error(f"Evaluation failed: Predictions file not found at {predictions_file_path}")
        return None
    except (JsonUtilsError, json.JSONDecodeError) as e:
        logger.error(f"Evaluation failed: Error reading or decoding JSON predictions file {predictions_file_path}. Error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Evaluation failed: An unexpected error occurred loading {predictions_file_path}. Error: {e}", exc_info=True)
        return None

def calculate_and_log_generation_metrics(processed_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Calculates and logs evaluation metrics for Answer Set Generation, 
    including overall dataset metrics and metrics per source_type.

    :param processed_data: List of dicts, each containing 'id', 'predicted_set', 
                           'golden_sets', 'fact_set', 'source_type', and 'token_info'.
    :type processed_data: List[Dict[str, Any]]
    :return: Dictionary containing aggregated metrics (overall and per source_type), 
             and data for CSV output, or None on failure.
    :rtype: Optional[Dict[str, Any]]
    """
    if not processed_data:
        logger.warning("Cannot calculate metrics: No processed data available.")
        return {'total_items': 0, 'average_exact_match': 0.0, 'average_max_f1': 0.0, 'average_max_f1_inferred': 0.0, 'metrics_by_source_type': {}, 'csv_data': [], 'token_metrics': {}}

    # --- Overall Dataset Metrics Initialization ---
    overall_exact_match_scores = []
    overall_max_f1_scores = []
    overall_max_f1_inferred_scores = []
    
    # --- Token Usage Metrics Initialization ---
    overall_total_tokens = []
    overall_prompt_tokens = []
    overall_completion_tokens = []
    overall_non_prompt_tokens = []
    
    # --- Per-source_type Metrics Initialization ---
    # Dynamically discover source_types present in the data, plus predefined ones we expect.
    # However, user stated it will always be one of the three.
    # We'll prepare for these three and log if others appear or if some are missing.
    expected_source_types = ["P_style", "related_word", "random_word"]
    metrics_by_source_type: Dict[str, Dict[str, Any]] = {
        st: {
            'total_items': 0, 
            'exact_match_sum': 0, 
            'max_f1_scores': [], 
            'max_f1_inferred_scores': [],
            'total_tokens': [],
            'prompt_tokens': [],
            'completion_tokens': [],
            'non_prompt_tokens': []
        } 
        for st in expected_source_types
    }
    # For items that might have an unexpected or missing source_type, if we decide to still process them globally
    # but not categorize them under the main three. For now, we assume valid_source_types are enforced upstream or items without it are handled.
    # If an item has source_type = None or an unexpected one, it contributes to overall but not to the specific categories.

    all_item_eval_details_for_possible_debug = [] # Not for primary CSV, but useful for internal checks

    logger.info("Calculating metrics for each item and categorizing by source_type...")
    for item in processed_data:
        predicted_set = item['predicted_set']
        golden_sets = item['golden_sets']
        fact_set = item.get('fact_set', set())
        item_id = item['id']
        item_source_type = item.get('source_type') # This should be one of "P_style", "related_word", "random_word", or None
        token_info = item.get('token_info', {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'non_prompt_tokens': 0})

        # 1. Exact Match calculation for the item
        item_exact_match = 0
        for golden_set in golden_sets:
            if predicted_set == golden_set:
                item_exact_match = 1
                break
        overall_exact_match_scores.append(item_exact_match)

        # 2. Max F1 Score calculation for the item
        item_f1_scores = []
        if not golden_sets:
            logger.warning(f"Item {item_id}: No golden sets provided. F1 score defaulting to 0.")
            item_f1_scores.append(0.0)
        else:
            for golden_set in golden_sets:
                _, _, f1 = calculate_set_prf1(predicted_set, golden_set)
                item_f1_scores.append(f1)
        item_max_f1 = max(item_f1_scores) if item_f1_scores else 0.0
        overall_max_f1_scores.append(item_max_f1)

        # 3. Max F1 Score for Inferred Conclusions calculation for the item
        item_f1_inferred_scores = []
        predicted_set_inferred = predicted_set - fact_set
        if not golden_sets:
            item_f1_inferred_scores.append(0.0)
        else:
            for golden_set in golden_sets:
                golden_set_inferred = golden_set - fact_set
                if not predicted_set_inferred and not golden_set_inferred:
                     _, _, f1_inferred = 1.0, 1.0, 1.0 
                elif not golden_set_inferred and len(predicted_set_inferred) > 0 :
                     _, _, f1_inferred = 0.0, 0.0, 0.0
                elif not predicted_set_inferred and len(golden_set_inferred) > 0:
                     _, _, f1_inferred = 0.0, 0.0, 0.0
                else:
                    _, _, f1_inferred = calculate_set_prf1(predicted_set_inferred, golden_set_inferred)
                item_f1_inferred_scores.append(f1_inferred)
        item_max_f1_inferred = max(item_f1_inferred_scores) if item_f1_inferred_scores else 0.0
        overall_max_f1_inferred_scores.append(item_max_f1_inferred)

        # 4. Collect token usage information
        overall_total_tokens.append(token_info['total_tokens'])
        overall_prompt_tokens.append(token_info['prompt_tokens'])
        overall_completion_tokens.append(token_info['completion_tokens'])
        overall_non_prompt_tokens.append(token_info['non_prompt_tokens'])
        
        all_item_eval_details_for_possible_debug.append({
            'id': item_id, 'source_type': item_source_type, 
            'exact_match': item_exact_match, 'max_f1': item_max_f1, 
            'max_f1_inferred': item_max_f1_inferred,
            'token_info': token_info
        })

        # Aggregate for the correct source_type
        if item_source_type in metrics_by_source_type:
            metrics_by_source_type[item_source_type]['total_items'] += 1
            metrics_by_source_type[item_source_type]['exact_match_sum'] += item_exact_match
            metrics_by_source_type[item_source_type]['max_f1_scores'].append(item_max_f1)
            metrics_by_source_type[item_source_type]['max_f1_inferred_scores'].append(item_max_f1_inferred)
            metrics_by_source_type[item_source_type]['total_tokens'].append(token_info['total_tokens'])
            metrics_by_source_type[item_source_type]['prompt_tokens'].append(token_info['prompt_tokens'])
            metrics_by_source_type[item_source_type]['completion_tokens'].append(token_info['completion_tokens'])
            metrics_by_source_type[item_source_type]['non_prompt_tokens'].append(token_info['non_prompt_tokens'])
        elif item_source_type is not None: # A source_type was present but not one of the expected ones
             logger.warning(f"Item {item_id} has source_type '{item_source_type}' which is not in expected list {expected_source_types}. It will contribute to overall metrics but not to a specific category count in this pre-defined structure.")
        # If item_source_type is None, it only contributes to overall metrics.

    # --- Calculate Overall Dataset Averages ---
    total_items_overall = len(processed_data)
    overall_avg_exact_match = sum(overall_exact_match_scores) / total_items_overall if total_items_overall > 0 else 0.0
    overall_sum_exact_match = sum(overall_exact_match_scores)
    overall_avg_max_f1 = sum(overall_max_f1_scores) / total_items_overall if total_items_overall > 0 else 0.0
    overall_avg_max_f1_inferred = sum(overall_max_f1_inferred_scores) / total_items_overall if total_items_overall > 0 else 0.0

    # --- Calculate Token Usage Statistics ---
    overall_total_tokens_sum = sum(overall_total_tokens)
    overall_prompt_tokens_sum = sum(overall_prompt_tokens)
    overall_completion_tokens_sum = sum(overall_completion_tokens)
    overall_non_prompt_tokens_sum = sum(overall_non_prompt_tokens)
    
    overall_avg_total_tokens = overall_total_tokens_sum / total_items_overall if total_items_overall > 0 else 0.0
    overall_avg_prompt_tokens = overall_prompt_tokens_sum / total_items_overall if total_items_overall > 0 else 0.0
    overall_avg_completion_tokens = overall_completion_tokens_sum / total_items_overall if total_items_overall > 0 else 0.0
    overall_avg_non_prompt_tokens = overall_non_prompt_tokens_sum / total_items_overall if total_items_overall > 0 else 0.0

    logger.info("--- Overall Dataset Evaluation Results ---")
    logger.info(f"Total items evaluated: {total_items_overall}")
    logger.info(f"Average Exact Match Score: {overall_avg_exact_match:.4f} ({overall_sum_exact_match}/{total_items_overall})")
    logger.info(f"Average Max F1 Score (Overall): {overall_avg_max_f1:.4f}")
    logger.info(f"Average Max F1 Score (Inferred): {overall_avg_max_f1_inferred:.4f}")
    logger.info("--- Token Usage Statistics ---")
    logger.info(f"Total Tokens Used: {overall_total_tokens_sum} (Average: {overall_avg_total_tokens:.2f} per item)")
    logger.info(f"Total Prompt Tokens: {overall_prompt_tokens_sum} (Average: {overall_avg_prompt_tokens:.2f} per item)")
    logger.info(f"Total Completion Tokens: {overall_completion_tokens_sum} (Average: {overall_avg_completion_tokens:.2f} per item)")
    logger.info(f"Total Non-Prompt Tokens: {overall_non_prompt_tokens_sum} (Average: {overall_avg_non_prompt_tokens:.2f} per item)")
    logger.info("---------------------------------------------")

    # --- Calculate and Log Category-Specific Metrics ---
    logger.info("--- Metrics by source_type ---")
    final_metrics_by_source_type = {}
    csv_data_rows = []

    for st_name, data in metrics_by_source_type.items():
        count = data["total_items"]
        em_sum = data["exact_match_sum"]
        
        avg_em = (em_sum / count) if count > 0 else 0.0
        avg_f1 = (sum(data["max_f1_scores"]) / count) if count > 0 else 0.0
        avg_f1_inferred = (sum(data["max_f1_inferred_scores"]) / count) if count > 0 else 0.0
        
        # Calculate token usage statistics for this source_type
        total_tokens_sum = sum(data["total_tokens"])
        prompt_tokens_sum = sum(data["prompt_tokens"])
        completion_tokens_sum = sum(data["completion_tokens"])
        non_prompt_tokens_sum = sum(data["non_prompt_tokens"])
        
        avg_total_tokens = total_tokens_sum / count if count > 0 else 0.0
        avg_prompt_tokens = prompt_tokens_sum / count if count > 0 else 0.0
        avg_completion_tokens = completion_tokens_sum / count if count > 0 else 0.0
        avg_non_prompt_tokens = non_prompt_tokens_sum / count if count > 0 else 0.0
        
        final_metrics_by_source_type[st_name] = {
            'total_items': count,
            'average_exact_match': round(avg_em, 4),
            'exact_match_sum': em_sum,
            'average_max_f1': round(avg_f1, 4),
            'average_max_f1_inferred': round(avg_f1_inferred, 4),
            'token_usage': {
                'total_tokens_sum': total_tokens_sum,
                'prompt_tokens_sum': prompt_tokens_sum,
                'completion_tokens_sum': completion_tokens_sum,
                'non_prompt_tokens_sum': non_prompt_tokens_sum,
                'avg_total_tokens': round(avg_total_tokens, 2),
                'avg_prompt_tokens': round(avg_prompt_tokens, 2),
                'avg_completion_tokens': round(avg_completion_tokens, 2),
                'avg_non_prompt_tokens': round(avg_non_prompt_tokens, 2)
            }
        }
        if count > 0: # Only log and add to CSV if there are items for this source_type
            logger.info(f"source_type: {st_name}")
            logger.info(f"  Total items: {count}")
            logger.info(f"  Average Exact Match: {avg_em:.4f} ({em_sum}/{count})")
            logger.info(f"  Average Max F1 (Overall): {avg_f1:.4f}")
            logger.info(f"  Average Max F1 (Inferred): {avg_f1_inferred:.4f}")
            logger.info(f"  Token Usage: Total={total_tokens_sum} (avg={avg_total_tokens:.2f}), Prompt={prompt_tokens_sum} (avg={avg_prompt_tokens:.2f}), Completion={completion_tokens_sum} (avg={avg_completion_tokens:.2f}), Non-Prompt={non_prompt_tokens_sum} (avg={avg_non_prompt_tokens:.2f})")

            # Add rows to CSV data list
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Exact Match', 
                'Result': f"{avg_em:.4f}", 'Support (Items)': count
            })
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Max F1 (Overall)', 
                'Result': f"{avg_f1:.4f}", 'Support (Items)': count
            })
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Max F1 (Inferred)', 
                'Result': f"{avg_f1_inferred:.4f}", 'Support (Items)': count
            })
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Total Tokens', 
                'Result': f"{avg_total_tokens:.2f}", 'Support (Items)': count
            })
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Prompt Tokens', 
                'Result': f"{avg_prompt_tokens:.2f}", 'Support (Items)': count
            })
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Completion Tokens', 
                'Result': f"{avg_completion_tokens:.2f}", 'Support (Items)': count
            })
            csv_data_rows.append({
                'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Non-Prompt Tokens', 
                'Result': f"{avg_non_prompt_tokens:.2f}", 'Support (Items)': count
            })
        else:
            logger.info(f"source_type: {st_name} (0 items)")
            # Add empty/default rows to CSV if you want to represent all expected source_types regardless of count
            # For now, only adding if count > 0, matching the logging.
            # If you always want a row for P_style, related_word, random_word:
            # csv_data_rows.append({'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Exact Match', 'Result': "0.0000", 'Support (Items)': 0})
            # csv_data_rows.append({'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Max F1 (Overall)', 'Result': "0.0000", 'Support (Items)': 0})
            # csv_data_rows.append({'Property': 'source_type', 'Value': st_name, 'Metric': 'Average Max F1 (Inferred)', 'Result': "0.0000", 'Support (Items)': 0})


    logger.info("---------------------------------------------")

    # Prepare final metrics dictionary for JSON output
    metrics_for_json = {
        'overall_dataset_metrics': {
            'total_items': total_items_overall,
            'average_exact_match': round(overall_avg_exact_match, 4),
            'exact_match_sum': overall_sum_exact_match,
            'average_max_f1': round(overall_avg_max_f1, 4),
            'average_max_f1_inferred': round(overall_avg_max_f1_inferred, 4),
            'token_usage': {
                'total_tokens_sum': overall_total_tokens_sum,
                'prompt_tokens_sum': overall_prompt_tokens_sum,
                'completion_tokens_sum': overall_completion_tokens_sum,
                'non_prompt_tokens_sum': overall_non_prompt_tokens_sum,
                'avg_total_tokens': round(overall_avg_total_tokens, 2),
                'avg_prompt_tokens': round(overall_avg_prompt_tokens, 2),
                'avg_completion_tokens': round(overall_avg_completion_tokens, 2),
                'avg_non_prompt_tokens': round(overall_avg_non_prompt_tokens, 2)
            }
        },
        'metrics_by_source_type': final_metrics_by_source_type
        # 'all_item_eval_details': all_item_eval_details_for_possible_debug # Optional: if needed for other purposes
    }
    
    # Return both JSON-structured metrics and the list for CSV generation
    return {
        "json_metrics": metrics_for_json,
        "csv_data": csv_data_rows
    }

def run_evaluation_answer_set_generation(
    results_file_path: str,
    original_dataset_file_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Main function to run the Answer Set Generation evaluation pipeline.

    :param results_file_path: Path to the JSONL file containing the results (model predictions).
    :param original_dataset_file_path: Optional path to the original JSONL dataset file for facts and golden truths.
    :return: Dictionary of aggregated metrics, or None on failure.
    """
    logger.info(f"Starting evaluation pipeline for Answer Set Generation using predictions file: {results_file_path}")
    if original_dataset_file_path:
        logger.info(f"Using original dataset file: {original_dataset_file_path}")

    processed_data = load_and_process_generation_data(
        predictions_file_path=results_file_path,
        original_dataset_file_path=original_dataset_file_path
    )

    if processed_data is None:
        # Error already logged in load_and_process_generation_data
        return None

    metrics = calculate_and_log_generation_metrics(processed_data)

    logger.info(f"Evaluation pipeline for {results_file_path} finished.")
    return metrics


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Answer Set Generation Evaluation")
    parser.add_argument("results_file", type=str, help="Path to the results JSONL file (output from generation step).")
    parser.add_argument("--original_dataset_file", type=str, default=None, help="Optional path to the original dataset JSONL file (containing facts, golden_answer_sets, and source_type).")
    # Add other arguments if needed

    args = parser.parse_args()
    results_file_basename = os.path.splitext(os.path.basename(args.results_file))[0]


    logger.info(f"Running Answer Set Generation evaluation directly for predictions file: {args.results_file}")
    if args.original_dataset_file:
        logger.info(f"Using original dataset file for source_type and other metadata: {args.original_dataset_file}")
        
    evaluation_output = run_evaluation_answer_set_generation(
        results_file_path=args.results_file,
        original_dataset_file_path=args.original_dataset_file
    )

    if evaluation_output and "json_metrics" in evaluation_output and "csv_data" in evaluation_output:
        json_metrics_to_save = evaluation_output["json_metrics"]
        csv_data_to_save = evaluation_output["csv_data"]

        logger.info(f"Direct evaluation summary (JSON format):{os.linesep}{json.dumps(json_metrics_to_save, indent=2)}")
        
        # Save summary metrics to JSON file
        json_output_filename = f"{results_file_basename}_eval_metrics.json"
        # Construct path relative to the input file's directory or a fixed 'results' dir
        # Assuming results_file arg might be a full path
        output_dir = os.path.dirname(args.results_file) # Save in the same dir as input results
        if not output_dir: # If results_file is just a name, save in current dir or a predefined one
            output_dir = "." 
        
        metrics_json_output_path = os.path.join(output_dir, json_output_filename)

        try:
            with open(metrics_json_output_path, 'w', encoding='utf-8') as f:
                 json.dump(json_metrics_to_save, f, indent=4, ensure_ascii=False)
            logger.info(f"Evaluation metrics summary saved to JSON: {metrics_json_output_path}")
        except IOError as e:
            logger.error(f"Failed to save evaluation metrics summary to JSON {metrics_json_output_path}: {e}")
        except TypeError as e: 
             logger.error(f"Failed to serialize evaluation metrics summary to JSON: {e}")

        # Save property impact analysis to CSV file
        if csv_data_to_save:
            try:
                property_impact_df = pd.DataFrame(csv_data_to_save)
                # Use the same directory as the JSON output, or specific analysis directory
                csv_output_filename = f"analysis_property_impact_{results_file_basename}.csv"
                property_impact_csv_path = os.path.join(output_dir, csv_output_filename)
                
                property_impact_df.to_csv(property_impact_csv_path, index=False, encoding='utf-8', float_format='%.4f')
                logger.info(f"Property impact analysis saved to CSV: {property_impact_csv_path}")
            except Exception as e:
                logger.error(f"Failed to save property impact analysis to CSV: {e}", exc_info=True)
        else:
            logger.info("No data available to save for property impact CSV.")
            
    else:
        logger.error("Direct evaluation run failed or did not produce expected output structure. Check logs for details.")
        if evaluation_output is None: # run_evaluation_answer_set_generation itself failed
             logger.error("run_evaluation_answer_set_generation returned None.")
        sys.exit(1) 