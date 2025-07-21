# src/symtex_evaluation/common_metrics.py
from typing import List, Dict, Any, Optional, Tuple, Set

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import pandas as pd

def calculate_set_prf1(predicted_set: Set[Any], golden_set: Set[Any]) -> Tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F1 score between two sets.

    :param predicted_set: Set of predicted items.
    :type predicted_set: Set[Any]
    :param golden_set: Set of golden (ground truth) items.
    :type golden_set: Set[Any]
    :return: A tuple containing (precision, recall, f1_score).
    :rtype: Tuple[float, float, float]
    """
    if not isinstance(predicted_set, set):
        # Attempt to convert if not already a set, assuming iterable elements
        try:
            predicted_set = set(predicted_set)
        except TypeError:
            raise ValueError("predicted_set must be a set or convertible to a set.")
            
    if not isinstance(golden_set, set):
        try:
            golden_set = set(golden_set)
        except TypeError:
            raise ValueError("golden_set must be a set or convertible to a set.")

    true_positives = len(predicted_set.intersection(golden_set))
    predicted_positives = len(predicted_set)
    actual_positives = len(golden_set)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def compute_classification_metrics(
    labels: List[str],
    predictions: List[str],
    labels_for_report: Optional[List[str]] = None,
    target_names_for_report: Optional[List[str]] = None,
    pos_label_for_binary_f1: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes common classification metrics using sklearn.

    :param labels: List of true labels.
    :type labels: List[str]
    :param predictions: List of predicted labels.
    :type predictions: List[str]
    :param labels_for_report: Explicit list of labels to include in metrics calculation (e.g., for confusion matrix or F1).
                              If None, inferred from data.
    :type labels_for_report: Optional[List[str]]
    :param target_names_for_report: Display names for labels in classification_report. If None, uses labels_for_report.
    :type target_names_for_report: Optional[List[str]]
    :param pos_label_for_binary_f1: The positive label if binary F1 score is desired.
    :type pos_label_for_binary_f1: Optional[str]
    :return: Dictionary containing calculated metrics.
    :rtype: Dict[str, Any]
    """
    if not labels or not predictions:
        return {
            'accuracy': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'weighted_f1': 0.0,
            'binary_f1': 0.0 if pos_label_for_binary_f1 else None,
            'classification_report': "N/A - No data",
            'confusion_matrix_string': "N/A - No data",
            'confusion_matrix_df': pd.DataFrame(),
            'unique_labels_found': []
        }

    if len(labels) != len(predictions):
        raise ValueError(f"Label count ({len(labels)}) and prediction count ({len(predictions)}) mismatch.")

    actual_unique_labels = sorted(list(set(labels) | set(predictions)))
    
    # Use provided labels_for_report if available, otherwise use all unique labels found in the data
    report_labels_to_use = labels_for_report if labels_for_report is not None else actual_unique_labels
    
    # Ensure target_names_for_report matches report_labels_to_use if not provided or if lengths differ
    report_target_names_to_use = target_names_for_report
    if report_target_names_to_use is None or len(report_target_names_to_use) != len(report_labels_to_use):
        report_target_names_to_use = report_labels_to_use

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, labels=report_labels_to_use, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, predictions, labels=report_labels_to_use, average='micro', zero_division=0) # Micro-F1 is accuracy for multi-class
    weighted_f1 = f1_score(labels, predictions, labels=report_labels_to_use, average='weighted', zero_division=0)
    
    binary_f1_score_val = None
    if pos_label_for_binary_f1:
        if pos_label_for_binary_f1 in report_labels_to_use: # Check if pos_label is among the considered labels
            binary_f1_score_val = f1_score(labels, predictions, labels=report_labels_to_use, pos_label=pos_label_for_binary_f1, average='binary', zero_division=0)
        else: # pos_label not in actual data labels, or not in the specified report_labels
             # This case might mean the positive class was not present, or not specified for detailed reporting.
             # F1 for a non-existent or non-specified positive class is typically 0 if it's expected.
            binary_f1_score_val = 0.0

    class_report_str = classification_report(
        labels, predictions, 
        labels=report_labels_to_use, 
        target_names=report_target_names_to_use, 
        digits=4, zero_division=0
    )
    
    cm = confusion_matrix(labels, predictions, labels=report_labels_to_use)
    # Ensure columns and index for cm_df also use report_labels_to_use for consistency
    cm_df = pd.DataFrame(cm, index=[f'True_{l}' for l in report_labels_to_use], columns=[f'Pred_{l}' for l in report_labels_to_use])
    cm_string = cm_df.to_string()

    return {
        'accuracy': accuracy, # Same as micro_f1 in multi-class settings
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'binary_f1': binary_f1_score_val, # F1 for the pos_label if specified
        'classification_report': class_report_str,
        'confusion_matrix_string': cm_string,
        'confusion_matrix_df': cm_df,
        'unique_labels_found': actual_unique_labels # All unique labels present in data
    }

def normalize_atom_string(atom_str: str) -> str:
    """
    Normalizes a string representation of a logical atom.
    Removes leading/trailing whitespace, trailing period, all internal quotes,
    and normalizes spaces around parentheses, commas, and leading signs.

    :param atom_str: The atom string to normalize.
    :type atom_str: str
    :return: The normalized atom string.
    :rtype: str
    """
    s = str(atom_str).strip()
    s = s.removesuffix('.')  # Python 3.9+
    s = s.replace('"', '')   # Remove all internal quotes

    # Normalize spaces around parentheses and commas
    s = s.replace(" (", "(").replace("( ", "(")
    s = s.replace(" )", ")").replace(") ", ")")
    s = s.replace(" ,", ",").replace(", ", ",")
    
    # Normalize spaces around ':-' if it's a rule (less common for facts but good for general utility)
    s = s.replace(" :-", ":-").replace(":- ", ":-")

    # Handle space after leading sign, e.g., "- predicate" -> "-predicate"
    if s.startswith("- ") and len(s) > 2:
        s = "-" + s[2:]
    # '+' is often implicit but handle if explicitly used with a space
    elif s.startswith("+ ") and len(s) > 2:
        s = "+" + s[2:]
    
    return s 