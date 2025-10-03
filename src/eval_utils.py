import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_exact_match_rate(predicted_actions: np.ndarray, gt_actions: np.ndarray) -> np.ndarray:
    """Get the exact match rate of the actions"""
    # Ensure inputs are numpy arrays and squeeze extra dimensions
    predicted_actions = np.asarray(predicted_actions).squeeze() # from (5, 1, 1) to (5,)
    gt_actions = np.asarray(gt_actions).squeeze() # from (5, 1, 1) to (5,)
    
    if predicted_actions.shape != gt_actions.shape or predicted_actions.size == 0:
        raise ValueError("Unmatched action shapes or empty action arrays")

    exact_match_rate = np.mean(predicted_actions == gt_actions)
    return float(exact_match_rate)


def calculate_tp_fp_fn_counts(
        predicted_actions: np.ndarray, gt_actions: np.ndarray, all_labels: list) -> tuple[int, int, int, int, int]:
    """
    Helper function to calculate true positives, false positives, and false negatives.
    
    Args:
        predicted_actions: Array of predicted action labels.
        gt_actions: Array of ground truth action labels.
        all_labels: List of all possible valid action labels.
        
    Returns:
        Tuple of (total_tp, total_fp, total_fn, valid_fp, invalid_fp)
    """
    # Ensure inputs are numpy arrays
    predicted_actions = np.asarray(predicted_actions).squeeze() # from (5, 1, 1) to (5,)
    gt_actions = np.asarray(gt_actions).squeeze() # from (5, 1, 1) to (5,)

    if predicted_actions.shape != gt_actions.shape:
        raise ValueError("Predicted and ground truth actions must have the same shape.")
        
    if len(all_labels) == 0 or predicted_actions.size == 0:
        raise ValueError("No valid action labels or empty action arrays")

    # Step 1: Handle invalid predictions
    # Each invalid prediction counts as one false positive
    invalid_predictions = ~np.isin(predicted_actions, all_labels)
    invalid_fp = np.sum(invalid_predictions)

    # Step 2: Handle valid predictions
    valid_fp = 0
    total_tp = 0
    
    # Count true positives and false positives for valid predictions
    for label in all_labels:
        pred_matches = (predicted_actions == label)
        gt_matches = (gt_actions == label)
        
        total_tp += np.sum(pred_matches & gt_matches)
        valid_fp += np.sum(pred_matches & ~gt_matches)
    
    # Step 3: Calculate total false negatives
    # For each ground truth label, count how many times we missed it
    total_fn = 0
    for label in all_labels:
        gt_matches = (gt_actions == label)
        pred_matches = (predicted_actions == label)
        total_fn += np.sum(gt_matches & ~pred_matches)
    
    # Step 4: Combine metrics
    total_fp = invalid_fp + valid_fp

    return total_tp, total_fp, total_fn, valid_fp, invalid_fp


def get_micro_precision_from_counts(total_tp: int, total_fp: int) -> float:
    """Calculate precision from precomputed counts."""
    if total_tp + total_fp == 0:  # No positive predictions
        return 0.0
    return float(total_tp / (total_tp + total_fp))


def get_micro_recall_from_counts(total_tp: int, total_fn: int) -> float:
    """Calculate recall from precomputed counts."""
    if total_tp + total_fn == 0:  # No positive ground truth
        return 0.0
    return float(total_tp / (total_tp + total_fn))


def get_micro_f1(precision: float, recall: float) -> float:
    """Calculate micro F1 score from precision and recall values."""
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))

def get_precision_per_class(
        predicted_actions: np.ndarray, gt_actions: np.ndarray, all_labels: list) -> dict[int, float]:
    """Calculate macro precision by computing precision for each class label.
    Returns a dictionary mapping class labels to their precision values."""
    predicted_actions = np.asarray(predicted_actions).squeeze()
    gt_actions = np.asarray(gt_actions).squeeze()

    if predicted_actions.shape != gt_actions.shape:
        raise ValueError("Predicted and ground truth actions must have the same shape.")
        
    if len(all_labels) == 0 or predicted_actions.size == 0:
        raise ValueError("No valid action labels or empty action arrays")
    
    # Initialize dictionary to store class precisions
    class_precisions = {}
    
    # For each unique class label
    for label in set(all_labels):
        # Get predictions and ground truth matches for this class
        pred_matches = (predicted_actions == label)
        gt_matches = (gt_actions == label)
        
        # Calculate true positives and false positives for this class
        class_tp = np.sum(pred_matches & gt_matches)
        class_fp = np.sum(pred_matches & ~gt_matches)
        
        # Calculate precision for this class
        if class_tp + class_fp == 0:
            class_precisions[label] = 0.0
        else:
            class_precisions[label] = float(class_tp / (class_tp + class_fp))
    
    return class_precisions

def get_recall_per_class(
        predicted_actions: np.ndarray, gt_actions: np.ndarray, all_labels: list) -> dict[int, float]:
    """Calculate macro recall by computing recall for each class label.
    Returns a dictionary mapping class labels to their recall values."""
    predicted_actions = np.asarray(predicted_actions).squeeze()
    gt_actions = np.asarray(gt_actions).squeeze()

    if predicted_actions.shape != gt_actions.shape:
        raise ValueError("Predicted and ground truth actions must have the same shape.")
        
    if len(all_labels) == 0 or predicted_actions.size == 0:
        raise ValueError("No valid action labels or empty action arrays")
    
    class_recalls = {}
    
    for label in set(all_labels):
        pred_matches = (predicted_actions == label)
        gt_matches = (gt_actions == label)
        
        # Calculate true positives and false negatives for this class
        class_tp = np.sum(pred_matches & gt_matches)
        class_fn = np.sum(~pred_matches & gt_matches)
        
        # Calculate recall for this class
        if class_tp + class_fn == 0:
            class_recalls[label] = 0.0
        else:
            class_recalls[label] = float(class_tp / (class_tp + class_fn))
    
    return class_recalls

def get_f1_per_class(
        class_precisions: dict[int, float], class_recalls: dict[int, float]) -> dict[int, float]:
    """Calculate macro F1 score by computing F1 for each class label.
    Returns a dictionary mapping class labels to their F1 values."""
    class_f1s = {}
    for label in set(class_precisions.keys()):
        precision = class_precisions[label]
        recall = class_recalls[label]
        if precision + recall == 0:
            class_f1s[label] = 0.0
        else:
            class_f1s[label] = float(2 * (precision * recall) / (precision + recall))

    return class_f1s

def get_macro_precision(class_precisions: dict[int, float]) -> float:
    """Calculate macro precision by averaging precision across all classes."""
    return float(np.mean(list(class_precisions.values())))

def get_macro_recall(class_recalls: dict[int, float]) -> float:
    """Calculate macro recall by averaging recall across all classes."""
    return float(np.mean(list(class_recalls.values())))

def get_macro_f1(class_f1s: dict[int, float]) -> float:
    """Calculate macro F1 score by averaging F1 scores across all classes."""
    return float(np.mean(list(class_f1s.values())))

def calculate_mae(predicted_action, actual_action) -> float:
    """Calculate mean absolute error from absolute errors"""
    return np.mean(np.abs(np.array(predicted_action) - np.array(actual_action)))

def calculate_mse(predicted, actual) -> float:
    """Calculate mean squared error between predicted and actual values"""
    return np.mean((np.array(predicted) - np.array(actual)) ** 2)

def calculate_brier_mae(predicted_probabilities, one_hot_label) -> float:
    """Calculate mean absolute error from absolute errors"""
    return np.sum(np.abs(np.array(predicted_probabilities) - np.array(one_hot_label)))

def calculate_brier_mse(predicted_probabilities, one_hot_label) -> float:
    """Calculate mean absolute error from absolute errors"""
    return np.sum((np.array(predicted_probabilities) - np.array(one_hot_label)) ** 2)

def calculate_success_rate(success_list) -> float:
    """Calculate success rate percentage from a list of success indicators"""
    if len(success_list) == 0:
        logger.warning("Success list is empty. Defaulting to 0.0 success rate.")
        return 0.0
    return (sum(success_list) / len(success_list)) * 100

def quantile_filter(values: list[float], quantile: float = 0.05) -> list[float]:
    """Filter values within `quantile` and `100 - quantile` quantiles"""
    if len(values) == 0:
        raise ValueError("No values collected")

    q5 = np.percentile(values, quantile)
    q95 = np.percentile(values, 100 - quantile)

    return [val for val in values if q5 <= val <= q95]

def min_max_normalize(values: list[float]) -> list[float]:
    """Normalize values using min-max scaling"""
    if len(values) == 0:
        raise ValueError("No values collected")

    if len(values) == 1:
        raise ValueError("Only one value collected")

    min_val = min(values)
    max_val = max(values)
    normalized_values = np.array(values)
    return (normalized_values - min_val) / (max_val - min_val) if max_val != min_val else np.nan

def calculate_mean(values: list[float]) -> float:
    """Calculate average of a list of values"""
    if len(values) == 0:
        raise ValueError("No values collected")
    return np.mean(values)


def calculate_max_relative_mae(mae_values: list[float]) -> float:
    """Calculate maximum relative error (outlier severity)"""
    if len(mae_values) == 0:
        raise ValueError("No values collected")

    abs_values = np.abs(np.array(mae_values))
    median_value = np.median(abs_values)
    if median_value == 0:
        median_value = 1e-10
    return max(abs_values) / median_value


def calculate_proportion_beyond_mae_threshold(mae_values: list[float], threshold_multiplier: float = 3.0) -> float:
    """Calculate proportion of values beyond threshold (outlier frequency)"""
    if len(mae_values) == 0:
        raise ValueError("No values collected")

    median_value = np.median(mae_values)
    threshold = threshold_multiplier * median_value
    return sum(1 for value in mae_values if value > threshold) / len(mae_values)

