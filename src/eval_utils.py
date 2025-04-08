
import numpy as np


def get_exact_match_rate(predicted_actions: np.ndarray, gt_actions: np.ndarray) -> np.ndarray:
    """Get the exact match rate of the actions"""
    # Ensure inputs are numpy arrays and squeeze extra dimensions
    predicted_actions = np.asarray(predicted_actions).squeeze() # from (5, 1, 1) to (5,)
    gt_actions = np.asarray(gt_actions).squeeze() # from (5, 1, 1) to (5,)
    
    if predicted_actions.shape != gt_actions.shape or predicted_actions.size == 0:
        raise ValueError("Unmatched action shapes or empty action arrays")

    exact_match_rate = np.mean(predicted_actions == gt_actions)
    return float(exact_match_rate)


def calculate_tp_fp_fn_counts(predicted_actions: np.ndarray, gt_actions: np.ndarray, all_labels: list) -> tuple[int, int, int]:
    """
    Helper function to calculate true positives, false positives, and false negatives.
    
    Args:
        predicted_actions: Array of predicted action labels.
        gt_actions: Array of ground truth action labels.
        all_labels: List of all possible valid action labels.
        
    Returns:
        Tuple of (total_tp, total_fp, total_fn)
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
    
    return total_tp, total_fp, total_fn


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
