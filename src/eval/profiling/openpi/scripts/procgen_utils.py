import numpy as np
from definitions.procgen import ProcGenDefinitions


class ActionUtils:
    @staticmethod
    def set_procgen_unused_special_action_to_stand_still(actions: np.ndarray, dataset_name: str) -> np.ndarray:
        """
        Set unused action in procgen to stand still action index 4

        Args:
            actions (np.ndarray): Array of actions for an episode.
            dataset_name (str): The name of the dataset to use for default values.

        Returns:
            np.ndarray: The modified action array with invalid actions replaced with stand still (4).
        """
        actions = np.atleast_1d(np.asarray(actions))
        valid_action_space = ProcGenDefinitions.get_valid_action_space(dataset_name, 'default')
        
        # Create a boolean mask for invalid actions
        invalid_mask = ~np.isin(actions, valid_action_space)
        
        # Create a copy and use the mask to replace invalid actions
        modified_actions = actions.copy()
        modified_actions[invalid_mask] = 4
        
        return modified_actions


class MetricUtils:    
    @staticmethod
    def get_exact_match_rate(predicted_actions: np.ndarray, gt_actions: np.ndarray) -> np.ndarray:
        """Get the exact match rate of the actions"""
        # Ensure inputs are numpy arrays and squeeze extra dimensions
        predicted_actions = np.asarray(predicted_actions).squeeze() # from (5, 1, 1) to (5,)
        gt_actions = np.asarray(gt_actions).squeeze() # from (5, 1, 1) to (5,)
        
        if predicted_actions.shape != gt_actions.shape or predicted_actions.size == 0:
            raise ValueError("Unmatched action shapes or empty action arrays")

        exact_match_rate = np.mean(predicted_actions == gt_actions)
        return float(exact_match_rate)

    @staticmethod
    def _calculate_metrics_counts(predicted_actions: np.ndarray, gt_actions: np.ndarray, all_labels: list) -> tuple[int, int, int]:
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

        # Initialize total counts
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Accumulate counts across all labels
        for label in all_labels:
            # Boolean arrays for the current label
            pred_is_label = (predicted_actions == label)
            gt_is_label = (gt_actions == label)

            # Add to total counts
            total_tp += np.sum(pred_is_label & gt_is_label)
            total_fp += np.sum(pred_is_label & ~gt_is_label)
            total_fn += np.sum(~pred_is_label & gt_is_label)

        return total_tp, total_fp, total_fn

    @staticmethod
    def get_micro_precision_from_counts(total_tp: int, total_fp: int) -> float:
        """Calculate precision from precomputed counts."""
        if total_tp + total_fp == 0:  # No positive predictions
            return 0.0
        return float(total_tp / (total_tp + total_fp))

    @staticmethod
    def get_micro_recall_from_counts(total_tp: int, total_fn: int) -> float:
        """Calculate recall from precomputed counts."""
        if total_tp + total_fn == 0:  # No positive ground truth
            return 0.0
        return float(total_tp / (total_tp + total_fn))

    @staticmethod
    def get_micro_f1(precision: int, recall: int) -> float:
        """Calculate micro F1 score from precomputed counts."""
        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))
