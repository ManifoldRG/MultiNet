
from typing import Dict, Any, List
import numpy as np
from src.eval_utils import (
    calculate_mse, calculate_mae,
    calculate_mean, min_max_normalize, quantile_filter,
    calculate_max_relative_mae, calculate_proportion_beyond_mae_threshold
)

def _validate_output(output: Any, shape: tuple[int]) -> bool:
    """Validate that output is a valid action vector."""

    if output is None or not isinstance(output, (list, tuple, np.ndarray)) or len(output) != shape[0]:
        return False

    # TODO: Allow outputs that are partially correctly formatted,
    # and give max penalty (max - min of action stats) for invalid dimensions
    for item in output:
        if not isinstance(item, (float, int, np.floating, np.integer)):
            return False
        
        # Disallow nan/inf
        if np.isnan(item) or np.isinf(item):
            return False
    
    return True

class RoboticsMetricsCalculator:
    """
    Simple calculator for robotics metrics.
    
    Takes predictions and ground truth actions, calculates MSE and MAE metrics.
    """
    
    def __init__(self, action_stats: Dict[str, Any]):
        """
        Initialize with action statistics.
        
        Args:
            action_stats: Dictionary with 'max' and 'min' action values
        """
        self.action_stats = action_stats
    
    def calculate_metrics(self, predictions: List[Any], ground_truth_actions: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calculate metrics for predictions vs ground truth.
        
        Args:
            predictions: List of structured predictions with "extracted_outputs" key
            ground_truth_actions: List of ground truth action vectors
            
        Returns:
            Dictionary containing calculated metrics
        """
        mses, maes = [], []
        total_invalid_preds = 0
        action_success = []

        for i, pred_dict in enumerate(predictions):
            # Extract action vector from structured format
            pred = pred_dict["extracted_outputs"] if isinstance(pred_dict, dict) else pred_dict

            if _validate_output(pred, shape=ground_truth_actions[i].shape):
                pred = [float(item) for item in pred]
                mses.append(calculate_mse(pred, ground_truth_actions[i]))
                maes.append(calculate_mae(pred, ground_truth_actions[i]))

                # Calculate action success for exact matches (mirroring OpenX implementation)
                if np.array_equal(np.array(pred), ground_truth_actions[i]):
                    action_success.append(1)
                else:
                    action_success.append(0)
            else:
                # max value of MSE/MAE for invalid outputs
                max_vals = np.array(self.action_stats['max'])
                min_vals = np.array(self.action_stats['min'])
                mse = calculate_mse(max_vals, min_vals)
                mae = calculate_mae(max_vals, min_vals)
                mses.append(mse)
                maes.append(mae)
                total_invalid_preds += 1
                action_success.append(0)  # Invalid predictions are considered failures

        return self._calculate_final_metrics(mses, maes, total_invalid_preds, action_success)
    
    def _calculate_final_metrics(self, timestep_mses: List[float], timestep_maes: List[float], total_invalid_preds: int, action_success: List[int]) -> Dict[str, Any]:
        """Calculate comprehensive final metrics."""
        result = {}

        # Calculate MSE metrics
        total_dataset_mse = sum(timestep_mses)
        num_timesteps = len(timestep_mses)
        avg_dataset_mse = total_dataset_mse / num_timesteps if num_timesteps > 0 else 0.0

        # Calculate normalized MSE
        if num_timesteps > 1:
            normalized_mses = min_max_normalize(timestep_mses)
            normalized_amse = calculate_mean(normalized_mses)
        else:
            normalized_amse = np.nan

        # Calculate MAE metrics
        total_dataset_mae = sum(timestep_maes)
        avg_dataset_mae = calculate_mean(timestep_maes)

        if num_timesteps > 1:
            normalized_maes = min_max_normalize(timestep_maes)
            normalized_amae = calculate_mean(normalized_maes)

            # Calculate quantile filtered MAE metrics
            quantile_filtered_maes = quantile_filter(timestep_maes)
            if len(quantile_filtered_maes) > 1:
                normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
                normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
            else:
                normalized_quantile_filtered_amae = np.nan

            # Calculate additional MAE metrics
            max_rel_mae = calculate_max_relative_mae(timestep_maes)
            prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
        else:
            normalized_amae = np.nan
            normalized_quantile_filtered_amae = np.nan
            max_rel_mae = np.nan
            prop_beyond_threshold_mae = np.nan

        action_success_rate = 0.0
        if len(action_success) > 0:
            action_success_rate = (sum(action_success) / len(action_success)) * 100

        # Calculate invalid prediction percentage
        invalid_percentage = (total_invalid_preds / num_timesteps * 100) if num_timesteps > 0 else 0.0

        result.update({
            'action_success_rate': action_success_rate,
            'total_dataset_amse': total_dataset_mse,
            'total_dataset_amae': total_dataset_mae,
            'num_timesteps': num_timesteps,
            'avg_dataset_amse': avg_dataset_mse,
            'avg_dataset_amae': avg_dataset_mae,
            'normalized_amse': normalized_amse,
            'normalized_amae': normalized_amae,
            'normalized_quantile_filtered_amae': normalized_quantile_filtered_amae,
            'max_relative_mae': max_rel_mae,
            'proportion_beyond_threshold_mae': prop_beyond_threshold_mae,
            'total_invalid_preds': total_invalid_preds,
            'invalid_percentage': invalid_percentage,
        })

        return result
