
from typing import Dict, Any, List
import numpy as np
from src.eval_utils import (
    calculate_mse, calculate_mae,
    calculate_mean, min_max_normalize, quantile_filter,
    calculate_max_relative_mae, calculate_proportion_beyond_mae_threshold
)

def _validate_output(output: Any, shape: tuple[int]) -> bool:
    """Validate that output is a valid action vector."""

    # TODO: correctly handle other types of errors, breaking this out into different functions for 
    # different models if necessary
    if output is None or not isinstance(output, (list, tuple, np.ndarray)) or len(output) != shape[0]:
        return False

    # check each element of output is a float or float-like
    # TODO: do we allow outputs that are partially correctly formatted?
    for item in output:
        if not isinstance(item, (float, int, np.floating, int, np.integer)):
            return False
        
        # TODO: do we allow nan/inf?
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
            predictions: List of model predictions
            ground_truth_actions: List of ground truth action vectors
            
        Returns:
            Dictionary containing calculated metrics
        """
        mses, maes = [], []
        total_invalid_preds = 0
        
        for i, pred in enumerate(predictions):
            if _validate_output(pred, shape=ground_truth_actions[i].shape):
                pred = [float(item) for item in pred]
                mses.append(calculate_mse(pred, ground_truth_actions[i]))
                maes.append(calculate_mae(pred, ground_truth_actions[i]))
            else:
                # max value of MSE/MAE for invalid outputs
                max_vals = np.array(self.action_stats['max'])
                min_vals = np.array(self.action_stats['min'])
                mse = calculate_mse(max_vals, min_vals)
                mae = calculate_mae(max_vals, min_vals)
                mses.append(mse)
                maes.append(mae)
                total_invalid_preds += 1
        
        return self._calculate_final_metrics(mses, maes, total_invalid_preds)
    
    def _calculate_final_metrics(self, timestep_mses: List[float], timestep_maes: List[float], total_invalid_preds: int) -> Dict[str, Any]:
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
            normalized_amse = 0.0
        
        # Calculate MAE metrics
        total_dataset_mae = sum(timestep_maes)
        avg_dataset_mae = calculate_mean(timestep_maes)
        
        if num_timesteps > 1:
            normalized_maes = min_max_normalize(timestep_maes)
            normalized_amae = calculate_mean(normalized_maes)
            
            # Calculate quantile filtered MAE metrics
            quantile_filtered_maes = quantile_filter(timestep_maes)
            normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
            normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
            
            # Calculate additional MAE metrics
            max_rel_mae = calculate_max_relative_mae(timestep_maes)
            prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
        else:
            normalized_amae = 0.0
            normalized_quantile_filtered_amae = 0.0
            max_rel_mae = 0.0
            prop_beyond_threshold_mae = 0.0
        
        # Calculate invalid prediction percentage
        invalid_percentage = (total_invalid_preds / num_timesteps * 100) if num_timesteps > 0 else 0.0
        
        result.update({
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
