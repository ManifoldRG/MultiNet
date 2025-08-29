"""
Gameplay Metrics Calculator for MultiNet Evaluation Harness

This module implements scoring logic for gameplay datasets with discrete action spaces.
Used for: OvercookedAI

Metrics calculated:
- Brier MAE (Mean Absolute Error)
- Brier MSE (Mean Squared Error)
- Normalized Brier MAE (Normalized Mean Absolute Error)
- Normalized Brier MSE (Normalized Mean Squared Error)
- Normalized Quantile Filtered Brier MAE (Normalized Mean Absolute Error)
- Max Relative MAE (Maximum Relative MAE)
- Proportion Beyond MAE Threshold (Proportion of Predictions Beyond MAE Threshold)
- Micro Precision/Recall/F1
- Macro Precision/Recall/F1 
- Invalid Percentage

The metrics are calculated based on predicted action probabilities vs ground truth actions.
"""

from typing import Dict, List, Any, Sequence, Optional, Tuple, Union
import numpy as np

# Import existing evaluation utilities
from src.eval_utils import (
    calculate_brier_mae, calculate_brier_mse,
    calculate_tp_fp_fn_counts, get_micro_precision_from_counts, 
    get_micro_recall_from_counts, get_micro_f1,
    get_precision_per_class, get_recall_per_class, get_f1_per_class,
    get_macro_precision, get_macro_recall, get_macro_f1,
    calculate_mean, min_max_normalize, quantile_filter,
    calculate_max_relative_mae, calculate_proportion_beyond_mae_threshold
)

INVALID_ACTION = -1

class GameplayMetricsCalculator:
    """
    Calculator for gameplay metrics used in discrete action space environments.
    
    This class processes model predictions and ground truth actions to compute
    comprehensive evaluation metrics for gameplay datasets.
    
    Attributes:
        num_actions (int): Size of the discrete action space
        max_brier_mae_error (float): Maximum possible Brier MAE error for invalid predictions
        max_brier_mse_error (float): Maximum possible Brier MSE error for invalid predictions
        noop_action (int): Default no-operation action index
    """
    
    def __init__(
        self, 
        num_actions: int,
        max_brier_mae_error: float = 2.0,
        max_brier_mse_error: float = 2.0,
        noop_action: Optional[int] = None
    ):
        """
        Initialize the GameplayMetricsCalculator.
        
        Args:
            num_actions: Size of the discrete action space
            max_brier_mae_error: Maximum Brier MAE penalty for invalid predictions
            max_brier_mse_error: Maximum Brier MSE penalty for invalid predictions  
            noop_action: Default action index for invalid predictions
        """
        self.num_actions = num_actions
        self.max_brier_mae_error = max_brier_mae_error
        self.max_brier_mse_error = max_brier_mse_error
        self.noop_action = noop_action
    
    def is_valid_prediction_format(
        self, 
        prediction: Any, 
        with_probabilities: bool = True
    ) -> bool:
        
        if with_probabilities:
            # Expected format: list of floats
            return self._is_valid_probability_prediction_format(prediction)

        # Expected format: single action index
        return self._is_valid_action_prediction_format(prediction)

    def _is_valid_action_prediction_format(self, prediction: Any) -> bool:
        if isinstance(prediction, (int, np.integer)):
            return 0 <= prediction < self.num_actions
        
        return False

    def _is_valid_probability_prediction_format(self, prediction: List[float]) -> bool:
        if not isinstance(prediction, (np.ndarray, list)) or not all([isinstance(d, (float, np.floating)) for d in prediction]):
            return False
        
        return len(prediction) == self.num_actions and all([0.0 <= d <= 1.0 for d in prediction]) and abs(sum(prediction) - 1.0) < 1e-5

    def process_prediction(
        self, 
        prediction: Any, 
        with_probabilities: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Process a single model prediction into action probabilities and predicted action.
        
        Args:
            prediction: Raw model prediction
            with_probabilities: Whether prediction includes probabilities
            
        Returns:
            Tuple of (action_probabilities, predicted_action_index)
            Returns (None, None) if prediction is invalid
        """
        if not self.is_valid_prediction_format(prediction, with_probabilities):
            return None, None
        
        if with_probabilities:
            return self._process_prediction_probabilities(prediction)
        else:
            return self._process_prediction_action(prediction)
    
    def _process_prediction_probabilities(self, prediction: List[float]) -> Tuple[np.ndarray, int]:
        """
        Process a single model prediction into action probabilities and predicted action.
        """
        probs = np.array(prediction)
        predicted_action = np.argmax(probs)
        return probs, predicted_action
    
    def _process_prediction_action(self, prediction: int) -> Tuple[np.ndarray, int]:
        """
        Process a single model prediction into action probabilities and predicted action.
        """
        # Create makeshift probability list
        probs = np.zeros(self.num_actions)
        probs[int(prediction)] = 1.0
        
        return probs, int(prediction)
    
    def calculate_metrics(
        self,
        predictions: Union[List[List[float]], List[int]],
        ground_truth_actions: List[int],
        with_probabilities: bool = True
    ) -> Dict[str, float]:
        """
        Calculate gameplay metrics for a batch of predictions.
        
        Args:
            predictions: List of model predictions, where each element is a list of probabilities 
                if with_probabilities is True, or an integer in the range [0, num_actions - 1] if with_probabilities is False
            ground_truth_actions: List of ground truth action indices
            with_probabilities: Whether predictions include probabilities
            
        Returns:
            Dictionary containing calculated metrics
        """
        if len(predictions) != len(ground_truth_actions):
            raise ValueError(f"Predictions length ({len(predictions)}) != ground truth length ({len(ground_truth_actions)})")
        
        brier_maes = []
        brier_mses = []
        predicted_actions = []
        # Process each prediction
        for pred, gt_action in zip(predictions, ground_truth_actions):            
            # Validate and clamp ground truth action to valid range
            if not 0 <= gt_action < self.num_actions:
                gt_action = self.noop_action if self.noop_action < self.num_actions else 0
            
            # Create one-hot encoding for ground truth
            one_hot_gt = np.zeros(self.num_actions)
            one_hot_gt[gt_action] = 1.0
            
            # Process prediction
            pred_probs, pred_action = self.process_prediction(pred, with_probabilities)
            
            if pred_probs is not None and pred_action is not None:
                # Valid prediction
                mae = calculate_brier_mae(pred_probs, one_hot_gt)
                mse = calculate_brier_mse(pred_probs, one_hot_gt)
                
                brier_maes.append(mae)
                brier_mses.append(mse)
                predicted_actions.append(pred_action)
            else:
                # Invalid prediction - assign maximum error
                brier_maes.append(self.max_brier_mae_error)
                brier_mses.append(self.max_brier_mse_error)
                predicted_actions.append(INVALID_ACTION)
        
        # Check if all predictions are invalid
        if all(pred == INVALID_ACTION for pred in predicted_actions):
            print("WARNING: All predictions are invalid. "
                  "Ensure that prediction is intended to be a list of 6 probabilities if with_probabilities is True, "
                  "or an integer in the range [0, 5] if with_probabilities is False.")

        if not isinstance(predicted_actions, np.ndarray):
            predicted_actions = np.array(predicted_actions)
        if not isinstance(ground_truth_actions, np.ndarray):
            ground_truth_actions = np.array(ground_truth_actions)

        # Calculate final metrics
        return self._calculate_final_metrics(
            brier_mses, brier_maes, predicted_actions, ground_truth_actions
        )
    
    def _calculate_final_metrics(
        self,
        timestep_mses: List[float],
        timestep_maes: List[float], 
        predicted_actions: np.ndarray,
        ground_truth_actions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from individual timestep results.
        
        Args:
            timestep_mses: Brier MSE values for each timestep
            timestep_maes: Brier MAE values for each timestep
            predicted_actions: Predicted action indices
            ground_truth_actions: Ground truth action indices
            total_invalid_preds: Number of invalid predictions
            
        Returns:
            Dictionary with all calculated metrics
        """
        result = {}
        
        # Basic aggregations
        num_timesteps = len(timestep_maes)
        total_dataset_amae = sum(timestep_maes)
        total_dataset_amse = sum(timestep_mses)
        
        # Calculate average metrics
        avg_dataset_amae = calculate_mean(timestep_maes)
        avg_dataset_amse = calculate_mean(timestep_mses)
        
        # Calculate normalized metrics
        normalized_maes = min_max_normalize(timestep_maes)
        normalized_mses = min_max_normalize(timestep_mses)
        normalized_amae = calculate_mean(normalized_maes)
        normalized_amse = calculate_mean(normalized_mses)
        
        # Calculate quantile filtered metrics
        quantile_filtered_maes = quantile_filter(timestep_maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
        normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
        
        # Calculate additional MAE metrics
        max_rel_mae = calculate_max_relative_mae(timestep_maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
        
        # Get TP, FP, FN, valid FP, and invalid FP counts
        possible_actions = list(range(self.num_actions))
        tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
            predicted_actions, ground_truth_actions, possible_actions
        )
        
        # Micro metrics (averaged across all samples)
        micro_precision = get_micro_precision_from_counts(tp, fp)
        micro_precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
        micro_recall = get_micro_recall_from_counts(tp, fn)
        micro_f1 = get_micro_f1(micro_precision, micro_recall)
        micro_f1_without_invalid = get_micro_f1(micro_precision_without_invalid, micro_recall)
        
        # Macro metrics (averaged across classes)
        class_precisions = get_precision_per_class(predicted_actions, ground_truth_actions, possible_actions)
        class_recalls = get_recall_per_class(predicted_actions, ground_truth_actions, possible_actions)
        class_f1s = get_f1_per_class(class_precisions, class_recalls)
        
        macro_precision = get_macro_precision(class_precisions)
        macro_recall = get_macro_recall(class_recalls)
        macro_f1 = get_macro_f1(class_f1s)
        
        # Invalid prediction metrics
        percentage_invalids = ((invalid_fp / num_timesteps) * 100) if num_timesteps > 0 else 0.0
        
        # Compile results
        result.update({
            # Basic metrics
            'total_dataset_amse': total_dataset_amse,
            'total_dataset_amae': total_dataset_amae,
            'num_timesteps': num_timesteps,
            'avg_dataset_amse': avg_dataset_amse,
            'avg_dataset_amae': avg_dataset_amae,
            
            # Normalized metrics
            'normalized_amse': normalized_amse,
            'normalized_amae': normalized_amae,
            'normalized_quantile_filtered_amae': normalized_quantile_filtered_amae,
            
            # Additional MAE metrics
            'max_relative_mae': max_rel_mae,
            'proportion_beyond_threshold_mae': prop_beyond_threshold_mae,
            
            # Micro metrics 
            'micro_precision': micro_precision,
            'micro_precision_without_invalid': micro_precision_without_invalid,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'micro_f1_without_invalid': micro_f1_without_invalid,
            
            # Macro metrics
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            
            # Per-class metrics
            'class_precisions': class_precisions,
            'class_recalls': class_recalls,
            'class_f1s': class_f1s,
            
            # Invalid prediction metrics
            'total_invalid_preds': int(invalid_fp),
            'percentage_invalids': percentage_invalids,
            
            # Raw predictions for analysis
            'predicted_actions': [int(action) for action in predicted_actions],
            'ground_truth_actions': [int(action) for action in ground_truth_actions],
        })
        
        return result

# Dataset-specific calculator classes for convenience and clarity
class OvercookedAIMetricsCalculator(GameplayMetricsCalculator):
    """Metrics calculator specifically for OvercookedAI dataset."""
    
    def __init__(self):
        # Actions: up, down, left, right, noop, and "interact"
        super().__init__(num_actions=6, noop_action=4)
        self.dataset_name = "overcooked_ai"