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
    calculate_max_relative_mae, calculate_proportion_beyond_mae_threshold,
    get_exact_match_rate
)

# Import OvercookedAI-specific definitions
from definitions.overcooked import OverCookedDefinitions

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
        noop_action: Optional[int] = None,
        with_probabilities: bool = False
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
        self.with_probabilities = with_probabilities
    
    def is_valid_prediction_format(
        self, 
        prediction: Any, 
    ) -> bool:
        
        if self.with_probabilities:
            return self._is_valid_probability_prediction_format(prediction)
        else:
            return self._is_valid_action_prediction_format(prediction)

    def _is_valid_action_prediction_format(self, prediction: Any) -> bool:
        if isinstance(prediction, (int, np.integer)):
            return 0 <= prediction < self.num_actions
        
        return False

    def _is_valid_probability_prediction_format(self, prediction: List[float]) -> bool:
        if not isinstance(prediction, (np.ndarray, list)):
            return False
        
        # Check if all elements are numeric (int, float, or numpy numeric types)
        try:
            numeric_pred = [float(d) for d in prediction]
        except (ValueError, TypeError):
            return False
        
        return len(numeric_pred) == self.num_actions and all([0.0 <= d <= 1.0 for d in numeric_pred]) and abs(sum(numeric_pred) - 1.0) < 1e-5

    def process_prediction(
        self, 
        prediction: Any, 
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Process a single model prediction into action probabilities and predicted action.
        
        Args:
            prediction: Raw model prediction
            
        Returns:
            Tuple of (action_probabilities, predicted_action_index)
            Returns (None, None) if prediction is invalid
        """
        if not self.is_valid_prediction_format(prediction):
            return None, None
        
        if self.with_probabilities:
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
        predictions: List[Dict[str, Any]],
        ground_truth_actions: List[int],
    ) -> Dict[str, float]:
        """
        Calculate gameplay metrics for a batch of predictions.
        
        Args:
            predictions: List of structured predictions with "extracted_outputs"
                        If self.with_probabilities=True, extracted_outputs should be List[float] (probabilities)
                        If self.with_probabilities=False, extracted_outputs should be int (action index)
            ground_truth_actions: List of ground truth action indices
            
        Returns:
            Dictionary containing calculated metrics
        """
        if len(predictions) != len(ground_truth_actions):
            raise ValueError(f"Predictions length ({len(predictions)}) != ground truth length ({len(ground_truth_actions)})")
        
        # Extract predictions from structured format
        extracted_preds = []
        for pred_dict in predictions:
            extracted_preds.append(pred_dict["extracted_outputs"])
        
        # Use extracted predictions for the rest of the calculation
        predictions = extracted_preds
        
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
            pred_probs, pred_action = self.process_prediction(pred)
            
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
            print("WARNING: All predictions are invalid. ")

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
        if len(quantile_filtered_maes) > 1:
            normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
            normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
        else:
            normalized_quantile_filtered_amae = np.nan
        
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
        
        # Exact match rate
        exact_match = get_exact_match_rate(predicted_actions, ground_truth_actions)
        
        # Compile results
        result.update({
            'exact_match': exact_match,
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
    """
    Metrics calculator specifically for OvercookedAI dataset.
    
    OvercookedAI is a two-player cooperative game with:
    - Joint action space: 36 actions (6 player0 actions × 6 player1 actions)
    - Individual action space: 6 actions per player (up, down, left, right, noop, interact)
    - NOOP_ACTION: 28 (both players taking noop: 4,4 -> index 28)
    
    This calculator provides both joint metrics and per-player metrics.
    """
    
    def __init__(self, with_probabilities: bool = False):
        # Joint action space: 6 player0 actions × 6 player1 actions = 36 total
        # NOOP_ACTION is 28 (both players taking noop: 4,4 -> index 28)
        super().__init__(num_actions=36, noop_action=28, with_probabilities=with_probabilities)
        self.dataset_name = "overcooked_ai"
        self.individual_action_space_size = 6
    
    def _get_individual_player_labels(self, joint_action_idx: int) -> Tuple[int, int]:
        """
        Extract individual player action indices from joint action.
        
        Args:
            joint_action_idx: Joint action index (0-35)
            
        Returns:
            Tuple of (player0_action_idx, player1_action_idx) where each is in range [0, 5]
        """
        individual_action_space = OverCookedDefinitions.INDIVIDUAL_ACTION_SPACE
        discrete_to_joint = OverCookedDefinitions.PLAYER_ACTION_SPACE_TUPLES
        
        player0_action, player1_action = discrete_to_joint[joint_action_idx]
        player0_label = individual_action_space[player0_action]
        player1_label = individual_action_space[player1_action]
        
        return player0_label, player1_label
    
    def _calculate_individual_player_metrics(
        self, 
        joint_probs: np.ndarray, 
        player0_label: int, 
        player1_label: int
    ) -> Tuple[float, float, int, float, float, int]:
        """
        Calculate per-player metrics from joint action probabilities.
        
        Marginalizes joint probabilities to individual player distributions,
        then calculates MAE, MSE, and predicted action for each player.
        
        Args:
            joint_probs: Probability distribution over 36 joint actions
            player0_label: Ground truth action index for player0 (0-5)
            player1_label: Ground truth action index for player1 (0-5)
            
        Returns:
            Tuple of (p0_mae, p0_mse, p0_pred, p1_mae, p1_mse, p1_pred)
        """
        individual_action_space = OverCookedDefinitions.INDIVIDUAL_ACTION_SPACE
        discrete_to_joint = OverCookedDefinitions.PLAYER_ACTION_SPACE_TUPLES
        
        # Convert joint discrete action probs to player0 and player1 discrete action probs
        player0_probs = [0.0] * self.individual_action_space_size
        player1_probs = [0.0] * self.individual_action_space_size
        
        for action_idx, prob in enumerate(joint_probs):
            player0_action, player1_action = discrete_to_joint[action_idx]
            player0_probs[individual_action_space[player0_action]] += prob
            player1_probs[individual_action_space[player1_action]] += prob
        
        # Get predicted actions
        player0_pred = np.argmax(player0_probs)
        player1_pred = np.argmax(player1_probs)
        
        # Create one-hot labels
        player0_one_hot_label = [0.0] * self.individual_action_space_size
        player1_one_hot_label = [0.0] * self.individual_action_space_size
        player0_one_hot_label[player0_label] = 1.0
        player1_one_hot_label[player1_label] = 1.0
        
        # Calculate metrics
        player0_mae = calculate_brier_mae(player0_probs, player0_one_hot_label)
        player1_mae = calculate_brier_mae(player1_probs, player1_one_hot_label)
        player0_mse = calculate_brier_mse(player0_probs, player0_one_hot_label)
        player1_mse = calculate_brier_mse(player1_probs, player1_one_hot_label)
        
        return player0_mae, player0_mse, player0_pred, player1_mae, player1_mse, player1_pred
    
    def calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_actions: List[int],
    ) -> Dict[str, float]:
        """
        Calculate gameplay metrics for OvercookedAI, including per-player metrics.
        
        Args:
            predictions: List of structured predictions with "extracted_outputs"
                        If self.with_probabilities=True, extracted_outputs should be List[float] (probabilities)
                        If self.with_probabilities=False, extracted_outputs should be int (action index)
            ground_truth_actions: List of ground truth joint action indices
            
        Returns:
            Dictionary containing joint metrics and per-player metrics
        """
        if len(predictions) != len(ground_truth_actions):
            raise ValueError(f"Predictions length ({len(predictions)}) != ground truth length ({len(ground_truth_actions)})")
        
        # Extract predictions from structured format
        extracted_preds = []
        for pred_dict in predictions:
            extracted_preds.append(pred_dict["extracted_outputs"])
        
        # Use extracted predictions for the rest of the calculation
        predictions = extracted_preds
        
        # Track per-player metrics
        player0_maes = []
        player0_mses = []
        player0_preds = []
        player0_trues = []
        
        player1_maes = []
        player1_mses = []
        player1_preds = []
        player1_trues = []
        
        brier_maes = []
        brier_mses = []
        predicted_actions = []
        
        # Process each prediction
        for pred, gt_action in zip(predictions, ground_truth_actions):
            # Validate and clamp ground truth action to valid range
            if not 0 <= gt_action < self.num_actions:
                gt_action = self.noop_action if self.noop_action < self.num_actions else 0
            
            # Get individual player labels
            player0_label, player1_label = self._get_individual_player_labels(gt_action)
            
            # Create one-hot encoding for ground truth
            one_hot_gt = np.zeros(self.num_actions)
            one_hot_gt[gt_action] = 1.0
            
            # Process prediction
            pred_probs, pred_action = self.process_prediction(pred)
            
            if pred_probs is not None and pred_action is not None:
                # Valid prediction - calculate joint metrics
                mae = calculate_brier_mae(pred_probs, one_hot_gt)
                mse = calculate_brier_mse(pred_probs, one_hot_gt)
                
                brier_maes.append(mae)
                brier_mses.append(mse)
                predicted_actions.append(pred_action)
                
                # Calculate per-player metrics
                player0_mae, player0_mse, player0_pred, player1_mae, player1_mse, player1_pred = \
                    self._calculate_individual_player_metrics(pred_probs, player0_label, player1_label)
                
                player0_maes.append(player0_mae)
                player0_mses.append(player0_mse)
                player0_preds.append(player0_pred)
                
                player1_maes.append(player1_mae)
                player1_mses.append(player1_mse)
                player1_preds.append(player1_pred)
            else:
                # Invalid prediction - assign maximum error
                brier_maes.append(self.max_brier_mae_error)
                brier_mses.append(self.max_brier_mse_error)
                predicted_actions.append(INVALID_ACTION)
                
                player0_maes.append(self.max_brier_mae_error)
                player0_mses.append(self.max_brier_mse_error)
                player0_preds.append(INVALID_ACTION)
                
                player1_maes.append(self.max_brier_mae_error)
                player1_mses.append(self.max_brier_mse_error)
                player1_preds.append(INVALID_ACTION)
            
            player0_trues.append(player0_label)
            player1_trues.append(player1_label)
        
        # Convert to numpy arrays
        if not isinstance(predicted_actions, np.ndarray):
            predicted_actions = np.array(predicted_actions)
        if not isinstance(ground_truth_actions, np.ndarray):
            ground_truth_actions = np.array(ground_truth_actions)
        
        player0_preds = np.array(player0_preds)
        player0_trues = np.array(player0_trues)
        player1_preds = np.array(player1_preds)
        player1_trues = np.array(player1_trues)
        
        # Calculate joint metrics
        joint_metrics = self._calculate_final_metrics(
            brier_mses, brier_maes, predicted_actions, ground_truth_actions
        )
        
        # Calculate per-player metrics using a temporary calculator for individual action space
        temp_calculator = GameplayMetricsCalculator(
            num_actions=self.individual_action_space_size,
            noop_action=4  # Individual noop action
        )
        
        player0_metrics = temp_calculator._calculate_final_metrics(
            player0_mses, player0_maes, player0_preds, player0_trues
        )
        player1_metrics = temp_calculator._calculate_final_metrics(
            player1_mses, player1_maes, player1_preds, player1_trues
        )
        
        # Add per-player metrics to joint metrics
        joint_metrics['player0_results'] = player0_metrics
        joint_metrics['player1_results'] = player1_metrics
        
        return joint_metrics