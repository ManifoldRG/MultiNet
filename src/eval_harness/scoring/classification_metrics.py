from typing import Dict, Any, List, Union
import numpy as np
import re
from collections import Counter

from src.eval_utils import (
    get_exact_match_rate,
    calculate_tp_fp_fn_counts,
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    get_precision_per_class,
    get_recall_per_class,
    get_f1_per_class,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)


def _validate_class_output(output: Any, num_classes: int = 2) -> bool:
    """
    Validate that output is a valid class index (adapter should have extracted).
    
    Args:
        output: The extracted class index from adapter
        num_classes: Number of valid classes (default 2)
        
    Returns:
        True if output is a valid class index, False otherwise
    """
    if output is None:
        return False
    
    # Should be an integer in valid range
    if isinstance(output, (int, np.integer)):
        return 0 <= int(output) < num_classes
    
    # Adapter should have returned int or -1
    return False


class ClassificationMetricsCalculator:
    """
    Calculator for classification task metrics.
    
    Handles predictions for visual classification tasks like ODinW and multiple choice
    tasks like PIQA where models output discrete class indices. Uses utility functions 
    from eval_utils.py for consistency.
    """
    
    def __init__(self, num_classes: int = 2):
        """
        Initialize classification metrics calculator.
        
        Args:
            num_classes: Number of classes in the classification task (default 2 for binary)
        """
        self.num_classes = num_classes
        self.valid_labels = list(range(num_classes))
    
    def calculate_metrics(
        self, 
        predictions: List[Any], 
        ground_truth_classes: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for classification predictions.
        
        Args:
            predictions: List of model predictions (discrete class indices)
            ground_truth_classes: List of ground truth class indices
            
        Returns:
            Dictionary containing calculated metrics
        """
        if len(predictions) != len(ground_truth_classes):
            raise ValueError(f"Number of predictions ({len(predictions)}) must match "
                            f"number of ground truth classes ({len(ground_truth_classes)})")
        
        # Process predictions
        predicted_classes = []
        total_invalid_preds = 0
        
        for pred_dict in predictions:
            # Extract class from structured format
            pred = pred_dict["extracted_outputs"] if isinstance(pred_dict, dict) else pred_dict
            
            # Validate only - adapter should have done extraction
            if _validate_class_output(pred, self.num_classes):
                predicted_classes.append(int(pred))
            else:
                # Invalid - treat as -1 (will naturally be incorrect)
                predicted_classes.append(-1)
                total_invalid_preds += 1
        
        # Convert to numpy arrays for easier computation
        predicted_classes = np.array(predicted_classes)
        ground_truth_classes = np.array(ground_truth_classes)
        
        # Calculate metrics using eval_utils functions
        return self._calculate_final_metrics(
            predicted_classes, 
            ground_truth_classes, 
            total_invalid_preds
        )
    
    def _calculate_final_metrics(
        self, 
        predicted_classes: np.ndarray, 
        ground_truth_classes: np.ndarray,
        total_invalid_preds: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive final metrics for classification evaluation using eval_utils."""
        result = {}
        
        total_samples = len(predicted_classes)
        
        # Basic accuracy metrics
        overall_accuracy = get_exact_match_rate(predicted_classes, ground_truth_classes)
        
        # Accuracy on valid predictions only
        valid_predictions = predicted_classes != -1
        if np.any(valid_predictions):
            valid_accuracy = get_exact_match_rate(
                predicted_classes[valid_predictions], 
                ground_truth_classes[valid_predictions]
            )
        else:
            valid_accuracy = 0.0
        
        # Calculate TP, FP, FN counts using eval_utils
        total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
            predicted_classes, ground_truth_classes, self.valid_labels
        )
        
        # Micro metrics using eval_utils
        micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
        micro_precision_without_invalid = get_micro_precision_from_counts(total_tp, valid_fp)
        micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
        micro_f1 = get_micro_f1(micro_precision, micro_recall)
        micro_f1_without_invalid = get_micro_f1(micro_precision_without_invalid, micro_recall)
        
        # Per-class metrics using eval_utils
        class_precisions = get_precision_per_class(predicted_classes, ground_truth_classes, self.valid_labels)
        class_recalls = get_recall_per_class(predicted_classes, ground_truth_classes, self.valid_labels)
        class_f1s = get_f1_per_class(class_precisions, class_recalls)
        
        # Macro metrics using eval_utils
        macro_precision = get_macro_precision(class_precisions)
        macro_recall = get_macro_recall(class_recalls)
        macro_f1 = get_macro_f1(class_f1s)

        # Invalid prediction percentage
        invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
        
        result.update({
            # Basic accuracy metrics
            'overall_accuracy': overall_accuracy,
            'valid_accuracy': valid_accuracy,
            
            # Per-class metrics
            'precision_per_class': class_precisions,
            'recall_per_class': class_recalls,
            'f1_per_class': class_f1s,
            
            # Macro averages
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            
            # Micro averages
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            
            # Invalid predictions
            'total_samples': total_samples,
            'total_invalid_preds': total_invalid_preds,
            'invalid_percentage': invalid_percentage,
            'valid_predictions': int(np.sum(valid_predictions)),
            
            # Additional detailed metrics from eval_utils
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'valid_fp': valid_fp,
            'invalid_fp': invalid_fp,
        })
        
        return result

