from typing import Dict, Any, List, Union
import numpy as np
import re


def _validate_choice_output(output: Any, num_choices: int = 2) -> bool:
    """
    Validate that output is a valid choice for multiple choice questions.
    
    Args:
        output: The model output to validate
        num_choices: Number of valid choices (e.g., 2 for binary choice)
        
    Returns:
        True if output is a valid choice, False otherwise
    """
    if output is None:
        return False
    
    # Handle string outputs that might contain the choice
    if isinstance(output, str):
        # Extract numbers from string (e.g., "0", "1", "choice 0", "answer: 1")
        numbers = re.findall(r'\d+', output.strip())
        
        # Extract the first number
        if numbers:
            try:
                return 0 <= int(numbers[0]) < num_choices
            except Exception:
                return False
        return False
    
    # Handle integer outputs
    if isinstance(output, (int, np.integer)):
        return 0 <= int(output) < num_choices
    
    # Handle float outputs (round to nearest integer)
    if isinstance(output, (float, np.floating)):
        # Disallow nan/inf
        if not np.isfinite(output):
            return False
        
        try:
            rounded = int(np.round(output))
            return 0 <= rounded < num_choices
        except Exception:
            return False
    
    return False


def _extract_choice(output: Any, num_choices: int = 2) -> int:
    """
    Extract choice index from model output.
    
    Args:
        output: The model output
        num_choices: Number of valid choices
        
    Returns:
        Choice index (0 to num_choices-1) or -1 if invalid
    """
    if not _validate_choice_output(output, num_choices):
        return -1
    
    # Handle string outputs
    if isinstance(output, str):
        numbers = re.findall(r'\d+', output.strip())
        if numbers:
            return int(numbers[0])
        return -1
    
    # Handle numeric outputs
    if isinstance(output, (int, np.integer)):
        return int(output)
    
    if isinstance(output, (float, np.floating)):
        return int(np.round(output))
    
    return -1


class MCQMetricsCalculator:
    """
    Calculator for Multiple Choice Question (MCQ) metrics.
    
    Designed for tasks like PIQA where:
    - Each question has a fixed number of choices (e.g., 2 for binary choice)
    - The choices are question-specific (not consistent classes across dataset)
    - The goal is simply to select the correct choice for each question
    """
    
    def __init__(self, num_choices: int = 2):
        """
        Initialize MCQ metrics calculator.
        
        Args:
            num_choices: Number of choices per question (default 2 for binary choice like PIQA)
        """
        self.num_choices = num_choices
    
    def calculate_metrics(
        self, 
        predictions: List[Union[int, str]], 
        ground_truth_choices: List[int]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for MCQ predictions.
        
        Args:
            predictions: List of model predictions (choice indices)
            ground_truth_choices: List of ground truth choice indices
            
        Returns:
            Dictionary containing:
            - overall_accuracy: Accuracy including invalid predictions as wrong
            - valid_accuracy: Accuracy only on valid predictions
            - total_samples: Total number of samples
            - valid_predictions: Number of valid predictions
            - total_invalid_preds: Number of invalid predictions
            - invalid_percentage: Percentage of invalid predictions
        """
        if len(predictions) != len(ground_truth_choices):
            raise ValueError(f"Number of predictions ({len(predictions)}) must match "
                            f"number of ground truth choices ({len(ground_truth_choices)})")
        
        predicted_choices = []
        total_invalid_preds = 0
        
        for pred in predictions:
            choice = _extract_choice(pred, self.num_choices)
            predicted_choices.append(choice)
            if choice == -1:
                total_invalid_preds += 1
        
        predicted_choices = np.array(predicted_choices)
        ground_truth_choices = np.array(ground_truth_choices)
        
        return self._calculate_final_metrics(
            predicted_choices, 
            ground_truth_choices, 
            total_invalid_preds
        )
    
    def _calculate_final_metrics(
        self, 
        predicted_choices: np.ndarray, 
        ground_truth_choices: np.ndarray,
        total_invalid_preds: int
    ) -> Dict[str, Any]:
        """Calculate final metrics for MCQ evaluation."""
        result = {}
        
        total_samples = len(predicted_choices)
        
        correct_predictions = np.sum(predicted_choices == ground_truth_choices)
        overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Accuracy on valid predictions only
        valid_mask = predicted_choices != -1
        num_valid = np.sum(valid_mask)
        
        if num_valid > 0:
            valid_correct = np.sum(
                (predicted_choices[valid_mask] == ground_truth_choices[valid_mask])
            )
            valid_accuracy = valid_correct / num_valid
        else:
            valid_accuracy = 0.0
        
        # Invalid prediction percentage
        invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
        
        # Distribution of predictions
        choice_distribution = {}
        for i in range(self.num_choices):
            choice_distribution[f'choice_{i}_count'] = int(np.sum(predicted_choices == i))
        
        result.update({
            # Basic accuracy metrics
            'overall_accuracy': float(overall_accuracy),
            'valid_accuracy': float(valid_accuracy),
            
            # Sample counts
            'total_samples': int(total_samples),
            'valid_predictions': int(num_valid),
            'total_invalid_preds': int(total_invalid_preds),
            'invalid_percentage': float(invalid_percentage),
            
            # Distribution info
            'num_choices': self.num_choices,
            'choice_distribution': choice_distribution,
        })
        
        return result

