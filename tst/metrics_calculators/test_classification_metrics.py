import sys
import os
import math

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
import pytest

from src.eval_harness.scoring.classification_metrics import (
    _validate_class_output,
    ClassificationMetricsCalculator,
)


def test_validate_class_output():
    """Test class validation - adapter should have already extracted to int."""
    # Valid integer classes (adapter already extracted)
    assert _validate_class_output(0, 2)
    assert _validate_class_output(1, 2)
    assert not _validate_class_output(2, 2)  # Out of range
    assert not _validate_class_output(-1, 2)  # Adapter returns -1 for invalid
    
    # Adapter should return int, not string (these should fail validation)
    assert not _validate_class_output("0", 2)
    assert not _validate_class_output("1", 2)
    assert not _validate_class_output("class 0", 2)
    assert not _validate_class_output("prediction: 1", 2)
    
    # Adapter should return int, not float (these should fail validation)
    assert not _validate_class_output(0.0, 2)
    assert not _validate_class_output(1.0, 2)
    assert not _validate_class_output(0.7, 2)
    
    # Invalid inputs
    assert not _validate_class_output(None, 2)
    assert not _validate_class_output("", 2)
    assert not _validate_class_output([0, 1], 2)


def test_classification_metrics_calculator_discrete_predictions():
    """Test classification metrics calculator with discrete class predictions."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # Test case 1: Perfect predictions
    predictions = [
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 1", "extracted_outputs": 1},
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 1", "extracted_outputs": 1},
        {"raw_output": "Class 0", "extracted_outputs": 0},
    ]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert math.isclose(metrics['valid_accuracy'], 1.0)
    assert math.isclose(metrics['macro_precision'], 1.0)
    assert math.isclose(metrics['macro_recall'], 1.0)
    assert math.isclose(metrics['macro_f1'], 1.0)
    assert metrics['total_invalid_preds'] == 0
    
    # Test case 2: Mixed predictions with some invalid
    predictions = [
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 1", "extracted_outputs": 1},
        {"raw_output": "invalid", "extracted_outputs": "invalid"},
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 2", "extracted_outputs": 2},  # 2 is out of range
    ]
    ground_truth = [0, 1, 1, 0, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # 3 out of 5 correct (including invalid as wrong)
    assert math.isclose(metrics['overall_accuracy'], 3/5)
    # 3 out of 3 valid predictions correct
    assert math.isclose(metrics['valid_accuracy'], 1.0)
    assert metrics['total_invalid_preds'] == 2
    
    # Test case 3: Mixed valid and invalid predictions
    predictions = [
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 1", "extracted_outputs": 1},
        {"raw_output": "invalid", "extracted_outputs": "invalid"},  # Invalid string
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 2", "extracted_outputs": 2},  # Out of range
    ]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # 2 out of 5 correct (including invalid as wrong)
    assert math.isclose(metrics['overall_accuracy'], 2/5)
    assert metrics['total_invalid_preds'] == 2


def test_classification_metrics_calculator_per_class_metrics():
    """Test per-class precision, recall, and F1 metrics."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # Test case: Imbalanced predictions
    predictions = [
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 1", "extracted_outputs": 1},
        {"raw_output": "Class 1", "extracted_outputs": 1},
    ]  # 3 class 0, 2 class 1
    ground_truth = [0, 0, 1, 1, 1]  # 2 class 0, 3 class 1
    
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # Class 0: 2 TP, 1 FP, 0 FN -> P=2/3, R=2/2, F1=4/5
    assert math.isclose(metrics['precision_per_class'][0], 2/3)
    assert math.isclose(metrics['recall_per_class'][0], 1.0)
    assert math.isclose(metrics['f1_per_class'][0], 4/5)
    
    # Class 1: 2 TP, 0 FP, 1 FN -> P=1.0, R=2/3, F1=4/5
    assert math.isclose(metrics['precision_per_class'][1], 1.0)
    assert math.isclose(metrics['recall_per_class'][1], 2/3)
    assert math.isclose(metrics['f1_per_class'][1], 4/5)


def test_classification_metrics_calculator_edge_cases():
    """Test edge cases and error handling."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # All invalid predictions
    predictions = [
        {"raw_output": "invalid", "extracted_outputs": "invalid"},
        {"raw_output": "", "extracted_outputs": None},
        {"raw_output": "bad", "extracted_outputs": "bad"},
        {"raw_output": "2", "extracted_outputs": 2},
        {"raw_output": "-1", "extracted_outputs": -1},
    ]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    assert metrics['total_invalid_preds'] == 5
    assert metrics['overall_accuracy'] == 0.0
    assert metrics['invalid_percentage'] == 100.0
    
    # Mismatched lengths should raise error
    with pytest.raises(ValueError):
        calc.calculate_metrics([
            {"raw_output": "0", "extracted_outputs": 0},
            {"raw_output": "1", "extracted_outputs": 1},
        ], [0, 1, 0])


def test_classification_metrics_calculator_different_num_classes():
    """Test with different numbers of classes."""
    # Test with 3 classes
    calc_3 = ClassificationMetricsCalculator(num_classes=3)
    
    predictions = [
        {"raw_output": "Class 0", "extracted_outputs": 0},
        {"raw_output": "Class 1", "extracted_outputs": 1},
        {"raw_output": "Class 2", "extracted_outputs": 2},
        {"raw_output": "Class 0", "extracted_outputs": 0},
    ]
    ground_truth = [0, 1, 2, 0]
    metrics = calc_3.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert len(metrics['precision_per_class']) == 3
    assert len(metrics['recall_per_class']) == 3
    assert len(metrics['f1_per_class']) == 3


def test_classification_metrics_calculator_float_rejection():
    """Test that float inputs are rejected (adapter should extract to int)."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # Test float inputs should be rejected
    predictions = [
        {"raw_output": "0.0", "extracted_outputs": 0.0},
        {"raw_output": "1.0", "extracted_outputs": 1.0},
        {"raw_output": "0.1", "extracted_outputs": 0.1},
        {"raw_output": "0.9", "extracted_outputs": 0.9},
    ]  # All should be invalid - adapter should extract to int
    ground_truth = [0, 1, 0, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)

    # All should be invalid since floats are not accepted
    assert metrics['overall_accuracy'] == 0.0
    assert metrics['total_invalid_preds'] == 4

if __name__ == "__main__":
    pytest.main(['-s', __file__])

