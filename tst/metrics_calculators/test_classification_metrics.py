import sys
import os
import math

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
import pytest

from src.eval_harness.scoring.classification_metrics import (
    _validate_class_output,
    _extract_class_index,
    ClassificationMetricsCalculator,
    FLOAT_NUM_TOLERANCE,
)


def test_validate_class_output():
    """Test class validation for different input types."""
    # Valid integer classes
    assert _validate_class_output(0, 2)
    assert _validate_class_output(1, 2)
    assert not _validate_class_output(2, 2)  # Out of range
    assert not _validate_class_output(-1, 2)  # Out of range
    
    # Valid string classes
    assert _validate_class_output("0", 2)
    assert _validate_class_output("1", 2)
    assert _validate_class_output("class 0", 2)
    assert _validate_class_output("prediction: 1", 2)
    assert not _validate_class_output("2", 2)  # Out of range
    assert not _validate_class_output("invalid", 2)  # No numbers
    
    # Valid float classes
    assert _validate_class_output(0.0, 2)
    assert _validate_class_output(1.0, 2)
    if FLOAT_NUM_TOLERANCE >= 0.3:
        assert _validate_class_output(0.7, 2)  # In tolerance
    if FLOAT_NUM_TOLERANCE < 0.3:
        assert not _validate_class_output(0.7, 2)  # Out of tolerance
    assert not _validate_class_output(2.0, 2)  # Out of range
    
    # Invalid inputs
    assert not _validate_class_output(None, 2)
    assert not _validate_class_output("", 2)
    assert not _validate_class_output([0, 1], 2)  # List not supported for discrete classes


def test_extract_class_index():
    """Test class index extraction from different input types."""
    # Integer inputs
    assert _extract_class_index(0, 2) == 0
    assert _extract_class_index(1, 2) == 1
    assert _extract_class_index(2, 2) == -1  # Invalid
    
    # String inputs
    assert _extract_class_index("0", 2) == 0
    assert _extract_class_index("1", 2) == 1
    assert _extract_class_index("class 0", 2) == 0
    assert _extract_class_index("prediction: 1", 2) == 1
    assert _extract_class_index("invalid", 2) == -1
    
    # Float inputs
    assert _extract_class_index(0.0, 2) == 0
    assert _extract_class_index(1.0, 2) == 1
    if FLOAT_NUM_TOLERANCE >= 0.3:
        assert _extract_class_index(0.7, 2) == 1  # Rounds to 1
    if FLOAT_NUM_TOLERANCE < 0.3:
        assert _extract_class_index(0.7, 2) == -1  # Invalid
    assert _extract_class_index(2.0, 2) == -1  # Invalid
    
    # Invalid inputs
    assert _extract_class_index(None, 2) == -1
    assert _extract_class_index("", 2) == -1


def test_classification_metrics_calculator_discrete_predictions():
    """Test classification metrics calculator with discrete class predictions."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # Test case 1: Perfect predictions
    predictions = [0, 1, 0, 1, 0]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert math.isclose(metrics['valid_accuracy'], 1.0)
    assert math.isclose(metrics['macro_precision'], 1.0)
    assert math.isclose(metrics['macro_recall'], 1.0)
    assert math.isclose(metrics['macro_f1'], 1.0)
    assert metrics['total_invalid_preds'] == 0
    
    # Test case 2: Mixed predictions with some invalid
    predictions = [0, 1, "invalid", 0, 2]  # 2 is out of range
    ground_truth = [0, 1, 1, 0, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # 3 out of 5 correct (including invalid as wrong)
    assert math.isclose(metrics['overall_accuracy'], 3/5)
    # 3 out of 3 valid predictions correct
    assert math.isclose(metrics['valid_accuracy'], 1.0)
    assert metrics['total_invalid_preds'] == 2
    
    # Test case 3: String predictions
    predictions = ["0", "class 1", "prediction: 0", "1", "invalid text"]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # 4 out of 5 correct
    assert math.isclose(metrics['overall_accuracy'], 4/5)
    assert metrics['total_invalid_preds'] == 1


def test_classification_metrics_calculator_per_class_metrics():
    """Test per-class precision, recall, and F1 metrics."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # Test case: Imbalanced predictions
    predictions = [0, 0, 0, 1, 1]  # 3 class 0, 2 class 1
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
    predictions = ["invalid", None, "bad", 2, -1]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    assert metrics['total_invalid_preds'] == 5
    assert metrics['overall_accuracy'] == 0.0
    assert metrics['invalid_percentage'] == 100.0
    
    # Mismatched lengths should raise error
    with pytest.raises(ValueError):
        calc.calculate_metrics([0, 1], [0, 1, 0])


def test_classification_metrics_calculator_different_num_classes():
    """Test with different numbers of classes."""
    # Test with 3 classes
    calc_3 = ClassificationMetricsCalculator(num_classes=3)
    
    predictions = [0, 1, 2, 0]
    ground_truth = [0, 1, 2, 0]
    metrics = calc_3.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert len(metrics['precision_per_class']) == 3
    assert len(metrics['recall_per_class']) == 3
    assert len(metrics['f1_per_class']) == 3


def test_classification_metrics_calculator_float_tolerance():
    """Test float tolerance for class validation."""
    calc = ClassificationMetricsCalculator(num_classes=2)
    
    # Test float inputs within tolerance
    predictions = [0.0, 1.0, 0.1, 0.9]  # .1 and .9 should be valid
    ground_truth = [0, 1, 0, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)

    if FLOAT_NUM_TOLERANCE >= 0.1:
        assert math.isclose(metrics['overall_accuracy'], 1.0)
        assert metrics['total_invalid_preds'] == 0
    if FLOAT_NUM_TOLERANCE < 0.1:
        assert math.isclose(metrics['overall_accuracy'], 0.5)
        assert metrics['total_invalid_preds'] == 2

if __name__ == "__main__":
    pytest.main(['-s', __file__])

