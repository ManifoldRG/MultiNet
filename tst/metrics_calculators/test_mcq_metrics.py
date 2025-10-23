import sys
import os
import math

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)
import pytest

from src.eval_harness.scoring.mcq_metrics import (
    _validate_choice_output,
    MCQMetricsCalculator,
)


def test_validate_choice_output():
    """Test choice validation - adapter should have already extracted to int."""
    # Valid integer choices (adapter already extracted)
    assert _validate_choice_output(0, 2)
    assert _validate_choice_output(1, 2)
    assert not _validate_choice_output(2, 2)  # Out of range
    assert not _validate_choice_output(-1, 2)  # Adapter returns -1 for invalid
    
    # Adapter should return int, not string (these should fail validation)
    assert not _validate_choice_output("0", 2)
    assert not _validate_choice_output("1", 2)
    assert not _validate_choice_output("choice 0", 2)
    assert not _validate_choice_output("answer: 1", 2)
    
    # Adapter should return int, not float (these should fail validation)
    assert not _validate_choice_output(0.0, 2)
    assert not _validate_choice_output(1.0, 2)
    assert not _validate_choice_output(0.7, 2)
    
    # Invalid inputs
    assert not _validate_choice_output(None, 2)
    assert not _validate_choice_output("", 2)
    assert not _validate_choice_output([0, 1], 2)


def test_mcq_metrics_calculator_perfect_predictions():
    """Test MCQ metrics calculator with perfect predictions."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    predictions = [
        {"raw_output": "Choice 0", "extracted_outputs": 0},
        {"raw_output": "Choice 1", "extracted_outputs": 1},
        {"raw_output": "Choice 0", "extracted_outputs": 0},
        {"raw_output": "Choice 1", "extracted_outputs": 1},
        {"raw_output": "Choice 0", "extracted_outputs": 0},
    ]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert math.isclose(metrics['valid_accuracy'], 1.0)
    assert metrics['total_invalid_preds'] == 0
    assert metrics['total_samples'] == 5
    assert metrics['valid_predictions'] == 5


def test_mcq_metrics_calculator_with_invalid():
    """Test MCQ metrics with some invalid predictions."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    # 2 correct, 1 wrong, 2 invalid (adapter returned -1 for invalid)
    predictions = [
        {"raw_output": "Choice 0", "extracted_outputs": 0},
        {"raw_output": "Choice 1", "extracted_outputs": 1},
        {"raw_output": "Invalid text", "extracted_outputs": -1},  # Adapter returned -1
        {"raw_output": "Choice 0", "extracted_outputs": 0},
        {"raw_output": "Choice 2 (out of range)", "extracted_outputs": -1},  # Adapter returned -1
    ]
    ground_truth = [0, 1, 1, 1, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # Overall: 2 out of 5 correct (invalids count as wrong)
    assert math.isclose(metrics['overall_accuracy'], 2/5)
    # Valid only: 2 out of 3 correct
    assert math.isclose(metrics['valid_accuracy'], 2/3)
    assert metrics['total_invalid_preds'] == 2
    assert metrics['total_samples'] == 5
    assert metrics['valid_predictions'] == 3


def test_mcq_metrics_calculator_string_predictions():
    """Test with adapter that properly extracted strings to integers."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    # Adapter should have extracted these to integers
    predictions = [
        {"raw_output": "0", "extracted_outputs": 0},  # Adapter extracted
        {"raw_output": "choice 1", "extracted_outputs": 1},  # Adapter extracted
        {"raw_output": "answer: 0", "extracted_outputs": 0},  # Adapter extracted
        {"raw_output": "1", "extracted_outputs": 1},  # Adapter extracted
        {"raw_output": "invalid text", "extracted_outputs": -1},  # Adapter returned -1
    ]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    # 4 out of 5 correct (invalid is wrong)
    assert math.isclose(metrics['overall_accuracy'], 4/5)
    assert metrics['total_invalid_preds'] == 1
    assert metrics['valid_predictions'] == 4


def test_mcq_metrics_calculator_all_invalid():
    """Test edge case with all invalid predictions (adapter returned -1)."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    # All predictions are invalid - adapter returned -1 for all
    predictions = [
        {"raw_output": "invalid", "extracted_outputs": -1},
        {"raw_output": "None", "extracted_outputs": -1},
        {"raw_output": "bad", "extracted_outputs": -1},
        {"raw_output": "2 (out of range)", "extracted_outputs": -1},
        {"raw_output": "-5 (negative)", "extracted_outputs": -1},
    ]
    ground_truth = [0, 1, 0, 1, 0]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    assert metrics['total_invalid_preds'] == 5
    assert metrics['overall_accuracy'] == 0.0
    assert metrics['invalid_percentage'] == 100.0
    assert metrics['valid_predictions'] == 0


def test_mcq_metrics_calculator_choice_distribution():
    """Test that choice distribution is correctly tracked."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    predictions = [
        {"raw_output": "0", "extracted_outputs": 0},
        {"raw_output": "0", "extracted_outputs": 0},
        {"raw_output": "0", "extracted_outputs": 0},
        {"raw_output": "1", "extracted_outputs": 1},
        {"raw_output": "1", "extracted_outputs": 1},
    ]
    ground_truth = [0, 0, 1, 1, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    assert metrics['choice_distribution']['choice_0_count'] == 3
    assert metrics['choice_distribution']['choice_1_count'] == 2
    assert metrics['num_choices'] == 2


def test_mcq_metrics_calculator_different_num_choices():
    """Test with different numbers of choices (e.g., 4 for some MCQ tasks)."""
    calc_4 = MCQMetricsCalculator(num_choices=4)
    
    predictions = [
        {"raw_output": "0", "extracted_outputs": 0},
        {"raw_output": "1", "extracted_outputs": 1},
        {"raw_output": "2", "extracted_outputs": 2},
        {"raw_output": "3", "extracted_outputs": 3},
        {"raw_output": "0", "extracted_outputs": 0},
    ]
    ground_truth = [0, 1, 2, 3, 0]
    metrics = calc_4.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert metrics['num_choices'] == 4
    assert metrics['choice_distribution']['choice_0_count'] == 2
    assert metrics['choice_distribution']['choice_1_count'] == 1
    assert metrics['choice_distribution']['choice_2_count'] == 1
    assert metrics['choice_distribution']['choice_3_count'] == 1


def test_mcq_metrics_calculator_mismatched_lengths():
    """Test error handling for mismatched lengths."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    with pytest.raises(ValueError):
        calc.calculate_metrics([
            {"raw_output": "0", "extracted_outputs": 0},
            {"raw_output": "1", "extracted_outputs": 1},
        ], [0, 1, 0])


def test_mcq_metrics_calculator_float_inputs():
    """Test that floats are treated as invalid (adapter should have converted to int)."""
    calc = MCQMetricsCalculator(num_choices=2)
    
    # Floats should be invalid - adapter should have rounded them to int
    # These would fail validation in the new approach
    predictions = [
        {"raw_output": "0.0", "extracted_outputs": 0},  # Adapter rounded to int
        {"raw_output": "1.0", "extracted_outputs": 1},  # Adapter rounded to int
        {"raw_output": "0.3", "extracted_outputs": 0},  # Adapter rounded to int
        {"raw_output": "0.7", "extracted_outputs": 1},  # Adapter rounded to int
    ]
    ground_truth = [0, 1, 0, 1]
    metrics = calc.calculate_metrics(predictions, ground_truth)
    
    assert math.isclose(metrics['overall_accuracy'], 1.0)
    assert metrics['total_invalid_preds'] == 0


if __name__ == "__main__":
    pytest.main(['-s', __file__])

