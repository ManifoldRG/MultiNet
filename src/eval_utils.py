import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_mae(predicted_action, actual_action) -> float:
    """Calculate mean absolute error from absolute errors"""
    return np.mean(np.abs(np.array(predicted_action) - np.array(actual_action)))

def calculate_mse(predicted, actual) -> float:
    """Calculate mean squared error between predicted and actual values"""
    return np.mean((np.array(predicted) - np.array(actual)) ** 2)

def calculate_brier_mae(predicted_probabilities, one_hot_label) -> float:
    """Calculate mean absolute error from absolute errors"""
    return np.sum(np.abs(np.array(predicted_probabilities) - np.array(one_hot_label)))

def calculate_brier_mse(predicted_probabilities, one_hot_label) -> float:
    """Calculate mean absolute error from absolute errors"""
    return np.sum((np.array(predicted_probabilities) - np.array(one_hot_label)) ** 2)

def calculate_success_rate(success_list) -> float:
    """Calculate success rate percentage from a list of success indicators"""
    if len(success_list) == 0:
        logger.warning("Success list is empty. Defaulting to 0.0 success rate.")
        return 0.0
    return (sum(success_list) / len(success_list)) * 100

def quantile_filter(values: list[float], quantile: float = 0.05) -> list[float]:
    """Filter values within `quantile` and `100 - quantile` quantiles"""
    if len(values) == 0:
        raise ValueError("No values collected")

    q5 = np.percentile(values, quantile)
    q95 = np.percentile(values, 100 - quantile)

    return [val for val in values if q5 <= val <= q95]

def min_max_normalize(values: list[float]) -> list[float]:
    """Normalize values using min-max scaling"""
    if len(values) == 0:
        raise ValueError("No values collected")

    if len(values) == 1:
        raise ValueError("Only one value collected")

    min_val = min(values)
    max_val = max(values)
    normalized_values = np.array(values)
    return (normalized_values - min_val) / (max_val - min_val) if max_val != min_val else np.zeros_like(normalized_values)

def calculate_mean(values: list[float]) -> float:
    """Calculate average of a list of values"""
    if len(values) == 0:
        raise ValueError("No values collected")
    return np.mean(values)


def calculate_max_relative_mae(mae_values: list[float]) -> float:
    """Calculate maximum relative error (outlier severity)"""
    if len(mae_values) == 0:
        raise ValueError("No values collected")

    abs_values = np.abs(np.array(mae_values))
    median_value = np.median(abs_values)
    if median_value == 0:
        median_value = 1e-10
    return max(abs_values) / median_value


def calculate_proportion_beyond_mae_threshold(mae_values: list[float], threshold_multiplier: float = 3.0) -> float:
    """Calculate proportion of values beyond threshold (outlier frequency)"""
    if len(mae_values) == 0:
        raise ValueError("No values collected")

    median_value = np.median(mae_values)
    threshold = threshold_multiplier * median_value
    return sum(1 for value in mae_values if value > threshold) / len(mae_values)