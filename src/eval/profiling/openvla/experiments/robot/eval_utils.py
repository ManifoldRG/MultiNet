import numpy as np
import logging
from src.eval.profiling.openvla.experiments.robot.openvla_action_decoding_utils import (
    SimpleMapping, ManualRuleMapping, ExpertActionUtils
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

OPENVLA_STANDARD_ACTION_DIM = 7


def get_action_decoding_strategy(model, dataset_name):
    """Get action decoding strategy with fallback to default"""
    try:
        return model.norm_stats.get(dataset_name, {}).get(
            'action_decoding_strategy', 
            model.default_action_decoding_strategy
        )
    except AttributeError:
        raise ValueError("Default action decoding strategy not found")


def calculate_mse(predicted, actual):
    """Calculate mean squared error between predicted and actual values"""
    return np.mean((np.array(predicted) - np.array(actual)) ** 2)


def calculate_success_rate(success_list):
    """Calculate success rate percentage from a list of success indicators"""
    if len(success_list) == 0:
        logger.warning("Success list is empty. Defaulting to 0.0 success rate.")
        return 0.0
    return (sum(success_list) / len(success_list)) * 100


def normalize_mse_values(mse_values):
    """Normalize MSE values using min-max scaling"""
    if len(mse_values) == 0:
        logger.warning("No MSE values collected. Setting normalized MSE to 0.0")
        return 0.0
    
    min_mse = min(mse_values)
    max_mse = max(mse_values)
    normalized_mse = np.array(mse_values)
    return (normalized_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else np.zeros_like(normalized_mse)


def load_preprocessed_expert_action(batch, dataset_name, batch_idx, idx, action_decoding_strategy):
    """Process batch actions with error handling"""
    try:
        action_data = batch['action'][batch_idx][idx] if isinstance(batch['action'][batch_idx], list) else batch['action'][batch_idx]
        if action_decoding_strategy == 'manual_rule_mapping':
            action_data = ManualRuleMapping.filter_bigfish_expert_special_actions(action_data, dataset_name)
        return ExpertActionUtils.drop_is_terminal_dim(action_data, dataset_name)
    except (IndexError, KeyError) as e:
        raise ValueError(f"Error processing actions: {e}")


def standardize_predicted_action(predicted_action, action_decoding_strategy, dataset_name):
    """Standardize predicted action based on decoding strategy"""
    if action_decoding_strategy == 'manual_rule_mapping':
        assert predicted_action.shape[0] == OPENVLA_STANDARD_ACTION_DIM, \
            f"predicted action shape {predicted_action.shape[0]} != OpenVLA standard action dimension {OPENVLA_STANDARD_ACTION_DIM}"
        return ManualRuleMapping.decode_action(predicted_action, dataset_name)
    elif action_decoding_strategy == 'simple_mapping':
        return SimpleMapping.decode_action(predicted_action, dataset_name)
    elif action_decoding_strategy == 'naive_dim_extension':
        return predicted_action
    else:
        raise ValueError(f"Unknown action decoding strategy: {action_decoding_strategy}")
