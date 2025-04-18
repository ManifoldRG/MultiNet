import numpy as np
import logging
from src.eval.profiling.openvla.experiments.robot.openvla_action_decoding_utils import (
    SimpleMapping, ManualRuleMapping, ExpertActionUtils
)
from definitions.procgen import ProcGenDefinitions

logger = logging.getLogger(__name__)

OPENVLA_STANDARD_ACTION_DIM = 7

def get_action_decoding_strategy(model, dataset_name) -> str:
    """Get action decoding strategy with fallback to default"""
    try:
        return model.norm_stats.get(dataset_name, {}).get(
            'action_decoding_strategy', 
            model.default_action_decoding_strategy
        )
    except AttributeError:
        raise ValueError("Default action decoding strategy not found")


def preprocess_expert_actions(batch, dataset_name, batch_idx, idx) -> list[float]:
    """Process batch actions with error handling"""
    try:
        action_data = batch['action'][batch_idx][idx]

        if dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
            action_data = ProcGenDefinitions.set_procgen_unused_special_action_to_stand_still(action_data, dataset_name)

        return ExpertActionUtils.drop_is_terminal_dim(action_data, dataset_name)
    except (IndexError, KeyError) as e:
        raise ValueError(f"Error processing actions for batch {batch_idx} and index {idx}: {e}")


def standardize_predicted_action(predicted_action, action_decoding_strategy, dataset_name) -> list[float]:
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
