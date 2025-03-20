import sys
import logging
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.data_utils.procgen_dataloader import get_procgen_dataloader
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.multinet_openvla_utils import convert_action, drop_is_terminal_dim

import numpy as np

logger = logging.getLogger(__name__)

OPENVLA_STANDARD_ACTION_DIM = 7

def get_action_decoding_strategy(model, dataset_name):
    """Get action decoding strategy with fallback to default"""
    return model.norm_stats.get(dataset_name, {}).get(
        'action_decoding_strategy', 
        model.default_action_decoding_strategy
    )


def evaluate_openvla_on_procgen(cfg, model, processor, tfds_shards, dataset_name):
    """
    Evaluate OpenVLA model on ProcGen dataset.
    
    Args:
        cfg: Configuration for evaluation
        model: OpenVLA model
        processor: Image processor for OpenVLA
        tfds_shards: List of TFDS shards
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse)
    """
    action_decoding_strategy = get_action_decoding_strategy(model, dataset_name)
    if action_decoding_strategy == model.default_action_decoding_strategy:
        logger.warning(f"Action decoding strategy not found for dataset {dataset_name}. Defaulting to {model.default_action_decoding_strategy}")

    _, dataloader = get_procgen_dataloader(tfds_shards, batch_size=1)

    action_success = []
    timestep_mse = []

    obs = {}

    for batch in dataloader:
        logger.debug(f"Batch keys: {batch.keys()}")
        
        if 'continuous_observation' in batch:
            obs_key = 'continuous_observation'
        elif 'image_observation' in batch:
            obs_key = 'image_observation'
        else:
            available_keys = list(batch.keys())
            logger.error(f"Neither 'continuous_observation' nor 'image_observation' found in batch. Available keys: {available_keys}")
            raise KeyError(f"Missing observation key in batch for dataset {dataset_name}")
        
        # Process each item in the batch
        for batch_idx, batch_observations in enumerate(batch[obs_key]):
            # For procgen datasets, each batch item might contain multiple observations
            if isinstance(batch_observations, list):
                obs_len = len(batch_observations)
            else:
                # If it's not a list, treat it as a single observation
                batch_observations = [batch_observations]
                obs_len = 1
            
            for idx in range(obs_len):
                # Get the actual (expert) action - handle different batch structures
                if batch_idx < len(batch['action']) and idx < len(batch['action'][batch_idx]):
                    action_data = batch['action'][batch_idx][idx] if isinstance(batch['action'][batch_idx], list) else batch['action'][batch_idx]
                    actual_action = drop_is_terminal_dim(action_data, dataset_name)
                else:
                    logger.warning(f"Action data not available for batch_idx={batch_idx}, idx={idx}")
                    continue
                
                # Get the preprocessed image from the dataloader
                obs['full_image'] = batch_observations[idx]
                
                # Check if the image is None
                if obs['full_image'] is None:
                    logger.warning(f"Image is None for timestep {idx} for dataset {dataset_name}. Skipping.")
                    continue
                
                # Get the text observation from the dataloader
                text_obs = batch['text_observation'][batch_idx][idx]
                
                # Get the model's predicted action
                predicted_action = get_action(cfg, model, obs, text_obs, processor)
                
                # Standardize the predicted action to match the actual action space
                if action_decoding_strategy == 'manual_rule_mapping':
                    assert predicted_action.shape[0] == OPENVLA_STANDARD_ACTION_DIM, \
                        f"predicted action shape {predicted_action.shape[0]} != OpenVLA standard action dimension {OPENVLA_STANDARD_ACTION_DIM}"
                    standardized_predicted_action = convert_action(predicted_action, dataset_name)
                elif action_decoding_strategy == 'simple_mapping':
                    standardized_predicted_action = convert_action(predicted_action, dataset_name)
                elif action_decoding_strategy == 'naive_dimension_extension':
                    standardized_predicted_action = predicted_action
                else:
                    raise ValueError(f"Unknown action decoding strategy: {action_decoding_strategy}")
                
                # Calculate RMSE for this timestep
                mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
                timestep_mse.append(mse)
                
                # At the last timestep, check if the predicted action is the same as the actual action
                try:
                    is_last = batch['is_last'][batch_idx][idx] if isinstance(batch['is_last'][batch_idx], list) else batch['is_last'][batch_idx]
                    if is_last == True:
                        logger.info(f"Episode final predicted action: {np.array(standardized_predicted_action)}")
                        logger.info(f"Episode final actual action: {np.array(actual_action)}")
                        if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                            action_success.append(1)
                        else:
                            action_success.append(0)
                except (IndexError, KeyError) as e:
                    logger.warning(f"Error checking is_last: {e}")

    # Calculate success rate
    if len(action_success) == 0:
        logger.warning("Action success list is EMPTY. Defaulting to 0.0 action success rate.")
        action_success_rate = 0.0
    else:
        action_success_rate = (sum(action_success) / len(action_success)) * 100

    # Calculate overall average RMSE across all episodes
    total_dataset_amse = sum(timestep_mse)
    logger.info(f"\nTotal MSE across {len(timestep_mse)} timesteps: {total_dataset_amse:.4f}")
    num_timesteps = len(timestep_mse)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0
    
    # Calculate min-max normalized AMSE
    if len(timestep_mse) > 0:
        min_mse = min(timestep_mse)
        max_mse = max(timestep_mse)
        normalized_mse = np.array(timestep_mse)
        normalized_mse = (normalized_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else np.zeros_like(normalized_mse)
        normalized_amse = sum(normalized_mse) / len(normalized_mse)
    else:
        logger.warning("No timestep MSE values collected. Setting normalized AMSE to 0.0")
        normalized_amse = 0.0

    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse