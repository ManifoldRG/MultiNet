import numpy as np
import logging
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.data_utils.procgen_dataloader import get_procgen_dataloader
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.eval_utils import (
    get_action_decoding_strategy,
    calculate_mse,
    calculate_success_rate,
    normalize_mse_values,
    standardize_predicted_action,
    load_preprocessed_expert_action
)
from definitions.procgen import ProcGenDefinitions


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

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
    if action_decoding_strategy == cfg.default_action_decoding_strategy:
        logger.info(f"Action decoding strategy not found for dataset {dataset_name}. Defaulting to {cfg.default_action_decoding_strategy}")

    _, dataloader = get_procgen_dataloader(tfds_shards, batch_size=1)

    action_success = []
    timestep_mses = []

    obs = {}

    for batch in dataloader:
        obs_key = 'continuous_observation' if 'continuous_observation' in batch else 'image_observation'

        for batch_idx, batch_observations in enumerate(batch[obs_key]):
            if isinstance(batch_observations, list):
                obs_len = len(batch_observations)
            else:
                batch_observations = [batch_observations]
                obs_len = 1

            for idx in range(obs_len):
                actual_action = load_preprocessed_expert_action(batch, dataset_name, batch_idx, idx, action_decoding_strategy)
                logger.debug(f"Actual action: {actual_action}")
                if actual_action is None:
                    continue

                obs['full_image'] = batch_observations[idx]
                if obs['full_image'] is None:
                    raise ValueError(f"Observation is None for dataset {dataset_name}")

                try:
                    text_obs = next(iter(ProcGenDefinitions.DESCRIPTIONS.get(dataset_name).values()))[0]
                except (IndexError, KeyError) as e:
                    raise ValueError(f"Error getting text observation: {e}")
                
                logger.debug(f"Observation shape: {obs['full_image'].shape}")
                logger.debug(f"Text observation: {text_obs}")
                
                predicted_action = get_action(cfg, model, obs, text_obs, processor)
                logger.debug(f"Predicted action: {predicted_action}")

                # Standardize the predicted action to match the actual action space
                standardized_predicted_action = standardize_predicted_action(
                    predicted_action,
                    action_decoding_strategy,
                    dataset_name
                )

                logger.debug(f"Standardized predicted action: {standardized_predicted_action}")
                mse = calculate_mse(standardized_predicted_action, actual_action)
                timestep_mses.append(mse)

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

    action_success_rate = calculate_success_rate(action_success)
    logger.debug(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")

    total_dataset_amse = sum(timestep_mses)
    logger.info(f"\nTotal MSE across {len(timestep_mses)} timesteps: {total_dataset_amse:.4f}")
    num_timesteps = len(timestep_mses)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0

    normalized_mses = normalize_mse_values(timestep_mses)
    normalized_amse = sum(normalized_mses) / len(normalized_mses) if len(normalized_mses) > 0 else 0.0
    logger.debug(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")

    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse