import numpy as np
import logging
from pathlib import Path
import sys
import os

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.eval_utils import (
    get_action_decoding_strategy,
    calculate_mse,
    calculate_success_rate,
    normalize_mse_values,
    standardize_predicted_action,
    load_preprocessed_expert_action
)

logger = logging.getLogger(__name__)
if os.environ.get('ENVIRONMENT', 'prod') == 'dev':
    logger.setLevel(logging.DEBUG)

def evaluate_openvla_on_openx(cfg, model, processor, tfds_shards, dataset_name):
    action_decoding_strategy = get_action_decoding_strategy(model, dataset_name)
    if action_decoding_strategy == cfg.default_action_decoding_strategy:
        logger.info(f"Action decoding strategy not found for dataset {dataset_name}. Defaulting to {cfg.default_action_decoding_strategy}")

    dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    action_success = []
    timestep_mses = []

    obs = {}
    batch_counter = 0

    for batch in dataloader:
        try:
            obs_len = len(batch['continuous_observation'][0])
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error accessing continuous_observation: {e}")
            logger.info(f"Available keys: {batch.keys()}")
            continue

        for idx in range(obs_len):
            logger.debug("================================")
            logger.debug(f"{dataset_name} batch-{batch_counter} timestep-{idx}")
            logger.debug("================================")
            try:
                # batch_idx is 0 for OpenX dataset
                obs['full_image'] = batch['image_observation'][0][idx]
                if obs['full_image'] is None:
                    raise ValueError(f"Observation is None for dataset {dataset_name}")

                text_obs = batch['text_observation'][0][idx]
                if text_obs is None:
                    raise ValueError(f"Text observation is None for dataset {dataset_name}")
                
                logger.debug(f"Batch fields: {batch.keys()}")
                logger.debug(f"Batch continuous obs: {batch['continuous_observation'][0][idx]}")
                logger.debug(f"Image obs shape: {obs['full_image'].shape}")
                logger.debug(f"Text obs: {text_obs}")

                predicted_action = get_action(cfg, model, obs, text_obs, processor)
                logger.debug(f"Predicted action: {predicted_action}")

                standardized_predicted_action = standardize_predicted_action(
                    predicted_action,
                    action_decoding_strategy,
                    dataset_name
                )
                actual_action = load_preprocessed_expert_action(batch, dataset_name, 0, idx, action_decoding_strategy)
                if actual_action is None:
                    raise ValueError(f"Actual action is None for dataset {dataset_name}")
                
                logger.debug(f"Standardized predicted action: {standardized_predicted_action}")
                logger.debug(f"Actual action: {actual_action}")

                mse = calculate_mse(standardized_predicted_action, actual_action)
                timestep_mses.append(mse)

                if batch['is_last'][0][idx] == True:
                    logger.info(f"Batch {batch_counter} final predicted action: {np.array(standardized_predicted_action)}")
                    logger.info(f"Batch {batch_counter} final actual action: {np.array(actual_action)}")
                    if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                        action_success.append(1)
                    else:
                        action_success.append(0)
            except (IndexError, KeyError) as e:
                logger.warning(f"Error processing OpenX dataset at index {idx}: {e}")
                continue

        batch_counter += 1
        if batch_counter == 5:
            break

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
