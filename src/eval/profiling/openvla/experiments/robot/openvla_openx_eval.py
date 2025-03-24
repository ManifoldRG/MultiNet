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

from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.eval_utils import (
    get_action_decoding_strategy,
    calculate_mse,
    calculate_mae,
    calculate_mean,
    calculate_success_rate,
    quantile_filter,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
    min_max_normalize,
    standardize_predicted_action,
    preprocess_expert_actions
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def evaluate_openvla_on_openx(cfg, model, processor, tfds_shards, dataset_name):
    action_decoding_strategy = get_action_decoding_strategy(model, dataset_name)
    if action_decoding_strategy == cfg.default_action_decoding_strategy:
        logger.info(f"Action decoding strategy not found for dataset {dataset_name}. Defaulting to {cfg.default_action_decoding_strategy}")

    dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    action_success = []
    timestep_mses = []
    timestep_maes = []

    obs = {}

    for batch in dataloader:
        try:
            obs_len = len(batch['continuous_observation'][0])
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error accessing continuous_observation: {e}")
            logger.info(f"Available keys: {batch.keys()}")
            continue

        for idx in range(obs_len):
            try:
                # batch_idx is 0 for OpenX dataset
                actual_action = preprocess_expert_actions(batch, dataset_name, 0, idx, action_decoding_strategy)
                logger.debug(f"Actual action: {actual_action}")
                if actual_action is None:
                    continue

                obs['full_image'] = batch['continuous_observation'][0][idx]
                if obs['full_image'] is None:
                    raise ValueError(f"Observation is None for dataset {dataset_name}")

                text_obs = batch['text_observation'][0][idx]
                if text_obs is None:
                    raise ValueError(f"Text observation is None for dataset {dataset_name}")
                
                logger.debug(f"Observation shape: {obs['full_image'].shape}")
                logger.debug(f"Text observation: {text_obs}")

                predicted_action = get_action(cfg, model, obs, text_obs, processor)
                logger.debug(f"Predicted action: {predicted_action}")

                standardized_predicted_action = standardize_predicted_action(
                    predicted_action,
                    action_decoding_strategy,
                    dataset_name
                )
                logger.debug(f"Standardized predicted action: {standardized_predicted_action}")

                mse = calculate_mse(standardized_predicted_action, actual_action)
                timestep_mses.append(mse)

                mae = calculate_mae(standardized_predicted_action, actual_action)
                timestep_maes.append(mae)
                
                if batch['is_last'][0][idx] == True:
                    logger.info(f"Episode final predicted action: {np.array(standardized_predicted_action)}")
                    logger.info(f"Episode final actual action: {np.array(actual_action)}")
                    if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                        action_success.append(1)
                    else:
                        action_success.append(0)
            except (IndexError, KeyError) as e:
                logger.warning(f"Error processing OpenX dataset at index {idx}: {e}")
                continue

    # Calculate MAE metrics
    normalized_maes = min_max_normalize(timestep_maes)
    average_normalized_mae = calculate_mean(normalized_maes)

    logger.debug(f"Normalized MAEs length for the dataset: {len(normalized_maes)}")
    logger.debug(f"Normalized Average MAE for the dataset: {average_normalized_mae:.4f}")

    # Calculate quantile filtered MAE metrics
    quantile_filtered_maes = quantile_filter(timestep_maes)
    normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
    average_quantile_filtered_normalized_mae = calculate_mean(normalized_quantile_filtered_maes)
    
    logger.debug(f"Quantile filtered MAEs length for the dataset: {len(quantile_filtered_maes)}")
    logger.debug(f"Average quantile filtered NMAE for the dataset: {average_quantile_filtered_normalized_mae:.4f}")

    max_rel_mae = calculate_max_relative_mae(timestep_maes)
    prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)

    logger.debug(f"Maximum Relative MAE: {max_rel_mae:.4f}")
    logger.debug(f"Proportion Beyond MAE Threshold (3x median): {prop_beyond_threshold_mae:.4f}")    

    # Multinet v0.1 metrics
    action_success_rate = calculate_success_rate(action_success)
    logger.debug(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")

    total_dataset_amse = sum(timestep_mses)
    logger.info(f"\nTotal MSE across {len(timestep_mses)} timesteps: {total_dataset_amse:.4f}")
    num_timesteps = len(timestep_mses)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0

    normalized_mses = min_max_normalize(timestep_mses)
    normalized_amse = calculate_mean(normalized_mses)
    logger.debug(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")

    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse, average_normalized_mae, average_quantile_filtered_normalized_mae
