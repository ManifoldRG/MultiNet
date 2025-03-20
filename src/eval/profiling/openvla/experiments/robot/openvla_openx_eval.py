import sys
from dataclasses import dataclass
import logging
import json
from typing import Union
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.multinet_openvla_utils import convert_action, drop_is_terminal_dim
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from definitions.procgen import ProcGenDefinitions

import numpy as np

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

OPENVLA_STANDARD_ACTION_DIM = 7

def evaluate_openvla_model(cfg, model, processor, tfds_shards, dataset_name):
    try:
        action_decoding_strategy = model.norm_stats[dataset_name]['action_decoding_strategy']
    except KeyError:
        logger.warning(f"Action decoding strategy not found for dataset {dataset_name}. Defaulting to manual_mapping.")
        action_decoding_strategy = 'manual_mapping'

    if dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
        _, dataloader = get_procgen_dataloader(tfds_shards, batch_size=1)
    else:
        dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    total_dataset_amse = 0.0
    action_success = []
    timestep_mse = []

    obs = {}

    for batch in dataloader:
        logger.debug(f"Batch keys: {batch.keys()}")
        
        # Handle different batch structures for procgen vs openx datasets
        if dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
            # For procgen datasets, the batch structure is different
            # Each key in the batch contains a list of items
            
            # Check if 'continuous_observation' exists in the batch
            if 'continuous_observation' in batch:
                obs_key = 'continuous_observation'
            elif 'image_observation' in batch:
                obs_key = 'image_observation'
            else:
                # Log available keys for debugging
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
                    
                    # Debug information
                    logger.debug(f"  Timestep {idx}:")
                    
                    # Safely access batch data with proper error handling
                    try:
                        logger.debug(f"Batch action: {batch['action'][batch_idx][idx] if isinstance(batch['action'][batch_idx], list) else batch['action'][batch_idx]}")
                        text_obs_data = batch['text_observation'][batch_idx][idx] if isinstance(batch['text_observation'][batch_idx], list) else batch['text_observation'][batch_idx]
                        logger.debug(f"Batch text_obs: {text_obs_data}")
                        logger.debug(f"Batch reward: {batch['reward'][batch_idx][idx] if isinstance(batch['reward'][batch_idx], list) else batch['reward'][batch_idx]}")
                        logger.debug(f"Batch is_last: {batch['is_last'][batch_idx][idx] if isinstance(batch['is_last'][batch_idx], list) else batch['is_last'][batch_idx]}")
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Error accessing batch data: {e}")
                    
                    logger.debug(f"Image shape: {obs['full_image'].shape if obs['full_image'] is not None else 'None'}")
                    
                    # Check if the image is None
                    if obs['full_image'] is None:
                        logger.warning(f"Image is None for timestep {idx} for dataset {dataset_name}. Skipping.")
                        continue
                    
                    # Get the text observation from the dataloader
                    text_obs = text_obs_data if 'text_obs_data' in locals() else None
                    
                    # Get the model's predicted action
                    predicted_action = get_action(cfg, model, obs, text_obs, processor)
                    
                    # Standardize the predicted action to match the actual action space
                    if action_decoding_strategy == 'manual_mapping':
                        assert predicted_action.shape[0] == OPENVLA_STANDARD_ACTION_DIM, \
                            f"predicted action shape {predicted_action.shape[0]} != OpenVLA standard action dimension {OPENVLA_STANDARD_ACTION_DIM}"
                        standardized_predicted_action = convert_action(predicted_action, dataset_name)
                    else: # naive dimension extension strategy
                        standardized_predicted_action = predicted_action
                    
                    # Debug information
                    logger.debug(f"Standardized predicted action: {standardized_predicted_action}")
                    logger.debug(f"Standardized actual action: {actual_action}")
                    
                    # Calculate RMSE for this timestep
                    mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
                    timestep_mse.append(mse)
                    
                    logger.debug(f"Predicted action types: {[type(x) for x in standardized_predicted_action]}")
                    logger.debug(f"Actual action types: {[type(x) for x in actual_action]}")
                    
                    # At the last timestep, check if the predicted action is the same as the actual action. If yes, it is considered a success
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
        else:
            # For OpenX datasets, the batch structure is different
            # Get the number of observations in the batch
            try:
                obs_len = len(batch['continuous_observation'][0])
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Error accessing continuous_observation: {e}")
                logger.info(f"Available keys: {batch.keys()}")
                if 'continuous_observation' not in batch:
                    logger.error("'continuous_observation' key not found in batch")
                    if len(batch.keys()) > 0:
                        logger.info(f"First key value type: {type(batch[list(batch.keys())[0]])}")
                continue

            for idx in range(obs_len):
                try:
                    # Get the actual (expert) action
                    actual_action = drop_is_terminal_dim(batch['action'][0][idx], dataset_name) # only drops is_last dim if the action space has one

                    # Get the preprocessed image from the dataloader
                    obs['full_image'] = batch['continuous_observation'][0][idx]
                    
                    # Debug information
                    logger.debug(f"  Timestep {idx}:")
                    logger.debug(f"Batch action: {batch['action'][0][idx]}")
                    logger.debug(f"Batch text_obs: {batch['text_observation'][0][idx]}")
                    logger.debug(f"Batch reward: {batch['reward'][0][idx]}")
                    logger.debug(f"Batch is_last: {batch['is_last'][0][idx]}")
                    logger.debug(f"Image shape: {obs['full_image'].shape if obs['full_image'] is not None else 'None'}")

                    # Check if the image is None
                    if obs['full_image'] is None:
                        logger.warning(f"Image is None for timestep {idx} for dataset {dataset_name}. Skipping.")
                        continue

                    # Get the text observation from the dataloader
                    text_obs = batch['text_observation'][0][idx]
                    
                    # Get the model's predicted action
                    predicted_action = get_action(cfg, 
                                                  model, 
                                                  obs, 
                                                  text_obs,
                                                  processor)
                    
                    # Standardize the predicted action to match the actual action space
                    if action_decoding_strategy == 'manual_mapping':
                        assert predicted_action.shape == (OPENVLA_STANDARD_ACTION_DIM,), f"action shape {predicted_action.shape} != {OPENVLA_STANDARD_ACTION_DIM}"
                        standardized_predicted_action = convert_action(predicted_action, dataset_name)
                    else: # naive dimension extension strategy
                        standardized_predicted_action = predicted_action
                    
                    # Debug information
                    logger.debug(f"Standardized predicted action: {standardized_predicted_action}")
                    logger.debug(f"Standardized actual action: {actual_action}")
                    
                    # Calculate RMSE for this timestep
                    mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
                    timestep_mse.append(mse)
                    
                    logger.debug(f"Predicted action types: {[type(x) for x in standardized_predicted_action]}")
                    logger.debug(f"Actual action types: {[type(x) for x in actual_action]}")
                    
                    # At the last timestep, check if the predicted action is the same as the actual action. If yes, it is considered a success
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


    # TODO: delete after testing, should probably catch the error during production runs?
    if len(action_success) == 0:
        logger.warning("Action success list is EMPTY. Defaulting to 0.0 action success rate.")
        action_success_rate = 0.0
    else:
        action_success_rate = (sum(action_success) / len(action_success)) * 100
    # print(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")

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
    
    # print(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
    # print(f"Timesteps for each episode: {timesteps}")
    # print(f"Total timesteps: {sum(timesteps)}")

    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse
