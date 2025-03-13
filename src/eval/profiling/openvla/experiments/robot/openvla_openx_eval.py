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
from src.eval.profiling.openvla.experiments.robot.openvla_procgen_dataloader import get_procgen_dataloader


import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def evaluate_openvla_model(cfg, model, processor, tfds_shards, dataset_name):
    if dataset_name in ['bigfish']:
        dataloader = get_procgen_dataloader(tfds_shards, batch_size=1)
    else:
        dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    total_dataset_amse = 0.0
    action_success = []
    timestep_mse = []

    obs = {}

    for batch in dataloader:
        # TODO: move logic to separate procgen_eval script
        if dataset_name in ['bigfish']:
            obs_len = len(batch['observation'][0][0])
        else:
            #Because the batch size is 1, 1 batch contains 1 episode, which is why the first element is indexed
            obs_len = len(batch['continuous_observation'][0])

        for idx in range(obs_len):
            # Get the actual (expert) action
            actual_action = drop_is_terminal_dim(batch['action'][0][idx], dataset_name)

            mse = 0.0

            if dataset_name in ['bigfish']:
                # TODO: move this into openvla_procgen_dataloader
                # Transform Bigfish image data to correct format
                image_data = batch['observation'][0][idx]
                if image_data.shape == (3, 64, 64):
                    # Convert from (channels, height, width) to (height, width, channels)
                    obs['full_image'] = np.transpose(image_data, (1, 2, 0))
                    logger.debug("Image shape transposed: {}".format(obs['full_image'].shape))
                    if obs['full_image'].dtype != np.uint8:
                        obs['full_image'] = (obs['full_image'] * 255).astype(np.uint8)
                        logger.debug("Image dtype not uint8, converted to uint8")

                # Debug information
                logger.debug(f"  Timestep {idx}:")
                logger.debug(f"Procgen batch action: {batch['action'][0][idx]}")
                logger.debug(f"Procgen batch reward: {batch['reward'][0][idx]}")
                logger.debug(f"Procgen batch is_last: {batch['is_last'][0][idx]}")
                logger.debug(f"    Image shape: {obs['full_image'].shape if obs['full_image'] is not None else 'None'}")
            else:
                obs['full_image'] = batch['continuous_observation'][0][idx]

            # Check if the image is None
            if obs['full_image'] is None:
                raise Exception(f"Image is None for timestep {idx} for dataset {dataset_name}.")

            if dataset_name in ['bigfish']:
                text_obs = "eat other fishes"  # bigfish dataset has no text for the task labels
            else:
                text_obs = batch['text_observation'][0][idx]

            # Get the model's predicted action
            predicted_action = get_action(cfg, 
                                          model, 
                                          obs, 
                                          text_obs,
                                          processor)
            

            # Standardize the predicted action to match the actual action space
            standardized_predicted_action = convert_action(predicted_action,
                                                           dataset_name)

            # Debug information
            logger.debug(f"Standardized predicted action: {standardized_predicted_action}")
            logger.debug(f"Standardized actual action: {actual_action}")

            # Calculate RMSE for this timestep
            mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
            timestep_mse.append(mse)

            # At the last timestep, check if the predicted action is the same as the actual action. If yes, it is considered a success
            if batch['is_last'][0][idx] == True:
                logger.info(f"Episode final predicted action: {np.array(standardized_predicted_action)}")
                logger.info(f"Episode final actual action: {np.array(actual_action)}")
                if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                    action_success.append(1)
                else:
                    action_success.append(0)


    # TODO: delete after testing, should probably catch the error during production runs?
    if len(action_success) == 0:
        logger.warning("Action success list is empty. Defaulting to 0.0 action success rate.")
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
    min_mse = min(timestep_mse)
    max_mse = max(timestep_mse)
    normalized_mse = (timestep_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else 0    
    normalized_amse = sum(normalized_mse) / len(normalized_mse)
    
    # print(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
    # print(f"Timesteps for each episode: {timesteps}")
    # print(f"Total timesteps: {sum(timesteps)}")

    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse
