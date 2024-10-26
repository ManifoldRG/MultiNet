import sys
from dataclasses import dataclass
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

import numpy as np


def evaluate_openvla_model(cfg, model, processor, tfds_shards, dataset_name):
    dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    total_dataset_amse = 0.0
    action_success = []
    timestep_mse = []

    obs = {}

    # batch_idx = 0
    for batch in dataloader:
        # num_timesteps = len(batch['continuous_observation'][0])
        # print(f"Batch {batch_idx}:")
        # print(f"  Number of timesteps: {num_timesteps}")

        #Because the batch size is 1, 1 batch contains 1 episode, which is why the first element is indexed
        for idx in range(len(batch['continuous_observation'][0])):
            # Get the actual (expert) action
            actual_action = batch['action'][0][idx]

            mse = 0.0

            obs['full_image'] = batch['image_observation'][0][idx]

            # Debug information
            # print(f"  Timestep {idx}:")
            # print(f"    Image shape: {batch['image_observation'][0][idx].shape if batch['image_observation'][0][idx] is not None else 'None'}")
            # print(f"    Text observation: {batch['text_observation'][0][idx]}")

            # Check if the image is None
            if obs['full_image'] is None:
                raise Exception(f"Image is None for timestep {idx} for dataset {dataset_name}.")


            # Get the model's predicted action
            predicted_action = get_action(cfg, 
                                          model, 
                                          obs, 
                                          batch['text_observation'][0][idx], 
                                          processor)
            

            # Get the actual (expert) action
            actual_action = drop_is_terminal_dim(batch['action'][0][idx], dataset_name)

            # Standardize the predicted action to match the actual action space
            standardized_predicted_action = convert_action(predicted_action,
                                                           dataset_name)

            # print(f"Predicted action: {predicted_action}")
            # print(f"Standardized predicted action: {standardized_predicted_action}")
            # print(f"Actual action: {actual_action}")

            # Calculate RMSE for this timestep
            mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
            timestep_mse.append(mse)

            # At the last timestep, check if the predicted action is the same as the actual action. If yes, it is considered a success
            if batch['is_last'][0][idx] == True:
                print(f"Standardized predicted action: {np.array(standardized_predicted_action)}")
                print(f"Standardized actual action: {np.array(actual_action)}")
                if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                    action_success.append(1)
                else:
                    action_success.append(0)


    action_success_rate = (sum(action_success) / len(action_success)) * 100
    # print(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")

    # Calculate overall average RMSE across all episodes
    total_dataset_amse = sum(timestep_mse)
    print(f"\nTotal MSE across {len(timestep_mse)} timesteps: {total_dataset_amse:.4f}")
    num_timesteps = len(timestep_mse)
    avg_dataset_amse = total_dataset_amse / num_timesteps
    
    # Calculate min-max normalized AMSE
    min_mse = min(timestep_mse)
    max_mse = max(timestep_mse)
    normalized_mse = (timestep_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else 0    
    normalized_amse = sum(normalized_mse) / len(normalized_mse)
    
    # print(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
    # print(f"Timesteps for each episode: {timesteps}")
    # print(f"Total timesteps: {sum(timesteps)}")


    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse
