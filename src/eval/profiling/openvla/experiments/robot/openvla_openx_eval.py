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

import numpy as np


def to_discrete(action):
    return np.where(action >= 0.5, 1, 0)


def standardize_predicted_action(predicted_action, dataset_name):
    def convert_to_usc(pred):
        return np.array([pred[0], pred[1], pred[2], to_discrete(pred[6])])
    
    def convert_to_utokyo_pr2(pred):
        """
        - utokyo_pr2 uses 8D: [3x pos delta, 3x RPY angles, 1x gripper, 1x terminal]
          We'll use the first 6 dimensions as-is, discretize the gripper, and add a dummy terminal action

        TODO:
        1. Robot-specific unnormalization
        2. Reference frame conversion
        3. Unit conversion
        """
        pos_range_scaler = 1000  # 1m = 1000mm

        return np.array([
            pred[0] * pos_range_scaler, pred[1] * pos_range_scaler, pred[2] * pos_range_scaler,  # positional delta
            pred[3], pred[4], pred[5],  # RPY angles
            to_discrete(pred[6]),       # discretized gripper command
            0                           # dummy terminal action (always 0 as OpenVLA doesn't predict this)
        ])


    def convert_to_nyu_rot(pred):
        # FIXME: NYU ROT actually is 7D but the google sheet says it uses 4D: [del_x, del_y, del_z, gripper]?
        return np.array([pred[0], pred[1], pred[2], 0, 0, 0, to_discrete(pred[6])])


    conversion_functions = {
        "usc_cloth_sim_converted_externally_to_rlds": convert_to_usc,
        "nyu_rot_dataset_converted_externally_to_rlds": convert_to_nyu_rot,
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": convert_to_utokyo_pr2
    }
    
    convert_func = conversion_functions.get(dataset_name)
    
    if convert_func is None:
        raise ValueError(f"Dataset {dataset_name} action space standardization not implemented")
    
    return convert_func(predicted_action)


def evaluate_openvla_model(cfg, model, processor, tfds_shards, resize_size, dataset_name):
    dataloader = get_openx_dataloader(tfds_shards, batch_size=1, resize_size=resize_size)

    total_dataset_amse = 0.0
    action_success = []
    timestep_mse = []

    obs = {}

    timesteps = []

    all_actions = []
    action_ranges = {'min': None, 'max': None}

    batch_idx = 0
    for batch in dataloader:
        num_timesteps = len(batch['continuous_observation'][0])
        print(f"Batch {batch_idx}:")
        print(f"  Number of timesteps: {num_timesteps}")

        timesteps.append(num_timesteps)

        # model.reset_rl()  # clear key-value cache for each episode

        #Because the batch size is 1, 1 batch contains 1 episode, which is why the first element is indexed
        for idx in range(len(batch['continuous_observation'][0])):
            # Get the actual (expert) action
            actual_action = batch['action'][0][idx]
            all_actions.append(actual_action)

            # Update action ranges
            if action_ranges['min'] is None:
                action_ranges['min'] = actual_action
                action_ranges['max'] = actual_action
            else:
                action_ranges['min'] = np.minimum(action_ranges['min'], actual_action)
                action_ranges['max'] = np.maximum(action_ranges['max'], actual_action)




            #Model is not given a reward prior to the first action it predicts
            # if idx == 0:
            #     reward = None
            # else:
            #     reward = batch['reward'][0][idx-1]

            reward = None
            mse = 0.0

            obs['full_image'] = batch['image_observation'][0][idx]
            
            # Get the model's predicted action
            predicted_action = get_action(cfg, 
                                          model, 
                                          obs, 
                                          batch['text_observation'][0][idx], 
                                          processor)
            


            # Get the actual (expert) action
            actual_action = batch['action'][0][idx]

            # Standardize the predicted action to match the actual action space
            standardized_predicted_action = standardize_predicted_action(predicted_action, dataset_name)

            # actual_action = batch['action'][0][idx]
            # predicted_action = actual_action
            # standardized_predicted_action = actual_action

            print(f"Predicted action: {standardized_predicted_action}")
            print(f"Actual action: {actual_action}")

            # Calculate RMSE for this timestep
            mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
            timestep_mse.append(mse)

            # At the last timestep, check if the predicted action is the same as the actual action. If yes, it is considered a success
            if batch['is_last'][0][idx] == True:
                print(np.array(standardized_predicted_action))
                print(np.array(actual_action))
                if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                    action_success.append(1)
                else:
                    action_success.append(0)

    # action success rate
    action_success_rate = (sum(action_success) / len(action_success)) * 100
    print(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")

    # Calculate overall average RMSE across all episodes
    total_dataset_amse = sum(timestep_mse)
    print(f"\nTotal MSE across {len(timestep_mse)} timesteps: {total_dataset_amse:.4f}")
    num_timesteps = len(timestep_mse)
    avg_dataset_amse = total_dataset_amse / num_timesteps
    
    # Calculate min-max normalized AMSE
    min_mse = min(timestep_mse)
    max_mse = max(timestep_mse)
    if max_mse != min_mse:
        normalized_mse = [(mse - min_mse) / (max_mse - min_mse) for mse in timestep_mse]
    else:
        normalized_mse = [0] * len(timestep_mse)
    
    normalized_amse = sum(normalized_mse) / len(normalized_mse)
    
    print(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
    print(f"Timesteps for each episode: {timesteps}")
    print(f"Total timesteps: {sum(timesteps)}")

    # After the loop, analyze the action space
    all_actions = np.array(all_actions)
    
    print(f"\nAction space analysis for {dataset_name}:")
    print(f"Action dimensions: {all_actions.shape[1]}")
    print(f"Min values: {action_ranges['min']}")
    print(f"Max values: {action_ranges['max']}")
    print(f"Mean values: {np.mean(all_actions, axis=0)}")
    print(f"Std values: {np.std(all_actions, axis=0)}")

    # return avg_mse_list, episode_count, total_dataset_amse, normalized_amse
    return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse