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


@dataclass
class EvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    center_crop: bool = True
    seed: int = 7
    unnorm_key = "bridge_orig"

def standardize_predicted_action(predicted_action, dataset_name):
    def convert_usc(pred):
        return np.array([pred[0], pred[1], pred[2], to_discrete(pred[6])])
    
    def convert_utokyo_pr2(pred):
        # utokyo_pr2 uses 8D: [3x pos delta, 3x RPY angles, 1x gripper, 1x terminal]
        # We'll use the first 6 dimensions as-is, discretize the gripper, and add a dummy terminal action
        pos_range_scaler = 1000  # 1m = 1000mm

        """
        TODO:
        1. Robot-specific unnormalization
        2. Reference frame conversion
        3. Unit conversion
        """
        return np.array([
            pred[0] * pos_range_scaler, pred[1] * pos_range_scaler, pred[2] * pos_range_scaler,  # positional delta
            pred[3], pred[4], pred[5],  # RPY angles
            to_discrete(pred[6]),       # discretized gripper command
            0                           # dummy terminal action (always 0 as OpenVLA doesn't predict this)
        ])

    
    def convert_nyu_rot(pred):
        # FIXME: NYU ROT actually is 7D but the google sheet says it uses 4D: [del_x, del_y, del_z, gripper]?
        return np.array([pred[0], pred[1], pred[2], 0, 0, 0, to_discrete(pred[6])])

    conversion_functions = {
        "usc_cloth_sim_converted_externally_to_rlds": convert_usc,
        "nyu_rot_dataset_converted_externally_to_rlds": convert_nyu_rot,
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": convert_utokyo_pr2
    }
    
    convert_func = conversion_functions.get(dataset_name)
    
    if convert_func is None:
        raise ValueError(f"Dataset {dataset_name} action space standardization not implemented")
    
    return convert_func(predicted_action)

def evaluate_openvla_model(cfg, model, processor, tfds_shards, resize_size, dataset_name):
    dataloader = get_openx_dataloader(tfds_shards, batch_size=1, resize_size=resize_size)

    avg_mse_list = []
    total_dataset_amse = 0.0
    episode_count = 0
    obs = {}

    timesteps = []

    for batch in dataloader:
    
        num_timesteps = len(batch['continuous_observation'][0])
        print(f"Episode {episode_count + 1}:")
        print(f"  Number of timesteps: {num_timesteps}")

        timesteps.append(num_timesteps)

        episode_mse = []

        # model.reset_rl() # clear key-value cache for each episode

        #Because the batch size is 1, 1 batch contains 1 episode, which is why the first element is indexed
        for idx in range(len(batch['continuous_observation'][0])):

            #Model is not given a reward prior to the first action it predicts
            if idx == 0:
                reward = None
            else:
                reward = batch['reward'][0][idx-1]

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

            print(f"Predicted action: {standardized_predicted_action}")
            print(f"Actual action: {actual_action}")

            # Calculate RMSE for this timestep
            mse = np.mean((np.array(standardized_predicted_action) - np.array(actual_action)) ** 2)
            episode_mse.append(mse)

        # Calculate average RMSE for the episode
        avg_episode_mse = np.mean(episode_mse)
        avg_mse_list.append(avg_episode_mse)
        total_dataset_amse += avg_episode_mse
        episode_count += 1

        print(f"Episode {episode_count} - Average MSE: {avg_episode_mse:.4f}")

    # Calculate overall average RMSE across all episodes
    print(f"\nTotal Average MSE across {episode_count} episodes: {total_dataset_amse:.4f}")

    # Calculate average AMSE over all episodes
    avg_dataset_amse = total_dataset_amse / episode_count
    
    # Calculate min-max normalized AMSE
    min_amse = min(avg_mse_list)
    max_amse = max(avg_mse_list)
    normalized_amse = (avg_dataset_amse - min_amse) / (max_amse - min_amse) if max_amse != min_amse else 0
    

    print(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
    print(f"Timesteps for each episode: {timesteps}")
    print(f"Total timesteps: {sum(timesteps)}")
    return avg_mse_list, episode_count, total_dataset_amse, normalized_amse
