import sys
import os
import time
import json
from dataclasses import dataclass
from typing import Union
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from PIL import Image
from src.eval.profiling.jat.scripts.openx_dataloader import get_openx_dataloader
from src.eval.profiling.openvla.experiments.robot.robot_utils import (get_model,
                                                                      get_image_resize_size,
                                                                      set_seed_everywhere,
                                                                      get_action)
from src.eval.profiling.openvla.experiments.robot.openvla_utils import get_processor

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


def evaluate_openvla_model(model, processor, tfds_shards, resize_size):
    # Initialize the dataloader for the OpenX dataset
    dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

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

            # Re-format the observation for OpenVLA
            image_raw = batch['image_observation'][0][idx]
            image = np.array(image_raw)
            image = Image.fromarray(image)
            image = image.resize((resize_size, resize_size))
            image = np.array(image).astype(np.uint8)
            obs['full_image'] = image
            
            # Get the model's predicted action
            predicted_action = get_action(cfg, model, obs, batch['text_observation'][0][idx], processor)

            predicted_action_4d = np.array([predicted_action[0], 
                                            predicted_action[1], 
                                            predicted_action[2], 
                                            to_discrete(predicted_action[6])])

            # Get the actual (expert) action
            actual_action = batch['action'][0][idx]
            # print(f"actual_action: {actual_action}")
            # print(f"predicted_action: {predicted_action_4d}")

            # Calculate RMSE for this timestep
            mse = np.mean((np.array(predicted_action_4d) - np.array(actual_action)) ** 2)
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


def profile_openvla_on_openx(cfg: EvalConfig):
    model = get_model(cfg)
    processor = get_processor(cfg)

    resize_size = get_image_resize_size(cfg)

    # Path to OpenX datasets
    openx_datasets_path = '/home/locke/ManifoldRG/MultiNet/data/translated'  # TODO: Add the path

    # Get list of all OpenX datasets
    openx_dataset_paths = os.listdir(openx_datasets_path)

    eval_results = {}

    for openx_dataset in openx_dataset_paths:
        print(f'\nEvaluating dataset: {openx_dataset}\n')

        # Get all shards for the current dataset
        shard_files = os.listdir(os.path.join(openx_datasets_path, openx_dataset))
        sorted_shard_files = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        tfds_shards = [os.path.join(openx_datasets_path, openx_dataset, f) 
                       for f in sorted_shard_files]

        # Start timing
        start_time = time.time()

        # Evaluate OpenVLA model on the current dataset
        avg_mse_list, episode_count, total_dataset_amse, normalized_amse = evaluate_openvla_model(model, 
                                                                                                  processor, 
                                                                                                  tfds_shards, 
                                                                                                  resize_size)

        # End timing
        end_time = time.time()

        # Calculate evaluation time
        eval_time = end_time - start_time

        # Store results
        eval_results[openx_dataset] = {
            'avg_mse_list': avg_mse_list,
            'episode_count': episode_count,
            'total_dataset_amse': total_dataset_amse,
            'normalized_amse': normalized_amse,
            'eval_time': eval_time
        }

        print(f'Evaluation time for {openx_dataset}: {eval_time:.2f} seconds')

    # Print overall results
    print('\nOverall Results:')
    for dataset, result in eval_results.items():
        print(f'\nDataset: {dataset}')
        print(f'Episodes: {result["episode_count"]}')
        print(f'Total AMSE: {result["total_dataset_amse"]:.4f}')
        print(f'Normalized AMSE: {result["normalized_amse"]:.4f}')
        print(f'Evaluation Time: {result["eval_time"]:.2f} seconds')

    # Save results to a JSON file
    with open('openvla_openx_usc_evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print("\nEval results have been saved to 'openvla_openx_usc_evaluation_results.json'")


if __name__ == "__main__":
    cfg = EvalConfig()
    profile_openvla_on_openx(cfg)
