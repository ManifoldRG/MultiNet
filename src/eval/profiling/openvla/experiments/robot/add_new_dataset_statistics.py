import os
import numpy as np
import json

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader


def calculate_action_statistics(actions):
    actions = np.array(actions)

    if actions[0].shape[0] < 7:
        actions = np.pad(actions, ((0, 0), (0, 7 - actions[0].shape[0])), mode='constant')
    elif actions[0].shape[0] == 8:
        actions = actions[:, :7]  # Assume last dimension is is_terminal
    else:
        raise ValueError(f"Unexpected action shape: {actions[0].shape}")

    stats = {
        "mean": np.mean(actions, axis=0).tolist(),
        "std": np.std(actions, axis=0).tolist(),
        "max": np.max(actions, axis=0).tolist(),
        "min": np.min(actions, axis=0).tolist(),
        "q01": np.percentile(actions, 1, axis=0).tolist(),
        "q99": np.percentile(actions, 99, axis=0).tolist(),
        "mask": [True] * (len(actions[0]) - 1) + [False]  # Assuming last dimension is gripper
    }
    return stats


def add_new_dataset_stats_to_json_file(dataloader, dataset_name, dataset_statistics_path):
    actions = []
    num_trajectories = 0


    for batch in dataloader:
        num_trajectories += 1
        for idx in range(len(batch['continuous_observation'][0])):
            actual_action = batch['action'][0][idx]
            actions.append(actual_action)

    action_stats = calculate_action_statistics(actions)

    new_robot_stats = {
        dataset_name: {
            "action": action_stats,
            "proprio": {
                "mean": [0.0] * 7,
                "std": [0.0] * 7,
                "max": [0.0] * 7,
                "min": [0.0] * 7,
                "q01": [0.0] * 7,
                "q99": [0.0] * 7
            },
            "num_transitions": len(actions),
            "num_trajectories": num_trajectories
        }
    }

    with open(dataset_statistics_path, 'r') as f:
        existing_stats = json.load(f)

    # Add new robot statistics
    existing_stats[dataset_name] = new_robot_stats[dataset_name]

    with open(dataset_statistics_path, 'w') as f:
        json.dump(existing_stats, f, indent=2)

    print(f"Updated statistics saved to {dataset_statistics_path}")

        
if __name__ == "__main__":
    # TODO: Update the paths
    openx_datasets_path = '/home/locke/ManifoldRG/MultiNet/data/translated'
    dataset_statistics_path = '/home/locke/ManifoldRG/MultiNet/src/eval/profiling/openvla/data/dataset_statistics.json'

    # Get list of all OpenX datasets
    openx_dataset_paths = [d for d in os.listdir(openx_datasets_path) if os.path.isdir(os.path.join(openx_datasets_path, d))]
    # openx_dataset_paths = ['utokyo_pr2_opening_fridge_converted_externally_to_rlds']

    for openx_dataset in openx_dataset_paths:
        # Get all shards for the current dataset
        shard_files = os.listdir(os.path.join(openx_datasets_path, openx_dataset))
        sorted_shard_files = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        tfds_shards = [os.path.join(openx_datasets_path, openx_dataset, f) 
                        for f in sorted_shard_files]
        
        dataloader = get_openx_dataloader(tfds_shards, batch_size=1, resize_size=224)  # 224 for OpenVLA

        add_new_dataset_stats_to_json_file(dataloader, openx_dataset, dataset_statistics_path)