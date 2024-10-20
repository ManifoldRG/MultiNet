import os
import time
import json
import numpy as np
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


from src.eval.profiling.openvla.experiments.robot.robot_utils import (get_model,
                                                                      get_image_resize_size
                                                                      )
from src.eval.profiling.openvla.experiments.robot.openvla_utils import get_processor
from src.eval.profiling.openvla.experiments.robot.openvla_openx_eval import evaluate_openvla_model


@dataclass
class EvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    center_crop: bool = True
    seed: int = 7
    unnorm_key = "bridge_orig"  # default unnorm_key bridge_orig


def profile_openvla_on_openx(cfg: EvalConfig):
    # Path to OpenX datasets
    openx_datasets_path = '/home/locke/ManifoldRG/MultiNet/data/translated'  # TODO: Add the path

    # Get list of all OpenX datasets
    # openx_dataset_paths = [d for d in os.listdir(openx_datasets_path) if os.path.isdir(os.path.join(openx_datasets_path, d))]
    openx_dataset_paths = ['utokyo_pr2_opening_fridge_converted_externally_to_rlds']
    # openx_dataset_paths = ['usc_cloth_sim_converted_externally_to_rlds']
    # openx_dataset_paths = ['nyu_rot_dataset_converted_externally_to_rlds']

    eval_results = {}

    for openx_dataset in openx_dataset_paths:
        print(f'\nEvaluating dataset: {openx_dataset}\n')

        # Get all shards for the current dataset
        shard_files = os.listdir(os.path.join(openx_datasets_path, openx_dataset))
        sorted_shard_files = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        tfds_shards = [os.path.join(openx_datasets_path, openx_dataset, f) 
                       for f in sorted_shard_files]
        

        # Load models with the corresponding cfg that affects the unnormalization of the action space
        cfg = EvalConfig()
        cfg.unnorm_key = openx_dataset
        model = get_model(cfg)
        processor = get_processor(cfg)
        resize_size = get_image_resize_size(cfg)

        # Start timing
        start_time = time.time()


        # Evaluate OpenVLA model on the current dataset
        action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse = evaluate_openvla_model(cfg, 
                                                                                                  model, 
                                                                                                  processor, 
                                                                                                  tfds_shards, 
                                                                                                  resize_size,
                                                                                                  openx_dataset)

        # End timing
        end_time = time.time()

        # Calculate evaluation time
        eval_time = end_time - start_time

        # Store resultsp
        eval_results[openx_dataset] = {
            'action_success_rate': action_success_rate,
            'total_dataset_amse': total_dataset_amse,
            'eval_time': eval_time,
            'num_timesteps': num_timesteps,
            'avg_dataset_amse': avg_dataset_amse,
            'normalized_amse': normalized_amse
        }

        print(f'Evaluation time for {openx_dataset}: {eval_time:.2f} seconds')

        # Save intermediate results to a JSON file to ensure progress is not lost
        # Check if the file already exists
        if os.path.exists('openvla_openx_evaluation_results.json'):
            # If it exists, load the existing data
            with open('openvla_openx_evaluation_results.json', 'r') as f:
                existing_results = json.load(f)
            # Append new data to existing data
            existing_results.update(eval_results)
        else:
            # If it doesn't exist, use the current eval_results
            existing_results = eval_results

        # Write the updated or new results to the file
        with open('openvla_openx_evaluation_results.json', 'w') as f:
            json.dump(existing_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        print(f'Evaluation time for {openx_dataset}: {eval_time:.2f} seconds')

    # Print overall results
    print('\nOverall Results:')
    for dataset, result in eval_results.items():
        print(f'\nDataset: {dataset}')
        print(f'Total AMSE: {result["total_dataset_amse"]:.4f}')
        print(f'Evaluation Time: {result["eval_time"]:.2f} seconds')
        print(f'Action Success Rate: {result["action_success_rate"]:.4f}')
        print(f'Average MSE: {result["avg_dataset_amse"]:.4f}')
        print(f'Number of Timesteps: {result["num_timesteps"]}')
        print(f'Normalized AMSE: {result["normalized_amse"]:.4f}')
    print("\nEval results have been saved to 'openvla_openx_evaluation_results.json'")


if __name__ == "__main__":
    cfg = EvalConfig()
    profile_openvla_on_openx(cfg)
