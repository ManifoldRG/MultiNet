import os
import time
import json
import numpy as np
import sys

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
from src.eval.profiling.openvla.experiments.robot.openvla_openx_eval import EvalConfig, evaluate_openvla_model


def profile_openvla_on_openx(cfg: EvalConfig):
    model = get_model(cfg)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    # Path to OpenX datasets
    openx_datasets_path = '/home/locke/ManifoldRG/MultiNet/data/translated'  # TODO: Add the path

    # Get list of all OpenX datasets
    # openx_dataset_paths = [d for d in os.listdir(openx_datasets_path) if os.path.isdir(os.path.join(openx_datasets_path, d))]

    openx_dataset_paths = ['utokyo_pr2_opening_fridge_converted_externally_to_rlds']

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
        avg_mse_list, episode_count, total_dataset_amse, normalized_amse = evaluate_openvla_model(cfg, 
                                                                                                  model, 
                                                                                                  processor, 
                                                                                                  tfds_shards, 
                                                                                                  resize_size,
                                                                                                  openx_dataset)

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
    with open('openvla_openx_usc_evaluation_results_1.json', 'w') as f:
        json.dump(eval_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print("\nEval results have been saved to 'openvla_openx_usc_evaluation_results.json'")


if __name__ == "__main__":
    cfg = EvalConfig()
    profile_openvla_on_openx(cfg)
