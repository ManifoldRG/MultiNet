import argparse
import os
import time
import json
import numpy as np
import sys
import gc
import torch
from dataclasses import dataclass
from datetime import datetime
from typing import Union

from pathlib import Path

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))


from src.eval.profiling.openvla.experiments.robot.robot_utils import get_model
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
    dataset_statistics_path: str = ""
    openx_datasets_path: str = ""


def sort_files_in_folder_by_name(dataset_path: str) -> list[str]:
    # Get all shards for the current dataset
    shard_files = os.listdir(dataset_path)

    if not shard_files:
        raise FileNotFoundError(f"No files found in dataset directory: {dataset_path}")

    if os.path.basename(dataset_path) == "bigfish":
    # Sort the procgen files by the timestamp in the filename before the first underscore _ in ascending order
        return sorted(
            shard_files, 
            key=lambda x: datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S")
        )
    else:
        return sorted(shard_files, key=lambda x: int(x.split('_')[-1]))


def profile_openvla_on_openx(cfg: EvalConfig, result_save_path: str):
    # Get list of all OpenX datasets
    try:
        openx_dataset_paths = [
            d for d in os.listdir(cfg.openx_datasets_path) 
            if os.path.isdir(os.path.join(cfg.openx_datasets_path, d))
        ]
        print(f"Found {len(openx_dataset_paths)} OpenX datasets in {cfg.openx_datasets_path}")
    except FileNotFoundError:
        print(f"Error: The specified openx_datasets_path does not exist: {cfg.openx_datasets_path}")
        return
    
    if not openx_dataset_paths:
        print(f"Warning: No subdirectories found in {cfg.openx_datasets_path}")
        return

    eval_results = {}
    result_file_path = Path(result_save_path) / 'openvla_openx_eval_results.json'

    for openx_dataset in openx_dataset_paths:
        if openx_dataset == "utokyo_xarm_bimanual_converted_externally_to_rlds":
            print(f'\nSkipping dataset: {openx_dataset}\n')
            continue
        # Skip if the dataset is already in the eval_results
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as f:
                completed_datasets = json.load(f)
            
            if openx_dataset in completed_datasets:
                print(f'\nSkipping dataset: {openx_dataset} (already evaluated)\n')
                continue
        
        dataset_path = Path(cfg.openx_datasets_path) / openx_dataset
        print(f'\nEvaluating dataset: {openx_dataset}')
        print(f'Dataset path: {dataset_path}')

        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset directory does not exist: {dataset_path}")
            continue

        try:
            sorted_shard_files = sort_files_in_folder_by_name(dataset_path)
        except FileNotFoundError:
            print(f"Error: Unable to access dataset directory: {dataset_path}")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        tfds_shards = [os.path.join(cfg.openx_datasets_path, openx_dataset, f) 
                       for f in sorted_shard_files]

        # Reset GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load models with the corresponding cfg that affects the unnormalization of the action space
        cfg = EvalConfig()
        cfg.openx_datasets_path = args.openx_datasets_path
        cfg.dataset_statistics_path = args.dataset_statistics_path
        cfg.unnorm_key = openx_dataset
        model = get_model(cfg)
        processor = get_processor(cfg)

        # Start timing
        start_time = time.time()


        # Evaluate OpenVLA model on the current dataset
        (
            action_success_rate,
            total_dataset_amse,
            avg_dataset_amse,
            num_timesteps,
            normalized_amse,
            total_huber_loss,
            avg_huber_loss
        ) = evaluate_openvla_model(
            cfg,
            model,
            processor,
            tfds_shards,
            openx_dataset,
            use_huber_loss=True
        )
        
        del model
        del processor
        torch.cuda.empty_cache()
        gc.collect()

        # End timing
        end_time = time.time()

        # Calculate evaluation time
        eval_time = end_time - start_time

        # Store results
        eval_results[openx_dataset] = {
            'action_success_rate': action_success_rate,
            'total_dataset_amse': total_dataset_amse,
            'eval_time': eval_time,
            'num_timesteps': num_timesteps,
            'avg_dataset_amse': avg_dataset_amse,
            'normalized_amse': normalized_amse,
            'total_huber_loss': total_huber_loss,
            'avg_huber_loss': avg_huber_loss
        }

        print(f'Evaluation time for {openx_dataset}: {eval_time:.2f} seconds')

        # Save intermediate results to a JSON file to ensure progress is not lost
        # Check if the file already exists

        if os.path.exists(result_file_path):
            # If it exists, load the existing data
            with open(result_file_path, 'r') as f:
                existing_results = json.load(f)
            # Append new data to existing data
            existing_results.update(eval_results)
        else:
            # If it doesn't exist, use the current eval_results
            existing_results = eval_results

        # Write the updated or new results to the file
        with open(result_file_path, 'w') as f:
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
        print(f'Total Huber Loss: {result["total_huber_loss"]:.4f}')
        print(f'Average Huber Loss: {result["avg_huber_loss"]:.4f}')
    print(f"\nEval results have been saved to '{result_file_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA on OpenX datasets")
    parser.add_argument("--openx_datasets_path", type=str, required=True, help="Path to the OpenX datasets")
    parser.add_argument("--dataset_statistics_path", type=str, required=True, help="Path to the dataset statistics")
    parser.add_argument("--result_save_path", type=str, required=True, help="Path to save the evaluation results")
    args = parser.parse_args()

    cfg = EvalConfig()
    cfg.openx_datasets_path = args.openx_datasets_path
    cfg.dataset_statistics_path = args.dataset_statistics_path
    profile_openvla_on_openx(cfg, args.result_save_path)
