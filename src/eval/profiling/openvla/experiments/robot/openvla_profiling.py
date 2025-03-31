import argparse
import logging
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

from src.eval.profiling.openvla.experiments.robot.robot_utils import get_model, set_seed_everywhere
from src.eval.profiling.openvla.experiments.robot.openvla_utils import get_processor
from src.eval.profiling.openvla.experiments.robot.openvla_openx_eval import evaluate_openvla_on_openx
from src.eval.profiling.openvla.experiments.robot.openvla_procgen_eval import evaluate_openvla_on_procgen
from definitions.procgen import ProcGenDefinitions
from definitions.openx import OpenXDefinitions


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

PROCGEN_DATASET_NAMES = ProcGenDefinitions.DESCRIPTIONS.keys()
OPENX_DATASET_NAMES = OpenXDefinitions.DESCRIPTIONS.keys()

# List of datasets to evaluate
PROFILING_DATASETS = [
    d for d in OPENX_DATASET_NAMES
] + [
    d for d in PROCGEN_DATASET_NAMES
]

@dataclass
class EvalConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    center_crop: bool = True
    seed: int = 7
    unnorm_key: str = "bridge_orig"  # default unnorm_key bridge_orig
    dataset_statistics_path: str = ""
    default_action_decoding_strategy: str = "simple_mapping"


def clear_gpu_memory() -> None:
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception as e:
            logger.error(f"Error deleting PyTorch tensor: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.debug("PyTorch memory cleared")
        torch.cuda.reset_peak_memory_stats()
    
    gc.collect()
    logger.debug("Garbage collector collected objects")


def log_gpu_memory_usage() -> None:
    if torch.cuda.is_available():
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        logger.debug(f"[{timestamp}] Peak GPU Usage: {peak_mb / 1024:.1f}GiB")


def get_dataset_files(datasets_dir: str) -> list[str]:
    try:
        dataset_paths = [
            d for d in os.listdir(datasets_dir) 
            if os.path.isdir(os.path.join(datasets_dir, d))
        ]
        if not dataset_paths:
            logger.warning(f"No subdirectories found in {datasets_dir}")
            return []
        
        return dataset_paths
    except FileNotFoundError:
        logger.error(f"The specified path does not exist: {datasets_dir}")
        return []


def sort_files_in_folder_by_name(dataset_path: str) -> list[str]:
    shard_files = os.listdir(dataset_path)

    if not shard_files:
        raise FileNotFoundError(f"No files found in dataset directory: {dataset_path}")

    if os.path.basename(dataset_path) in PROCGEN_DATASET_NAMES:
        # Sort the procgen files by the timestamp in the filename before the first underscore _ in ascending order
        return sorted(
            shard_files,
            key=lambda x: datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S")
        )
    elif os.path.basename(dataset_path) in OPENX_DATASET_NAMES:
        return sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
    else:
        raise ValueError(f"Dataset type undefined in definitions: {os.path.basename(dataset_path)}")


def is_dataset_completed(dataset_name: str, result_file_path: Path) -> bool:
    if not result_file_path.exists():
        return False
        
    try:
        with open(result_file_path, 'r') as f:
            completed_datasets = json.load(f)
        
        if dataset_name in completed_datasets:
            return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading results file: {e}")
        
    return False


def process_single_dataset(
    dataset_name: str,
    model,
    processor,
    eval_cfg: EvalConfig,
    tfds_shards: list[str]
) -> dict[str, any]:
    logger.info(f"\nEvaluating {dataset_name}...")
    start_time = time.time()

    if dataset_name in PROCGEN_DATASET_NAMES:
        logger.debug(f"Evaluating {dataset_name} on procgen...")
        (
            num_timesteps,
            action_success_rate,
            total_dataset_amae,
            avg_dataset_amae,
            average_normalized_mae,
            total_quantile_filtered_mae,
            average_quantile_filtered_normalized_mae,
            max_rel_mae,
            prop_beyond_threshold_mae
        ) = evaluate_openvla_on_procgen(
            eval_cfg,
            model,
            processor,
            tfds_shards,
            dataset_name
        )

        # End timing
        end_time = time.time()
        eval_time = end_time - start_time
        logger.info(f'Evaluation time for {dataset_name}: {eval_time:.2f} seconds')

        return {
            'num_timesteps': num_timesteps,
            'action_success_rate': action_success_rate,
            'total_dataset_amae': total_dataset_amae,
            'avg_dataset_amae': avg_dataset_amae,
            'average_normalized_mae': average_normalized_mae,
            'total_quantile_filtered_mae': total_quantile_filtered_mae,
            'average_quantile_filtered_normalized_mae': average_quantile_filtered_normalized_mae,
            'max_rel_mae': max_rel_mae,
            'prop_beyond_threshold_mae': prop_beyond_threshold_mae,
            'eval_time': eval_time
        }
    elif dataset_name in OPENX_DATASET_NAMES:
        logger.debug(f"Evaluating {dataset_name} on openx...")
        (
            num_timesteps,
            action_success_rate,
            total_dataset_amae,
            avg_dataset_amae,
            average_normalized_mae,
            total_quantile_filtered_mae,
            average_quantile_filtered_normalized_mae,
            max_rel_mae,
            prop_beyond_threshold_mae
        ) = evaluate_openvla_on_openx(
            eval_cfg,
            model,
            processor,
            tfds_shards,
            dataset_name
        )

        end_time = time.time()
        eval_time = end_time - start_time
        logger.info(f'Evaluation time for {dataset_name}: {eval_time:.2f} seconds')

        return {
            'num_timesteps': num_timesteps,
            'action_success_rate': action_success_rate,
            'total_dataset_amae': total_dataset_amae,
            'avg_dataset_amae': avg_dataset_amae,
            'average_normalized_mae': average_normalized_mae,
            'total_quantile_filtered_mae': total_quantile_filtered_mae,
            'average_quantile_filtered_normalized_mae': average_quantile_filtered_normalized_mae,
            'max_rel_mae': max_rel_mae,
            'prop_beyond_threshold_mae': prop_beyond_threshold_mae,
            'eval_time': eval_time
        }
    else:
        raise ValueError(f"Dataset type undefined in definitions: {dataset_name}")


def save_results(results: dict[str, dict], result_file_path: Path) -> None:
    # Ensure directory exists
    result_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if available
    existing_results = {}
    if result_file_path.exists():
        try:
            with open(result_file_path, 'r') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error reading existing results file, creating new file")
    
    # Update with new results
    existing_results.update(results)
    
    # Save updated results
    with open(result_file_path, 'w') as f:
        json.dump(existing_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    logger.info(f"Eval results have been saved to '{result_file_path}'")


def profile_openvla(cfg: EvalConfig, profiling_dataset_folder_path: str, result_save_path: str):
    eval_dataset_files = get_dataset_files(profiling_dataset_folder_path)

    logger.info(f"Found {len(eval_dataset_files)} datasets in {profiling_dataset_folder_path}")
    logger.debug(f"Datasets: {eval_dataset_files}")

    if not eval_dataset_files:
        logger.error(f"No datasets found in {profiling_dataset_folder_path}")
        return

    eval_results = {}
    result_file_path = Path(result_save_path) / 'openvla_eval_results.json'
    
    clear_gpu_memory()
    set_seed_everywhere(cfg.seed)

    # Load model and processor
    try:
        processor = get_processor(cfg)
        model = get_model(cfg)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model or processor: {e}")
        return

    for dataset in eval_dataset_files:
        # Skip unsupported datasets
        if dataset not in PROFILING_DATASETS:
            logger.info(f"SKIPPING: {dataset} (not in list)")
            continue

        # Skip if the dataset is already in the eval_results
        if is_dataset_completed(dataset, result_file_path):
            logger.info(f"SKIPPING: {dataset} (already evaluated)")
            continue

        # Prepare dataset path
        dataset_path = Path(profiling_dataset_folder_path) / dataset

        if not os.path.exists(dataset_path):
            logger.warning(f"Warning: Dataset directory does not exist: {dataset_path}")
            continue
        else:
            logger.info(f'\nDATASET PATH: {dataset_path}')

        try:
            sorted_shard_files = sort_files_in_folder_by_name(dataset_path)
            tfds_shards = [os.path.join(profiling_dataset_folder_path, dataset, f) 
                            for f in sorted_shard_files]

            # Set unnormalization key for the dataset
            cfg.unnorm_key = dataset
        
            results = process_single_dataset(dataset, model, processor, cfg, tfds_shards)

            # Store results
            eval_results[dataset] = results
            save_results(eval_results, result_file_path)

            log_gpu_memory_usage()

        except FileNotFoundError:
            logger.error(f"Error: Unable to access dataset directory: {dataset_path}")
            continue
        except Exception as e:
            logger.exception(f"Error: {e}")
            continue

    # Cleanup
    if model is not None:
        del model
    if processor is not None:
        del processor

    clear_gpu_memory()

    # Print overall results
    logger.info('\n===== Overall Results =====')
    for dataset, result in eval_results.items():
        logger.info(f'\nDataset: {dataset}')
        logger.info(f'Total AMSE: {result["total_dataset_amse"]:.4f}')
        logger.info(f'Evaluation Time: {result["eval_time"]:.2f} seconds')
        logger.info(f'Action Success Rate: {result["action_success_rate"]:.4f}')
        logger.info(f'Average MSE: {result["avg_dataset_amse"]:.4f}')
        logger.info(f'Number of Timesteps: {result["num_timesteps"]}')
        logger.info(f'Normalized AMSE: {result["normalized_amse"]:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenVLA on datasets")
    parser.add_argument("--profiling_dataset_folder_path", type=str, required=True, help="Path to the parent folder of the profiling datasets")
    parser.add_argument("--dataset_statistics_path", type=str, required=True, help="Path to the dataset statistics")
    parser.add_argument("--result_save_path", type=str, required=True, help="Path to save the evaluation results")
    args = parser.parse_args()

    cfg = EvalConfig()
    cfg.dataset_statistics_path = args.dataset_statistics_path
    profile_openvla(cfg, args.profiling_dataset_folder_path, args.result_save_path)
