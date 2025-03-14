import argparse
import logging
import contextlib
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
import tensorflow as tf


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

SUPPORTED_DATASETS = [
        #     'nyu_door_opening_surprising_effectiveness',            'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
        #     'nyu_rot_dataset_converted_externally_to_rlds',         'usc_cloth_sim_converted_externally_to_rlds',
        #     'columbia_cairlab_pusht_real',                          'plex_robosuite',
        #     'conq_hose_manipulation',                               'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
        #     'eth_agent_affordances',                                'stanford_mask_vit_converted_externally_to_rlds',       
        #     'imperialcollege_sawyer_wrist_cam',                     'utokyo_pr2_opening_fridge_converted_externally_to_rlds'
        # ]
        'bigfish', 'bossfight', 'caveflyer', 'chaser']


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


def clear_gpu_memory() -> None:
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                logger.debug(f"Found CUDA tensor with shape: {obj.shape}")
                del obj
        except Exception as e:
            logger.error(f"Error deleting PyTorch tensor: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.debug("PyTorch memory cleared")
    
    gc.collect()
    logger.debug("Garbage collector collected objects")


def log_memory_usage():
    if torch.cuda.is_available():
        # Log only peak memory at DEBUG level
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        logger.debug(f"- Peak GPU usage: {peak_mb / 1024:.1f}GiB")
        
        # Detailed metrics only at DEBUG level
        if logger.isEnabledFor(logging.DEBUG):
            allocated = torch.cuda.memory_allocated() / 1024**2
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory / 1024**2
            logger.debug(f"- Memory details: {allocated / 1024:.1f}/{total / 1024:.1f}GiB ({allocated/total:.1%})")


@contextlib.contextmanager
def gpu_memory_context():
    try:
        clear_gpu_memory()
        log_memory_usage()
        yield
    finally:
        clear_gpu_memory()
        log_memory_usage()


def get_dataset_paths(datasets_dir: str) -> list[str]:
    try:
        dataset_paths = [
            d for d in os.listdir(datasets_dir) 
            if os.path.isdir(os.path.join(datasets_dir, d))
        ]
        if not dataset_paths:
            logger.warning(f"No subdirectories found in {datasets_dir}")
            return []
        
        logger.info(f"Found {len(dataset_paths)} datasets in {datasets_dir}")
        return dataset_paths
    except FileNotFoundError:
        logger.error(f"The specified path does not exist: {datasets_dir}")
        return []


def sort_files_in_folder_by_name(dataset_path: str) -> list[str]:
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


def is_dataset_completed(dataset_name: str, result_file_path: Path) -> bool:
    if not result_file_path.exists():
        return False
        
    try:
        with open(result_file_path, 'r') as f:
            completed_datasets = json.load(f)
        
        if dataset_name in completed_datasets:
            logger.info(f"Skipping dataset: {dataset_name} (already evaluated)")
            return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error reading results file: {e}")
        
    return False


def process_single_dataset(
    dataset_name: str, 
    model, 
    processor,
    dataset_cfg: EvalConfig, 
    tfds_shards: list[str]
) -> dict[str, any]:
    try:
        logger.info(f"\nEvaluating {dataset_name}...")
        # Start timing
        start_time = time.time()
        
        # Evaluate model
        results = evaluate_openvla_model(
            dataset_cfg,
            model,
            processor,
            tfds_shards,
            dataset_name
        )

        (action_success_rate, total_dataset_amse, avg_dataset_amse,
         num_timesteps, normalized_amse) = results
        
        # End timing
        end_time = time.time()
        eval_time = end_time - start_time
        logger.info(f'Evaluation time for {dataset_name}: {eval_time:.2f} seconds')

        # Return results
        return {
            'action_success_rate': action_success_rate,
            'total_dataset_amse': total_dataset_amse,
            'eval_time': eval_time,
            'num_timesteps': num_timesteps,
            'avg_dataset_amse': avg_dataset_amse,
            'normalized_amse': normalized_amse
        }
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor


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
    
    logger.info(f"Results saved to {result_file_path}")


def profile_openvla_on_openx(cfg: EvalConfig, result_save_path: str):
    openx_dataset_paths = get_dataset_paths(cfg.openx_datasets_path)
    if not openx_dataset_paths:
        return

    eval_results = {}
    result_file_path = Path(result_save_path) / 'openvla_openx_eval_results.json'

    # Load model and processor
    try:
        processor = get_processor(cfg)
        model = get_model(cfg)
        model.eval()
    except Exception as e:
        logger.error(f"Error loading model or processor: {e}")
        return

    for openx_dataset in openx_dataset_paths:
        with gpu_memory_context():
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Skip unsupported datasets
            if openx_dataset not in SUPPORTED_DATASETS:
                logger.info(f"SKIPPING: {openx_dataset} (not in list)")
                continue

            # Skip if the dataset is already in the eval_results
            if is_dataset_completed(openx_dataset, result_file_path):
                continue
            
            # Prepare dataset path
            dataset_path = Path(cfg.openx_datasets_path) / openx_dataset
            logger.info(f'\nDATASET PATH: {dataset_path}')

            if not os.path.exists(dataset_path):
                logger.warning(f"Warning: Dataset directory does not exist: {dataset_path}")
                continue

            try:
                sorted_shard_files = sort_files_in_folder_by_name(dataset_path)
                tfds_shards = [os.path.join(cfg.openx_datasets_path, openx_dataset, f) 
                                for f in sorted_shard_files]

                # Set unnormalization key for the dataset
                cfg.unnorm_key = openx_dataset
            
                results = process_single_dataset(openx_dataset, model, processor, cfg, tfds_shards)

                # Store results
                eval_results[openx_dataset] = results
                save_results(eval_results, result_file_path)

            except FileNotFoundError:
                logger.error(f"Error: Unable to access dataset directory: {dataset_path}")
                continue
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
            

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
    
    logger.info(f"\nEval results have been saved to '{result_file_path}'")


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
