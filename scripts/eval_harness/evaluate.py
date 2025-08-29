#!/usr/bin/env python3
"""
MultiNet Evaluation Harness Script

This is the main evaluation script for the MultiNet benchmarking toolkit.
It loads datasets, runs model predictions, and calculates evaluation metrics.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import evaluation harness components
from src.eval_harness.model_adapter import ModelAdapter
from src.eval_harness.scoring.robotics_metrics import RoboticsMetricsCalculator
from src.data_utils.shard_finder import find_shards
from src.data_utils.openx_dataloader import get_openx_dataloader

class EvaluationConfig:
    """Configuration class for evaluation parameters."""
    
    def __init__(self, args):
        self.dataset = args.dataset
        self.model_adapter_module_path = args.model_adapter_module_path
        self.output_path = Path(args.output_path)
        self.data_split = args.data_split
        self.disk_root_dir = args.disk_root_dir
        self.batch_size = args.batch_size
        self.max_samples = args.max_samples
        self.device = args.device
        self.seed = args.seed
        
        # Validate and setup paths
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.output_path / "predictions"
        self.predictions_path.mkdir(exist_ok=True)
        
        # Auto-detect device if needed
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Validate dataset
        self.task_type = self._get_task_type()
        
    def _get_task_type(self) -> str:
        """Determine task type based on dataset name."""
        # TODO: update this to have better task type descriptions
        # TODO: create dataset_info.json file and move this logic there
        gameplay_datasets = ['overcooked_ai']
        robotics_datasets = ['openx']
        mcq_datasets = ['piqa']
        text_gen_datasets = ['sqa3d', 'robovqa']
        grounding_datasets = ['odinw']
        tool_use_datasets = ['bfclv3']
        
        if self.dataset in gameplay_datasets:
            return 'discrete_action'
        elif self.dataset in robotics_datasets:
            return 'continuous_action'
        elif self.dataset in mcq_datasets:
            return 'multiple_choice'
        elif self.dataset in text_gen_datasets:
            return 'text_generation'
        elif self.dataset in grounding_datasets:
            return 'grounding'
        elif self.dataset in tool_use_datasets:
            return 'tool_use'
        else:
            # Default to discrete action for unknown datasets
            print(f"Warning: Unknown dataset '{self.dataset}', defaulting to discrete_action task type")
            return 'discrete_action'


def load_model_adapter(config: EvaluationConfig) -> ModelAdapter:
    """Load the user's model adapter from the specified module."""
    print(f"Loading model adapter from: {config.model_adapter_module_path}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("user_adapter", config.model_adapter_module_path)
    if spec is None:
        raise ImportError(f"Could not load module from {config.model_adapter_module_path}")
        
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)
    
    # Find the adapter class
    adapter_class = None
    for attr_name in dir(user_module):
        attr = getattr(user_module, attr_name)
        if (isinstance(attr, type) and 
            issubclass(attr, ModelAdapter) and 
            attr != ModelAdapter and
            not bool(getattr(attr, "__abstractmethods__", False))):
                adapter_class = attr
                break
            
    if adapter_class is None:
        raise ValueError("No ModelAdapter subclass found in the provided module")
        
    # Instantiate the adapter
    print(f"Found adapter class: {adapter_class.__name__}")

    model_adapter = adapter_class()
    # Initialize the model
    print("Initializing model...")
    model_adapter.initialize(device=config.device, seed=config.seed)
    
    return model_adapter

def save_predictions(predictions: List[Dict[str, Any]], config: EvaluationConfig) -> None:
    """Save predictions to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions in JSONL format
    predictions_file = config.predictions_path / f"{config.dataset}_predictions_{timestamp}.jsonl"
    print(f"Saving {config.dataset} predictions to: {predictions_file}")
    
    with open(predictions_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, default=str) + '\n')
    
    print(f"{config.dataset} predictions saved to: {predictions_file}")

def save_results(metrics: Dict[str, Any], config: EvaluationConfig) -> None:
    """Save generated metrics to file."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    results_file = config.output_path / f"{config.dataset}_metrics_{timestamp}.json"
    print(f"Saving metrics to: {results_file}")
    
    summary = {
        'evaluation_config': {
            'dataset': config.dataset,
            'task_type': config.task_type,
            'model_adapter_path': str(config.model_adapter_module_path),
            'data_split': config.data_split,
            'batch_size': config.batch_size,
            'seed': config.seed,
            'timestamp': timestamp
        },
        'metrics': metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Metrics saved to: {results_file}")



def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="MultiNet Model Evaluation Script")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help="Dataset name (overcooked_ai, openx, piqa, sqa3d, robovqa, odinw, bfclv3)")
    parser.add_argument('--model_adapter_module_path', type=str, required=True,
                       help="Path to Python module containing ModelAdapter implementation")
    parser.add_argument('--output_path', type=str, required=True,
                       help="Directory to save predictions and results")

    # Optional arguments
    parser.add_argument('--disk_root_dir', type=str, default='/mnt/disks/mount_dir/MultiNet/src/v1/processed',
                       help="Root directory containing translated dataset files")
    parser.add_argument('--data_split', type=str, default='public', choices=['public', 'private'],
                       help="Data split to use ('public' for public samples, 'private' for private eval data)")
    parser.add_argument('--batch_size', type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument('--max_samples', type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help="Device to use for evaluation")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for deterministic evaluation")
    
    args = parser.parse_args()

    # Initialize configuration
    config = EvaluationConfig(args)
    print(f"Starting MultiNet evaluation...")
    print(f"Dataset: {config.dataset} (Task type: {config.task_type})")
    print(f"Output path: {config.output_path}")
    print(f"Device: {config.device}")
    
    # Step 1: Load dataset
    print(f"\n{'='*60}")
    print("STEP 1: LOADING DATASET")
    print(f"{'='*60}")
    
    if config.dataset == 'openx':
        shard_paths = find_shards('openx', config.disk_root_dir, split=config.data_split)
        if len(shard_paths) == 0:
            error_msg = f"No shards found for dataset {config.dataset} in split {config.data_split}. "
            if config.data_split == 'private':
                error_msg += "Please check if the private data is available under 'disk_root_dir/openx_*/test/'."
            else:
                error_msg += "Please check if the public data is available under 'disk_root_dir/openx_*/public/'."
            raise ValueError(error_msg)

        dataset, data_loader = get_openx_dataloader(
            shard_paths, batch_size=config.batch_size, dataset_name='openx', num_workers=0, by_episode=False
        )
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    
    # Step 2: Load model adapter
    print(f"\n{'='*60}")
    print("STEP 2: LOADING MODEL ADAPTER")
    print(f"{'='*60}")
    
    model_adapter = load_model_adapter(config)
    print(f"Model adapter loaded: {model_adapter.__class__.__name__}")
    
    # Step 3: Run and save predictions to file
    print(f"\n{'='*60}")
    print("STEP 3: RUNNING PREDICTIONS")
    print(f"{'='*60}")
    
    # Get predictions and ground truth actions
    if config.dataset == 'openx':
        predictions = []
        ground_truth_actions = []
        # Process all batches from the dataloader
        for batch in data_loader:
            batch_predictions = model_adapter.batch_predict_actions(batch)
            predictions.extend(batch_predictions)
            ground_truth_actions.extend(batch['action'])

        # save predictions to file
        save_predictions(predictions, config)
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    
    # Step 4: Calculate metrics
    print(f"\n{'='*60}")
    print("STEP 4: CALCULATING METRICS")
    print(f"{'='*60}")
    
    if config.dataset == 'openx':
        # load action stats
        action_stats = dataset.action_stats
        metrics_calculator = RoboticsMetricsCalculator(action_stats)
        
        # calculate metrics given predictions and ground truth actions
        metrics = metrics_calculator.calculate_metrics(predictions, ground_truth_actions)
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    
    # Step 5: Save results
    print(f"\n{'='*60}")
    print("STEP 5: SAVING METRICS")
    print(f"{'='*60}")
    
    save_results(metrics, config)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
