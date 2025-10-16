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

import torch

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import evaluation harness components
from src.eval_harness.model_adapter import ModelAdapter

from src.data_utils import *
from src.eval_harness.scoring import *


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
        vqa_datasets = ['sqa3d', 'robot_vqa']
        grounding_datasets = ['odinw']
        tool_use_datasets = ['bfcl']
        
        if self.dataset in gameplay_datasets:
            return 'discrete_action'
        elif any(dataset in self.dataset for dataset in robotics_datasets):
            return 'continuous_action'
        elif self.dataset in mcq_datasets:
            return 'multiple_choice'
        elif self.dataset in vqa_datasets:
            return 'visual_question_answering'
        elif self.dataset in grounding_datasets:
            return 'visual_grounding'
        elif self.dataset in tool_use_datasets:
            return 'tool_use'
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset}")

def get_dataset_and_dataloader(config: EvaluationConfig) -> tuple:
    # Find data files
    if 'openx' in config.dataset:
        files, path = find_data_files('openx', config.disk_root_dir, dataset=config.dataset, split=config.data_split)
    elif 'robot_vqa' in config.dataset:
        files, path = find_data_files('openx', config.disk_root_dir, dataset='openx_multi_embodiment', split=config.data_split)
    else:
        files, path = find_data_files(config.dataset, config.disk_root_dir, split=config.data_split)
    
    if len(files) == 0:
        error_msg = f"No data found for dataset {config.dataset} in {path}. "
        raise FileNotFoundError(error_msg)
    
    # Get dataloader
    if 'openx' in config.dataset:
        dataset, data_loader = get_openx_dataloader(
            files, batch_size=config.batch_size, dataset_name=config.dataset
        )
    elif config.dataset == 'robot_vqa':
        dataset, data_loader = get_openx_dataloader(
            files, batch_size=config.batch_size, dataset_name='robot_vqa'
        )
    elif config.dataset == 'overcooked_ai':
        if len(files) > 1:
            raise NotImplementedError("No support for multiple datasets yet.")
        pickle_file = files[0]
            
        dataset, data_loader = get_overcooked_dataloader(
            pickle_file, batch_size=config.batch_size
        )
    elif config.dataset == 'piqa':
        if len(files) > 1:
            raise NotImplementedError("No support for multiple datasets yet.")
        jsonl_file = files[0]
        dataset, data_loader = get_piqa_dataloader(
            jsonl_file, batch_size=config.batch_size,
        )
    elif config.dataset == 'sqa3d':
        if len(files) > 1:
            raise NotImplementedError("No support for multiple datasets yet.")
        data_dict = files[0]
        question_file = data_dict['question_file']
        annotation_file = data_dict['annotation_file']
        images_dir = data_dict['images_dir']
        dataset, data_loader = get_sqa3d_dataloader(
            question_file, annotation_file, images_dir, batch_size=config.batch_size
        )
    elif config.dataset == 'odinw':
        dataset, data_loader = get_odinw_multi_dataloader(
            files, batch_size=config.batch_size
        )

    elif config.dataset == 'bfcl':
        if len(files) > 1:
            raise NotImplementedError("No support for multiple datasets yet.")
        data_dict = files[0]
        question_file = data_dict['question_file']
        answer_file = data_dict['answer_file']
        dataset, data_loader = get_bfcl_dataloader(
            question_file, answer_file, batch_size=config.batch_size
        )
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    return dataset, data_loader

def get_metrics_calculator(config: EvaluationConfig, dataset: torch.utils.data.Dataset):
    if 'openx' in config.dataset:
        action_stats = dataset.action_stats
        metrics_calculator = RoboticsMetricsCalculator(action_stats)
    elif config.dataset == 'robot_vqa':
        metrics_calculator = VQAMetricsCalculator()
    elif config.dataset == 'overcooked_ai':
        metrics_calculator = OvercookedAIMetricsCalculator()
    elif config.dataset == 'piqa':
        metrics_calculator = MCQMetricsCalculator()
    elif config.dataset == 'sqa3d':
        metrics_calculator = VQAMetricsCalculator()
    elif config.dataset == 'odinw':
        metrics_calculator = MCQMetricsCalculator()
    elif config.dataset == 'bfcl':
        metrics_calculator = VQAMetricsCalculator()
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    return metrics_calculator

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

def bordered_print(text: str):
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="MultiNet Model Evaluation Script")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (overcooked_ai, any openx morphology, piqa, sqa3d, robot_vqa, odinw, bfcl)")
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
    bordered_print("STEP 1: LOADING DATASET")

    dataset, data_loader = get_dataset_and_dataloader(config)

    # Step 2: Load model adapter
    bordered_print("STEP 2: LOADING MODEL ADAPTER")
    
    model_adapter = load_model_adapter(config)
    print(f"Model adapter loaded: {model_adapter.__class__.__name__}")
    
    # Step 3: Run and save predictions to file
    bordered_print("STEP 3: RUNNING PREDICTIONS")
    if not isinstance(data_loader, list):
        data_loaders = [data_loader]
    else:
        data_loaders = data_loader
    
    for dl in data_loaders:
        predictions = []
        ground_truth_actions = []
        for batch in dl:
            batch_predictions = model_adapter.batch_predict_actions(batch)
            predictions.extend(batch_predictions)
            ground_truth_actions.extend(batch['action'])
        save_predictions(predictions, config)

    # Step 4: Calculate metrics
    bordered_print("STEP 4: CALCULATING METRICS")
    
    metrics_calculator = get_metrics_calculator(config)
    metrics = metrics_calculator.calculate_metrics(predictions, ground_truth_actions)
    
    # Step 5: Save results
    bordered_print("STEP 5: SAVING METRICS")

    save_results(metrics, config)
    
    bordered_print("EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
