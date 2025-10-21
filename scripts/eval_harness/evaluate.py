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
from typing import Any, Dict, List, Optional

import torch

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import evaluation harness components
from src.eval_harness.model_adapter import ModelAdapter
from src.eval_harness.v1_supported_datasets import V1SupportedDatasets
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
        image_classification_datasets = ['odinw']
        tool_use_datasets = ['bfcl']
        
        if self.dataset in gameplay_datasets:
            return 'discrete_action'
        elif any(dataset in self.dataset for dataset in robotics_datasets):
            return 'continuous_action'
        elif self.dataset in mcq_datasets:
            return 'multiple_choice'
        elif self.dataset in vqa_datasets:
            return 'visual_question_answering'
        elif self.dataset in image_classification_datasets:
            return 'image_classification'
        elif self.dataset in tool_use_datasets:
            return 'tool_use'
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset}")

def get_dataset_and_dataloader(config: EvaluationConfig) -> tuple:
    # Find data files
    if 'openx' in config.dataset:
        files = find_data_files('openx', config.disk_root_dir, dataset=config.dataset, split=config.data_split)
    elif 'robot_vqa' in config.dataset:
        files = find_data_files('openx', config.disk_root_dir, dataset='openx_multi_embodiment', split=config.data_split)
    elif config.dataset == 'odinw':
        files = []
        for dataset in V1SupportedDatasets().datasets['odinw']:
            file = find_data_files('odinw', config.disk_root_dir, dataset=dataset, split=config.data_split)
            files.extend(file)
    else:
        files = find_data_files(config.dataset, config.disk_root_dir, split=config.data_split)
    
    if len(files) == 0:
        error_msg = f"No data found for dataset {config.dataset} in {config.disk_root_dir}. "
        raise FileNotFoundError(error_msg)
    
    # Get dataloader
    if 'openx' in config.dataset and 'multi_embodiment' not in config.dataset:
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

    if isinstance(dataset, list):
        datasets = dataset
        data_loaders = data_loader
    else:
        datasets = [dataset]
        data_loaders = [data_loader]
    return datasets, data_loaders

def get_metrics_calculator(config: EvaluationConfig, dataset: torch.utils.data.Dataset):
    if 'openx' in config.dataset:
        action_stats = dataset.action_stats
        metrics_calculator = RoboticsMetricsCalculator(action_stats)
    elif config.dataset == 'robot_vqa':
        metrics_calculator = VQAMetricsCalculator()
    elif config.dataset == 'overcooked_ai':
        metrics_calculator = OvercookedAIMetricsCalculator()
    elif config.dataset == 'piqa':
        metrics_calculator = MCQMetricsCalculator(num_choices=2)
    elif config.dataset == 'sqa3d':
        metrics_calculator = VQAMetricsCalculator()
    elif config.dataset == 'odinw':
        num_classes = dataset.get_num_classes()
        metrics_calculator = ClassificationMetricsCalculator(num_classes=num_classes)
    elif config.dataset == 'bfcl':
        metrics_calculator = BFCLMetricsCalculator()
    else:
        raise ValueError(f"Invalid dataset name: {config.dataset}")
    return metrics_calculator
    
def get_ground_truth_key(config: EvaluationConfig) -> str:
    """
    Get the correct ground truth key for each dataset.
    
    Each dataloader returns ground truth values with different keys:
    - OpenX robotics: 'action' (continuous action vector)
    - robot_vqa: 'text_answer' (text string answer)
    - overcooked_ai: 'action' (discrete action index)
    - piqa: 'label' (0 or 1 for binary choice)
    - sqa3d: 'answer' (text string answer)
    - odinw: 'correct_option_idx' (integer index of correct category)
    - bfcl: 'ground_truth_functions' (list of function calls per turn)
    """
    dataset_to_key = {
        # VQA datasets
        'robot_vqa': 'text_answer',
        'sqa3d': 'answer',
        'piqa': 'label',
        'odinw': 'correct_option_idx',
        'bfcl': 'ground_truth_functions',
        'overcooked_ai': 'action',
    }
    
    if config.dataset in dataset_to_key:
        return dataset_to_key[config.dataset]
    
    if 'openx' in config.dataset:
        return 'action'
    return 'action'
        
def load_model_adapter(config: EvaluationConfig) -> ModelAdapter:
    """Load the user's model adapter from the specified module."""
    adapter_path = Path(config.model_adapter_module_path).resolve()
    print(f"Loading model adapter from: {adapter_path}")
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("user_adapter", adapter_path)
    if spec is None:
        raise ImportError(f"Could not load module from {adapter_path}")
        
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

def save_predictions(predictions: List[Dict[str, Any]], config: EvaluationConfig, sub_dataset_name: str = None) -> None:
    """Save predictions to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions in JSONL format
    if sub_dataset_name is None:
        predictions_file = config.predictions_path / f"{config.dataset}_predictions_{timestamp}.jsonl"
    else:
        predictions_file = config.predictions_path / f"{config.dataset}_{sub_dataset_name}_predictions_{timestamp}.jsonl"
    print(f"Saving {config.dataset} {sub_dataset_name if sub_dataset_name is not None else ''} predictions to: {predictions_file}")
    
    with open(predictions_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, default=str) + '\n')

def save_results(metrics: Dict[str, Any], config: EvaluationConfig, sub_dataset_name: Optional[str] = None) -> None:
    """Save generated metrics to file."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    if sub_dataset_name is None:
        results_file = config.output_path / f"{config.dataset}_metrics_{timestamp}.json"
    else:
        results_file = config.output_path / f"{config.dataset}_{sub_dataset_name}_metrics_{timestamp}.json"
    print(f"Saving metrics to: {results_file}")
    
    summary = {
        'evaluation_config': {
            'dataset': config.dataset,
            'task_type': config.task_type,
            'model_adapter_path': str(config.model_adapter_module_path),
            'data_split': config.data_split,
            'sub_dataset_name': sub_dataset_name if sub_dataset_name is not None else '',
            'batch_size': config.batch_size,
            'seed': config.seed,
            'timestamp': timestamp
        },
        'metrics': metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Metrics saved to: {results_file}")

def is_multiturn_dataset(dataset_name: str) -> bool:
    """
    Determine if a dataset requires multi-turn evaluation with conversation history.
    
    Multi-turn datasets require the harness to maintain conversation state across turns
    and pass history to the model adapter.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        True if dataset requires multi-turn evaluation, False otherwise
    """
    multiturn_datasets = ['bfcl']  # Add 'osworld' and others as they are implemented
    return dataset_name in multiturn_datasets

def bordered_print(text: str):
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")

def profile_and_save_results_multiturn(
    model_adapter: ModelAdapter,
    dataset: torch.utils.data.Dataset,
    data_loader: torch.utils.data.DataLoader,
    config: EvaluationConfig,
    sub_dataset_name: Optional[str] = None
):
    """Evaluation function for multi-turn datasets with conversation history."""
    all_predictions = []
    all_ground_truths = []
    
    print(f"Running multi-turn evaluation for {config.dataset}")
    
    for batch_idx, batch in enumerate(data_loader):
        batch_size = len(batch['conversation_id'])
        
        max_turns = max(len(turns) for turns in batch['turns'])
        
        batch_histories = [[] for _ in range(batch_size)]
        batch_predictions = [[] for _ in range(batch_size)]
        batch_ground_truths = [[] for _ in range(batch_size)]
        
        for turn_idx in range(max_turns):
            turn_observations = []
            turn_instructions = []
            turn_histories = []
            active_conv_indices = []
            
            for conv_idx in range(batch_size):
                if turn_idx < len(batch['turns'][conv_idx]):
                    active_conv_indices.append(conv_idx)
                    
                    user_messages_for_turn = batch['turns'][conv_idx][turn_idx]
                    for msg in user_messages_for_turn:
                        batch_histories[conv_idx].append(msg)
                    
                    observation = {
                        'conversation_id': batch['conversation_id'][conv_idx],
                        'turn_index': turn_idx,
                        'initial_config': batch['initial_config'][conv_idx],
                        'involved_classes': batch['involved_classes'][conv_idx],
                        'path': batch['path'][conv_idx],
                    }
                    
                    current_instruction = user_messages_for_turn[-1].get('content', '') if user_messages_for_turn else ''
                    
                    turn_observations.append(observation)
                    turn_instructions.append(current_instruction)
                    turn_histories.append(batch_histories[conv_idx].copy())
            
            if not turn_observations:
                continue
            
            responses = model_adapter.batch_predict_actions(
                observations=turn_observations,
                instructions=turn_instructions,
                dataset_name=config.dataset,
                histories=turn_histories
            )
            
            for i, conv_idx in enumerate(active_conv_indices):
                response = responses[i]
                
                if response and isinstance(response, dict):
                    raw_output = response.get('raw_output', '')
                    
                    if raw_output:
                        batch_histories[conv_idx].append({"role": "assistant", "content": raw_output})
                    
                    batch_predictions[conv_idx].append(response)
                else:
                    response_str = str(response) if response else ''
                    if response_str:
                        batch_histories[conv_idx].append({"role": "assistant", "content": response_str})
                    
                    batch_predictions[conv_idx].append({
                        'raw_output': response_str,
                        'extracted_calls': []
                    })
                
                batch_ground_truths[conv_idx].append(
                    batch['ground_truth_functions'][conv_idx][turn_idx]
                )
        
        for conv_idx in range(batch_size):
            all_predictions.append({
                'conversation_id': batch['conversation_id'][conv_idx],
                'predictions': batch_predictions[conv_idx],
                'num_turns': len(batch_predictions[conv_idx])
            })
            all_ground_truths.append({
                'conversation_id': batch['conversation_id'][conv_idx],
                'ground_truth': batch_ground_truths[conv_idx],
                'num_turns': len(batch_ground_truths[conv_idx])
            })
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches ({len(all_predictions)} conversations)")
    
    print(f"Completed evaluation of {len(all_predictions)} conversations")
    
    # Save predictions
    save_predictions(all_predictions, config, sub_dataset_name=sub_dataset_name)
    
    # Calculate metrics using the appropriate metrics calculator
    metrics_calculator = get_metrics_calculator(config, dataset)
    
    # BFCLMetricsCalculator expects structured multi-turn data
    # Other metrics calculators may need flattened data
    if isinstance(metrics_calculator, BFCLMetricsCalculator):
        # Pass structured multi-turn data directly
        metrics = metrics_calculator.calculate_metrics(all_predictions, all_ground_truths)
    else:
        # Fallback: flatten predictions and ground truths for standard metric calculators
        flattened_predictions = []
        flattened_ground_truths = []
        for pred, gt in zip(all_predictions, all_ground_truths):
            flattened_predictions.extend(pred['predictions'])
            flattened_ground_truths.extend(gt['ground_truth'])
        
        metrics = metrics_calculator.calculate_metrics(flattened_predictions, flattened_ground_truths)
    
    # Add multi-turn specific metadata
    metrics['evaluation_mode'] = 'multi_turn'
    metrics['total_conversations'] = len(all_predictions)
    metrics['total_turns'] = sum(p['num_turns'] for p in all_predictions)
    metrics['avg_turns_per_conversation'] = metrics['total_turns'] / metrics['total_conversations'] if metrics['total_conversations'] > 0 else 0
    
    save_results(metrics, config, sub_dataset_name=sub_dataset_name)

def profile_and_save_results(model_adapter: ModelAdapter,
                             dataset: torch.utils.data.Dataset,
                             data_loader: torch.utils.data.DataLoader, 
                             config: EvaluationConfig,
                             sub_dataset_name: Optional[str] = None):
    predictions = []
    ground_truth_actions = []
    for batch in data_loader:
        batch_predictions = model_adapter.batch_predict_actions(batch)
        predictions.extend(batch_predictions)
        
        ground_truth_actions.extend(batch[get_ground_truth_key(config)])
    save_predictions(predictions, config, sub_dataset_name=sub_dataset_name)

    metrics_calculator = get_metrics_calculator(config, dataset)
    metrics = metrics_calculator.calculate_metrics(predictions, ground_truth_actions)
    
    save_results(metrics, config, sub_dataset_name=sub_dataset_name)
    
def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="MultiNet Model Evaluation Script")
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (overcooked_ai, any supported openx morphology, piqa, sqa3d, robot_vqa, odinw, bfcl)")
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
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help="Device to use for evaluation")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for deterministic evaluation")
    
    args = parser.parse_args()

    # Check if dataset is supported
    supported_datasets = list(V1SupportedDatasets().datasets.keys())
    if args.dataset not in supported_datasets:
        raise ValueError(f"Dataset {args.dataset} is not supported. Supported datasets: {supported_datasets}")

    # Initialize configuration
    config = EvaluationConfig(args)
    print(f"Starting MultiNet evaluation...")
    print(f"Dataset: {config.dataset} (Task type: {config.task_type})")
    print(f"Output path: {config.output_path}")
    print(f"Device: {config.device}")
    
    # Step 1: Load dataset
    bordered_print("LOADING DATASET")

    datasets, data_loaders = get_dataset_and_dataloader(config)

    # Step 2: Load model adapter
    bordered_print("LOADING MODEL ADAPTER")
    
    model_adapter = load_model_adapter(config)
    print(f"Model adapter loaded: {model_adapter.__class__.__name__}")
    
    
    bordered_print("RUNNING EVALUATION")

    # Determine if this is a multi-turn dataset
    is_multiturn = is_multiturn_dataset(config.dataset)
    
    if is_multiturn:
        print(f"Detected multi-turn dataset: {config.dataset}")
        print("Using conversation history management")
        evaluation_function = profile_and_save_results_multiturn
    else:
        evaluation_function = profile_and_save_results

    if len(datasets) > 1:
        for dataset, data_loader in zip(datasets, data_loaders):
            # Get sub-dataset name for multi-dataset evaluation (like ODinW)
            sub_dataset_name = dataset.get_dataset_name() if hasattr(dataset, 'get_dataset_name') else None
            print(f"Running evaluation for {config.dataset} {sub_dataset_name if sub_dataset_name else ''}")
            evaluation_function(model_adapter, dataset, data_loader, config, sub_dataset_name=sub_dataset_name)
            
    else:
        print(f"Running evaluation for {config.dataset}")
        evaluation_function(model_adapter, datasets[0], data_loaders[0], config)
        

    bordered_print("EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()
