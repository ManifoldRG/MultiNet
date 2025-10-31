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
import numpy as np
import logging
import tensorflow as tf
logger = logging.getLogger(__name__)

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import evaluation harness components
from src.eval_harness.model_adapter import ModelAdapter
from src.eval_harness.v1_supported_datasets import V1SupportedDatasets
from src.data_utils import *
from src.eval_harness.scoring import *
from definitions.overcooked import OverCookedDefinitions
from definitions.openx import OpenXDefinitions

def first_non_none_shape(images_list):
    """Return (H, W) of the first non-None image; None if all None."""
    for img in images_list:
        if img is not None:
            return img.shape[:2]
    return None


def ensure_image_or_placeholder(img, ref_shape):
    """Return img if present; else a black placeholder of ref_shape (H, W, 3)."""
    if img is None:
        if ref_shape is None:
            # Should not happen if we gate earlier; keep defensive check
            raise ValueError("No reference shape available for placeholder")
        h, w = ref_shape
        print(f"Warning: Missing image; substituting placeholder of shape ({h}, {w})")
        return np.zeros((h, w, 3), dtype=np.uint8)
    return img


def first_non_none_shape(images_list):
    """Return (H, W) of the first non-None image; None if all None."""
    for img in images_list:
        if img is not None:
            return img.shape[:2]
    return None


def _get_meaningful_action_dims(dataset_name: str) -> int:
    """
    Get the number of meaningful action dimensions for OpenX datasets.

    Args:
        dataset_name: OpenX dataset name (e.g., 'openx_wheeled_robot')

    Returns:
        Number of meaningful dimensions from ACTION_SPACES definition
    """
    openx_subtasks_mapping = {
        'openx_wheeled_robot': 'berkeley_gnm_sac_son',
        'openx_quadrupedal': 'utokyo_saytap_converted_externally_to_rlds',
        'openx_single_arm': 'bridge',
        'openx_bimanual': 'utokyo_xarm_bimanual_converted_externally_to_rlds',
        'openx_mobile_manipulation': 'fractal20220817_data'
    }

    if dataset_name not in openx_subtasks_mapping:
        return None

    subtask = openx_subtasks_mapping[dataset_name]
    action_space = OpenXDefinitions.ACTION_SPACES[subtask]['default']

    # Count non-None dimensions
    meaningful_dims = sum(1 for v in action_space.values() if v is not None)
    return meaningful_dims if meaningful_dims > 0 else None


def validate_structured_prediction(pred: Any, dataset_name: str) -> Dict[str, Any]:
    """
    Validate that a prediction has the correct structured format.
    
    Args:
        pred: Prediction from model adapter
        dataset_name: Name of the dataset
        
    Returns:
        Validated prediction dictionary
        
    Raises:
        ValueError: If prediction format is invalid
    """
    if not isinstance(pred, dict):
        raise ValueError(
            f"Expected prediction to be a dict with 'raw_output' and 'extracted_outputs', "
            f"but got {type(pred).__name__}"
        )
    
    if "raw_output" not in pred:
        raise ValueError(
            f"Prediction dict must contain 'raw_output' key. Got keys: {list(pred.keys())}"
        )
    
    if "extracted_outputs" not in pred:
        raise ValueError(
            f"Prediction dict must contain 'extracted_outputs' key. Got keys: {list(pred.keys())}"
        )
    
    if not isinstance(pred["raw_output"], str):
        raise ValueError(
            f"'raw_output' must be a string, got {type(pred['raw_output']).__name__}"
        )
    
    return pred


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
        self.batch_process = args.batch_process
        self.max_samples = args.max_samples
        
        # Override batch_size when batch_process=False
        if not self.batch_process:
            self.batch_size = 1
        
        # Clip batch_size to max_samples if max_samples is set
        if self.max_samples is not None:
            self.batch_size = min(self.batch_size, self.max_samples)
        
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

def _convert_action_dict_to_tensor(action_dict: dict, dataset_name: str) -> np.ndarray:
    """
    Convert action dictionary to tensor for data format conversion.
    This is NOT a transformation for model compatibility, just data format conversion.
    
    Args:
        action_dict: Action dictionary with keys like world_vector, rotation_delta, etc.
        dataset_name: Name of the dataset
        
    Returns:
        Action tensor (7D for both mobile_manipulation and single_arm)
    """
    action_components = []
    
    if dataset_name == 'openx_mobile_manipulation':
        if 'world_vector' in action_dict and 'rotation_delta' in action_dict and 'gripper_closedness_action' in action_dict:
            # Add world_vector (3D)
            world_vector = np.array(action_dict['world_vector'])
            if world_vector.ndim == 0:
                world_vector = world_vector.reshape(1)
            elif world_vector.ndim > 1:
                world_vector = world_vector.flatten()
            action_components.append(world_vector)
            
            # Add rotation_delta (3D)
            rotation_delta = np.array(action_dict['rotation_delta'])
            if rotation_delta.ndim == 0:
                rotation_delta = rotation_delta.reshape(1)
            elif rotation_delta.ndim > 1:
                rotation_delta = rotation_delta.flatten()
            action_components.append(rotation_delta)
            
            # Add gripper_closedness_action (1D) - convert relative to absolute
            gripper_raw = np.array(action_dict['gripper_closedness_action'])
            if gripper_raw.ndim == 0:
                gripper_raw = gripper_raw.reshape(1)
            elif gripper_raw.ndim > 1:
                gripper_raw = gripper_raw.flatten()
            
            # Convert relative to absolute: -1 for closing, 1 for opening -> 0 = closed, 1 = open
            opening_mask = gripper_raw < -0.1
            closing_mask = gripper_raw > 0.1
            thresholded_actions = np.where(opening_mask, 1, np.where(closing_mask, -1, 0))
            gripper_action = thresholded_actions / 2 + 0.5
            
            if gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1)
            action_components.append(gripper_action)
        else:
            raise KeyError(f"RT1 dataset missing required keys: world_vector, rotation_delta, gripper_closedness_action")
            
    elif dataset_name == 'openx_single_arm':
        if 'world_vector' in action_dict and 'rotation_delta' in action_dict and 'open_gripper' in action_dict:
            # Add world_vector (3D)
            world_vector = np.array(action_dict['world_vector'])
            if world_vector.ndim == 0:
                world_vector = world_vector.reshape(1)
            elif world_vector.ndim > 1:
                world_vector = world_vector.flatten()
            action_components.append(world_vector)
            
            # Add rotation_delta (3D)
            rotation_delta = np.array(action_dict['rotation_delta'])
            if rotation_delta.ndim == 0:
                rotation_delta = rotation_delta.reshape(1)
            elif rotation_delta.ndim > 1:
                rotation_delta = rotation_delta.flatten()
            action_components.append(rotation_delta)
            
            # Add open_gripper (1D) cast to float32
            gripper_raw = np.array(action_dict['open_gripper'])
            if gripper_raw.ndim == 0:
                gripper_raw = gripper_raw.reshape(1)
            elif gripper_raw.ndim > 1:
                gripper_raw = gripper_raw.flatten()
            gripper_action = gripper_raw.astype(np.float32)
            action_components.append(gripper_action)
        else:
            raise KeyError(f"Bridge OXE dataset missing required keys: world_vector, rotation_delta, open_gripper")
    
    if action_components:
        action_tensor = np.concatenate(action_components)
        return action_tensor
    else:
        raise ValueError(f"No valid action components found for dataset {dataset_name}")

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
            'batch_process': config.batch_process,
            'max_samples': config.max_samples,
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
                    
                    # Pass text_observation from batch['prompt'] as persistent context
                    observation = {
                        'text_observation': batch['prompt'][conv_idx]  # Persistent context from dataloader
                    }
                    
                    # Extract instruction from the last user message
                    current_instruction = user_messages_for_turn[-1].get('content', '') if user_messages_for_turn else ''
                    
                    turn_observations.append(observation)
                    turn_instructions.append(current_instruction)
                    # Pass history WITHOUT current turn's user messages
                    turn_histories.append(batch_histories[conv_idx].copy())
            
            if not turn_observations:
                continue
            
            responses = model_adapter.batch_predict_actions(
                observations=turn_observations,
                instructions=turn_instructions,
                dataset_name=config.dataset,
                histories=turn_histories
            )
            
            # Validate responses have correct structured format
            validated_responses = []
            for response in responses:
                try:
                    validated_response = validate_structured_prediction(response, config.dataset)
                    validated_responses.append(validated_response)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid prediction format from model adapter for dataset '{config.dataset}': {e}"
                    )
            
            for i, conv_idx in enumerate(active_conv_indices):
                response = validated_responses[i]
                
                # Add user messages from current turn to history
                user_messages_for_turn = batch['turns'][conv_idx][turn_idx]
                batch_histories[conv_idx].extend(user_messages_for_turn)
                
                if response and isinstance(response, dict):
                    raw_output = response.get('raw_output', '')
                    
                    # Add assistant response to history
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
        
        # Early exit logic for max_samples
        if config.max_samples is not None:
            if len(all_predictions) >= config.max_samples:
                print(f"Processed {len(all_predictions)} conversations (max_samples={config.max_samples}). Exiting.")
                break
    
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

def extract_observations_and_instructions(
    batch: Dict[str, Any],
    dataset_name: str,
    dataset: torch.utils.data.Dataset
) -> tuple:
    """
    Extract standardized observations and instructions from batch.
    
    Returns:
        observations: List of dicts with 'text_observation' and/or 'image_observation', or None for PIQA
        instructions: List of instruction strings, or None if dataset has no instructions
    """
    # Determine batch size from a known key
    if 'openx' in dataset_name or dataset_name == 'robot_vqa' or dataset_name == 'overcooked_ai':
        batch_size = len(batch['image_observation'])
    elif dataset_name == 'piqa':
        batch_size = len(batch['question'])
    elif dataset_name == 'sqa3d':
        batch_size = len(batch['scene_image'])
    elif dataset_name == 'odinw':
        batch_size = len(batch['image'])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    observations = []
    instructions = []
    
    if 'openx' in dataset_name:
        openx_subtasks_mapping= {
            'openx_wheeled_robot': 'berkeley_gnm_sac_son',
            'openx_quadrupedal': 'utokyo_saytap_converted_externally_to_rlds',
            'openx_single_arm': 'bridge',
            'openx_bimanual': 'utokyo_xarm_bimanual_converted_externally_to_rlds',
            'openx_mobile_manipulation': 'fractal20220817_data'
        }
        
        for i in range(batch_size):
            obs_dict = {}
            env_desc = batch['text_observation'][i].strip()
            obs_dict['text_observation'] = env_desc

            inst_dict = OpenXDefinitions.DESCRIPTIONS[openx_subtasks_mapping[dataset_name]]

            instruction = inst_dict.get(env_desc, inst_dict.get(env_desc.lower().rstrip('.'), None))
            instructions.append(instruction)

            action_space = OpenXDefinitions.ACTION_SPACES[openx_subtasks_mapping[dataset_name]]['default']
            obs_dict['options'] = action_space

            action_stats = dataset.action_stats
            # Convert TensorFlow tensors to numpy arrays
            for key in ['min', 'max', 'mean', 'std', 'q01', 'q99']:
                if key in action_stats:
                    if tf.is_tensor(action_stats[key]):
                        action_stats[key] = action_stats[key].numpy()
                    else:
                        action_stats[key] = np.array(action_stats[key])
            
            obs_dict['action_stats'] = action_stats

            obs_dict['image_observation'] = batch['image_observation'][i]

            observations.append(obs_dict)

    elif dataset_name == 'robot_vqa':
        # Robot VQA: instruction is the question, observation only has image
        # Check if all images are missing
        ref = first_non_none_shape(batch['image_observation'])
        if ref is None:
            print(f"Skipping batch: no images available for robot_vqa")
            return [], []  # Return empty lists to skip this batch
        
        for i in range(batch_size):
            img = ensure_image_or_placeholder(batch['image_observation'][i], ref)
            observations.append({
                'image_observation': img,
            })
            instructions.append(batch['text_observation'][i])
    
    elif dataset_name == 'piqa':
        # PIQA: instruction is the question, observation has options only (text-only task)
        for i in range(batch_size):
            observations.append({
                'options': [0, 1],  # PIQA always has 2 choices
            })
            instructions.append(batch['question'][i])
    
    elif dataset_name == 'sqa3d':
        # SQA3D: instruction is the question, observation only has image
        # Check if all images are missing
        ref = first_non_none_shape(batch['scene_image'])
        if ref is None:
            print(f"Skipping batch: no images available for SQA3D")
            return [], []  # Return empty lists to skip this batch
        
        for i in range(batch_size):
            img = ensure_image_or_placeholder(batch['scene_image'][i], ref)
            observations.append({
                'image_observation': img,
            })
            instructions.append(batch['question'][i])
    
    elif dataset_name == 'odinw':
        # ODinW: instruction is the question, observation has image and options
        # Check if all images are missing
        ref = first_non_none_shape(batch['image'])
        if ref is None:
            print(f"Skipping batch: no images available for odinw")
            return [], []  # Return empty lists to skip this batch
        
        for i in range(batch_size):
            img = ensure_image_or_placeholder(batch['image'][i], ref)
            observations.append({
                'image_observation': img,
                'options': batch['options'][i]
            })
            instructions.append(batch['question'][i])
    
    elif dataset_name == 'overcooked_ai':
        # Overcooked: observation has layout + time info
        # Check if all images are missing
        ref = first_non_none_shape(batch['image_observation'])
        if ref is None:
            print(f"Skipping batch: no images available for overcooked_ai")
            return [], []  # Return empty lists to skip this batch
        
        for i in range(batch_size):
            # Format text observation with time information
            text_obs = batch['text_observation'][i]
            time_left = batch['time_left'][i]
            time_elapsed = batch['time_elapsed'][i]

            img = ensure_image_or_placeholder(batch['image_observation'][i], ref)
            observations.append({
                'text_observation': OverCookedDefinitions.ACTION_MEANINGS,
                'image_observation': img,
                'options': OverCookedDefinitions.ACTION_SPACES['overcooked_ai']['default']
            })

            # Combine into single text observation for instruction
            combined_text = f"Layout: {text_obs}\nTime left: {time_left:.1f}s\nTime elapsed: {time_elapsed:.1f}s"
            
            instructions.append(combined_text)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return observations, instructions

def profile_and_save_results(
    model_adapter: ModelAdapter,
    dataset: torch.utils.data.Dataset,
    data_loader: torch.utils.data.DataLoader, 
    config: EvaluationConfig,
    sub_dataset_name: Optional[str] = None
):
    predictions = []
    ground_truth_actions = []
    skipped_batches_no_image = 0
    skipped_samples_no_image = 0
    
    for batch in data_loader:
        # Extract standardized observations and instructions
        observations, instructions = extract_observations_and_instructions(
            batch, config.dataset, dataset
        )
        
        # Skip this batch if no observations (all images missing)
        if len(observations) == 0:
            skipped_batches_no_image += 1
            
            # Count samples based on dataset-specific image field
            if config.dataset == 'sqa3d':
                skipped_samples_no_image += len(batch['scene_image'])
            elif config.dataset == 'robot_vqa':
                skipped_samples_no_image += len(batch['image_observation'])
            elif config.dataset == 'odinw':
                skipped_samples_no_image += len(batch['image'])
            elif config.dataset == 'overcooked_ai':
                skipped_samples_no_image += len(batch['image_observation'])
            else:
                # For other datasets, use a fallback
                skipped_samples_no_image += len(batch.get('image_observation', []))
            
            continue
        
        if config.batch_process:
            # Use batch processing
            batch_predictions = model_adapter.batch_predict_actions(
                observations=observations,
                instructions=instructions,
                dataset_name=config.dataset
            )
            
            # Validate predictions have correct structured format
            validated_predictions = []
            for pred in batch_predictions:
                try:
                    validated_pred = validate_structured_prediction(pred, config.dataset)
                    validated_predictions.append(validated_pred)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid prediction format from model adapter for dataset '{config.dataset}': {e}"
                    )
        else:
            # Use single-item processing
            validated_predictions = []
            for i in range(len(observations)):
                # Get single observation and instruction
                single_observation = observations[i]
                single_instruction = instructions[i] if instructions else None
                
                # Call predict_action for single item
                single_prediction = model_adapter.predict_action(
                    observation=single_observation,
                    instruction=single_instruction,
                    dataset_name=config.dataset
                )
                
                # Validate prediction has correct structured format
                try:
                    validated_pred = validate_structured_prediction(single_prediction, config.dataset)
                    validated_predictions.append(validated_pred)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid prediction format from model adapter for dataset '{config.dataset}': {e}"
                    )
        
        predictions.extend(validated_predictions)

        # Process ground truth actions for OpenX datasets
        if 'openx' in config.dataset:
            # Get raw actions from batch
            raw_gt_actions = batch['action']

            # Process based on dataset type
            if config.dataset in ['openx_mobile_manipulation', 'openx_single_arm']:
                # Convert action_dict to tensor (data format conversion, not transformation)
                processed_gt_actions = []
                for i, action_dict in enumerate(batch.get('action_dict', [])):
                    if action_dict and len(action_dict) > 0:
                        action_tensor = _convert_action_dict_to_tensor(action_dict, config.dataset)
                        processed_gt_actions.append(action_tensor)
                    else:
                        processed_gt_actions.append(np.array(raw_gt_actions[i]))
                ground_truth_actions.extend(processed_gt_actions)

            else:
                # No processing needed
                ground_truth_actions.extend(raw_gt_actions)
            
            # Convert all ground truth actions to numpy array format
            ground_truth_actions = [np.array(action) for action in ground_truth_actions]
        else:
            # Non-OpenX datasets use existing logic
            ground_truth_actions.extend(batch[get_ground_truth_key(config)])

        # Early exit logic for max_samples
        if config.max_samples is not None:
            if len(predictions) >= config.max_samples:
                print(f"Processed {len(predictions)} samples (max_samples={config.max_samples}). Exiting.")
                break

    save_predictions(predictions, config, sub_dataset_name=sub_dataset_name)

    # Check if no predictions were generated (all batches skipped)
    if len(predictions) == 0:
        print(f"No predictions generated (all batches skipped due to missing images). Skipping metrics.")
        print(f"Skipped {skipped_batches_no_image} batches ({skipped_samples_no_image} samples) due to missing images.")
        
        # Create a summary with skipped counts instead of metrics
        summary = {
            'evaluation_config': {
                'dataset': config.dataset,
                'task_type': config.task_type,
                'model_adapter_path': str(config.model_adapter_module_path),
                'data_split': config.data_split,
                'sub_dataset_name': sub_dataset_name if sub_dataset_name is not None else '',
                'batch_size': config.batch_size,
                'batch_process': config.batch_process,
                'max_samples': config.max_samples,
                'seed': config.seed,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            'skipped_summary': {
                'skipped_batches_no_image': skipped_batches_no_image,
                'skipped_samples_no_image': skipped_samples_no_image,
                'reason': 'All batches skipped due to missing images'
            }
        }
        
        # Save the summary instead of metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if sub_dataset_name is None:
            results_file = config.output_path / f"{config.dataset}_skipped_summary_{timestamp}.json"
        else:
            results_file = config.output_path / f"{config.dataset}_{sub_dataset_name}_skipped_summary_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Skipped summary saved to: {results_file}")
        return

    metrics_calculator = get_metrics_calculator(config, dataset)

    # Clip to meaningful dimensions for OpenX datasets
    if 'openx' in config.dataset:
        meaningful_dims = _get_meaningful_action_dims(config.dataset)
        if meaningful_dims is not None:

            # Clip predictions
            clipped_predictions = []
            for pred in predictions:
                clipped_pred = pred.copy()
                clipped_pred['extracted_outputs'] = pred['extracted_outputs'][:meaningful_dims]
                clipped_predictions.append(clipped_pred)

            # Clip ground truth
            clipped_ground_truth = [gt[:meaningful_dims] for gt in ground_truth_actions]

            metrics = metrics_calculator.calculate_metrics(clipped_predictions, clipped_ground_truth)
        else:
            metrics = metrics_calculator.calculate_metrics(predictions, ground_truth_actions)
    else:
        metrics = metrics_calculator.calculate_metrics(predictions, ground_truth_actions)
    
    # Add skipped batch information to metrics if any batches were skipped
    if skipped_batches_no_image > 0:
        metrics['skipped_batches_no_image'] = skipped_batches_no_image
        metrics['skipped_samples_no_image'] = skipped_samples_no_image
    
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
    parser.add_argument('--batch_process', action='store_true', default=False,
                        help="Use batch processing (batch_predict_actions). If False, uses single-item processing (predict_action) with batch_size=1")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to process. If set, clips batch_size to max_samples. If batch_process=True, exits after one batch. If batch_process=False, exits after processing max_samples.")
    
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

    # Validate BFCL requires batch processing
    if config.dataset == 'bfcl' and not config.batch_process:
        raise ValueError("BFCL dataset requires batch_process=True (single-item processing not yet supported for multi-turn datasets)")
    
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
            print(f"Total samples in dataset: {len(dataset)}")
            evaluation_function(model_adapter, dataset, data_loader, config, sub_dataset_name=sub_dataset_name)

    else:
        print(f"Running evaluation for {config.dataset}")
        print(f"Total samples in dataset: {len(datasets[0])}")
        evaluation_function(model_adapter, datasets[0], data_loaders[0], config)

    bordered_print("EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()
