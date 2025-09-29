import argparse
import datetime
import json
import os
import sys
import re
import logging
import time
import gc
from dataclasses import dataclass, field, fields
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import jax
import numpy as np
import tensorflow as tf
from src.eval.profiling.openpi.src.openpi.models import pi0
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.models.model import Observation
from src.eval.profiling.openpi.src.openpi.models.tokenizer import PaligemmaTokenizer
from src.eval.profiling.openpi.src.openpi.transforms import pad_to_dim
from src.data_utils.openx_dataloader import get_openx_dataloader
from src.eval.profiling.openpi.src.openpi.shared import download
from src.eval.profiling.openpi.src.openpi.shared import normalize
from src.eval.profiling.openpi.src.openpi.transforms import Unnormalize
from src.eval.profiling.openpi.src.openpi.shared.normalize import RunningStats
from src.eval_utils import (quantile_filter, calculate_mean, min_max_normalize, 
                            calculate_mse, calculate_mae, calculate_max_relative_mae, 
                            calculate_proportion_beyond_mae_threshold)

# Constants
class ModelConfig:
    DEFAULT_ACTION_HORIZON = 1
    DEFAULT_BATCH_SIZE = 16
    DEFAULT_MAX_TOKEN_LENGTH = 48
    DEFAULT_NUM_STEPS = 10
    DEFAULT_ACTION_DIM = 32

class DatasetConfig:
    SHARD_PREFIX = 'translated_shard_'
    RESULTS_FILENAME = 'pi0_base_openx_results.json'
    STATS_FILENAME_TEMPLATE = '{dataset_name}_stats.json'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _binarize_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """
    Converts gripper actions from continuous to binary values (0 and 1).
    Simplified version for numpy arrays.
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    # For values in between, use 0.5 threshold
    return np.where(actions > 0.5, 1.0, 0.0)


def _invert_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """Invert gripper actions: 1 - actions"""
    return 1.0 - actions


def _rel2abs_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).
    Simplified version for numpy arrays.
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1
    thresholded_actions = np.where(opening_mask, 1, np.where(closing_mask, -1, 0))
    
    # Convert to 0 = closed, 1 = open
    new_actions = thresholded_actions / 2 + 0.5
    
    return new_actions


def _create_action_tensor_from_dict(action_dict: Dict[str, np.ndarray], dataset_name: str) -> np.ndarray:
    """
    Create action tensor from action dictionary based on dataset type.
    Follows specific transforms from transforms.py for rt1 and bridge_oxe datasets.
    """
    if action_dict is None:
        return None
    
    action_components = []
    logger.info(f"Processing action_dict for {dataset_name} with keys: {list(action_dict.keys())}")
    
    # RT1 dataset transform (openx_mobile_manipulation)
    # trajectory["action"] = tf.concat((world_vector, rotation_delta, rel2abs_gripper_actions(gripper_closedness_action)), axis=-1)
    if dataset_name == 'openx_mobile_manipulation':
        # RT1 dataset expects: world_vector + rotation_delta + gripper_closedness_action
        if 'world_vector' in action_dict and 'rotation_delta' in action_dict and 'gripper_closedness_action' in action_dict:
            # Add world_vector (3D)
            world_vector = np.array(action_dict['world_vector'])
            if world_vector.ndim == 0:
                world_vector = world_vector.reshape(1)
            elif world_vector.ndim > 1:
                world_vector = world_vector.flatten()
            action_components.append(world_vector)
            logger.info(f"RT1: Added world_vector with shape {world_vector.shape}")
            
            # Add rotation_delta (3D)
            rotation_delta = np.array(action_dict['rotation_delta'])
            if rotation_delta.ndim == 0:
                rotation_delta = rotation_delta.reshape(1)
            elif rotation_delta.ndim > 1:
                rotation_delta = rotation_delta.flatten()
            action_components.append(rotation_delta)
            logger.info(f"RT1: Added rotation_delta with shape {rotation_delta.shape}")
            
            # Add processed gripper_closedness_action (1D) with rel2abs conversion
            gripper_raw = np.array(action_dict['gripper_closedness_action'])
            if gripper_raw.ndim == 0:
                gripper_raw = gripper_raw.reshape(1)
            elif gripper_raw.ndim > 1:
                gripper_raw = gripper_raw.flatten()
            
            # Apply rel2abs_gripper_actions transform (RT1 specific)
            gripper_action = _rel2abs_gripper_actions(gripper_raw)
            if gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1)
            action_components.append(gripper_action)
            logger.info(f"RT1: Added gripper_closedness_action with shape {gripper_action.shape}")
        else:
            logger.warning(f"RT1 dataset missing required keys: world_vector, rotation_delta, gripper_closedness_action")
            return None
    
    # Bridge OXE dataset transform (openx_single_arm)  
    # trajectory["action"] = tf.concat((world_vector, rotation_delta, tf.cast(open_gripper[:, None], tf.float32)), axis=-1)
    elif dataset_name == 'openx_single_arm':
        # Bridge OXE dataset expects: world_vector + rotation_delta + open_gripper
        if 'world_vector' in action_dict and 'rotation_delta' in action_dict and 'open_gripper' in action_dict:
            # Add world_vector (3D)
            world_vector = np.array(action_dict['world_vector'])
            if world_vector.ndim == 0:
                world_vector = world_vector.reshape(1)
            elif world_vector.ndim > 1:
                world_vector = world_vector.flatten()
            action_components.append(world_vector)
            logger.info(f"Bridge OXE: Added world_vector with shape {world_vector.shape}")
            
            # Add rotation_delta (3D)
            rotation_delta = np.array(action_dict['rotation_delta'])
            if rotation_delta.ndim == 0:
                rotation_delta = rotation_delta.reshape(1)
            elif rotation_delta.ndim > 1:
                rotation_delta = rotation_delta.flatten()
            action_components.append(rotation_delta)
            logger.info(f"Bridge OXE: Added rotation_delta with shape {rotation_delta.shape}")
            
            # Add open_gripper (1D) cast to float32
            gripper_raw = np.array(action_dict['open_gripper'])
            if gripper_raw.ndim == 0:
                gripper_raw = gripper_raw.reshape(1)
            elif gripper_raw.ndim > 1:
                gripper_raw = gripper_raw.flatten()
            gripper_action = gripper_raw.astype(np.float32)
            action_components.append(gripper_action)
            logger.info(f"Bridge OXE: Added open_gripper with shape {gripper_action.shape}")
        else:
            logger.warning(f"Bridge OXE dataset missing required keys: world_vector, rotation_delta, open_gripper")
            return None
    
    # Concatenate all action components if we have any
    if action_components:
        action_tensor = np.concatenate(action_components)
        logger.info(f"Created action tensor with shape {action_tensor.shape} for dataset {dataset_name}")
        return action_tensor
    else:
        logger.warning(f"No valid action components found for dataset {dataset_name}")
        return None


def _calculate_batch_metrics(pred_actions, gt_actions, action_stats=None):
    """Calculate MSE and MAE metrics for a batch of continuous actions."""
    if action_stats is None:
        raise ValueError("action_stats is required for proper invalid prediction handling in OpenX evaluation")
    
    mses, maes = [], []
    total_invalid_preds = 0
    
    for i in range(len(pred_actions)):
        pred = np.array(pred_actions[i])
        gt = np.array(gt_actions[i])
        
        # Check for invalid predictions (NaN, inf, or non-numeric values)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)) or pred.size == 0:
            total_invalid_preds += 1
            # Use worst-case MSE/MAE for invalid predictions using dataset stats directly
            max_vals = np.array(action_stats['max'])
            min_vals = np.array(action_stats['min'])
            mse = calculate_mse(max_vals[:len(gt)], min_vals[:len(gt)])
            mae = calculate_mae(max_vals[:len(gt)], min_vals[:len(gt)])
        else:
            mse = calculate_mse(pred, gt)
            mae = calculate_mae(pred, gt)
        
        mses.append(mse)
        maes.append(mae)
    
    return mses, maes, total_invalid_preds

def _calculate_final_metrics(timestep_mses, timestep_maes, action_success):
    """Calculate comprehensive final metrics for OpenX evaluation."""
    result = {}
    
    # Calculate MSE metrics
    total_dataset_mse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_mse = total_dataset_mse / num_timesteps if num_timesteps > 0 else 0.0
    
    # Calculate normalized MSE
    if num_timesteps > 1:
        normalized_mses = min_max_normalize(timestep_mses)
        normalized_amse = calculate_mean(normalized_mses)
    else:
        normalized_amse = 0.0
    
    # Calculate MAE metrics
    total_dataset_mae = sum(timestep_maes)
    avg_dataset_mae = calculate_mean(timestep_maes)
    
    if num_timesteps > 1:
        normalized_maes = min_max_normalize(timestep_maes)
        normalized_amae = calculate_mean(normalized_maes)
        
        # Calculate quantile filtered MAE metrics
        quantile_filtered_maes = quantile_filter(timestep_maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
        normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
        
        # Calculate additional MAE metrics
        max_rel_mae = calculate_max_relative_mae(timestep_maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
    else:
        normalized_amae = 0.0
        normalized_quantile_filtered_amae = 0.0
        max_rel_mae = 0.0
        prop_beyond_threshold_mae = 0.0
    
    # Calculate action success rate
    action_success_rate = None
    if len(action_success) > 0:
        action_success_rate = (sum(action_success) / len(action_success)) * 100
    
    result['action_success_rate'] = action_success_rate
    result['total_dataset_amse'] = total_dataset_mse
    result['total_dataset_amae'] = total_dataset_mae
    result['num_timesteps'] = num_timesteps
    result['avg_dataset_amse'] = avg_dataset_mse
    result['avg_dataset_amae'] = avg_dataset_mae
    result['normalized_amse'] = normalized_amse
    result['normalized_amae'] = normalized_amae
    result['normalized_quantile_filtered_amae'] = normalized_quantile_filtered_amae
    result['max_relative_mae'] = max_rel_mae
    result['proportion_beyond_threshold_mae'] = prop_beyond_threshold_mae
    
    return result


@dataclass
class DatasetResults:
    all_preds: list[list[float]] = field(default_factory=list)
    all_gt: list[list[float]] = field(default_factory=list)
    timestep_mses: list[float] = field(default_factory=list)
    timestep_maes: list[float] = field(default_factory=list)
    action_success: list[int] = field(default_factory=list)

    total_batches: int = 0
    total_timesteps: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    invalid_predictions_percentage: float = 0.0
    

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }




class OpenXInference:
    def __init__(self, model, tokenizer: PaligemmaTokenizer, config: pi0.Pi0Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def prepare_observation(self, batch: dict, batch_size: int,
                           action_dim: int = ModelConfig.DEFAULT_ACTION_DIM,
                           max_token_length: int = ModelConfig.DEFAULT_MAX_TOKEN_LENGTH,
                           dataset_name: str = None) -> dict:
        """Prepare observation dictionary for model inference."""
        try:
            # Process images
            base_image, zero_image = self._prepare_images(batch)

            # Process text observations
            tokens, token_mask = self._prepare_text_tokens(batch)

            # Process state
            state = self._prepare_state(batch, batch_size, dataset_name)

            return {
                "state": state,
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": zero_image,
                    "right_wrist_0_rgb": zero_image
                },
                "image_mask": {
                    "base_0_rgb": jax.numpy.ones(base_image.shape[0], dtype=bool),
                    "left_wrist_0_rgb": jax.numpy.zeros(base_image.shape[0], dtype=bool),
                    "right_wrist_0_rgb": jax.numpy.zeros(base_image.shape[0], dtype=bool)
                },
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask
            }
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            raise

    def _prepare_images(self, batch: dict) -> tuple:
        """Prepare image observations for model input."""
        base_image = jax.numpy.array(np.stack(batch["image_observation"]))
        zero_image = jax.numpy.zeros_like(base_image)
        return base_image, zero_image

    def _prepare_text_tokens(self, batch: dict) -> tuple:
        """Prepare text tokens and masks for model input."""
        text_obs = batch["text_observation"]
        if isinstance(text_obs, tf.Tensor):
            text_obs = text_obs.numpy().decode('utf-8')

        tokens_list = []
        token_mask_list = []

        text_list = text_obs if isinstance(text_obs, list) else [text_obs]

        for text in text_list:
            text = text if text is not None else ""
            tokens, token_mask = self.tokenizer.tokenize(text)
            tokens_list.append(tokens)
            token_mask_list.append(token_mask)

        tokens = jax.numpy.array(tokens_list)
        token_mask = jax.numpy.array(token_mask_list)

        # Add batch dimension if needed
        if len(tokens.shape) == 1:
            tokens = tokens[None, ...]
            token_mask = token_mask[None, ...]

        return tokens, token_mask

    def _prepare_state(self, batch: dict, batch_size: int, dataset_name: str = None) -> jax.Array:
        """Prepare state vector for model input with robust dimension handling."""
        state_components = []
        
        # Dataset-specific state field mappings for datasets without explicit 'state' key
        # These fields were selected based on OpenX dataset survey - they represent robot state
        # (pose, position, sensor readings) rather than action commands or visual observations
        dataset_state_fields = {
            'openx_bimanual': ['pose_l', 'pose_r'],  # Left/right arm poses (12 dims: 6+6)
            'openx_mobile_manipulation': ['gripper_closed', 'base_pose_tool_reached', 'height_to_bottom', 'yaw', 'orientation_box']  # Robot state sensors (~16 dims)
        }
        
        logger.info(f"Batch keys: {list(batch.keys())}")

        # First check for explicit 'state' key (highest priority)
        if 'state' in batch:
            state_values = batch['state']
            if isinstance(state_values, list) and len(state_values) > 0:
                try:
                    # Handle list of numpy arrays from state key
                    state_arrays = []
                    for state_val in state_values:
                        if isinstance(state_val, np.ndarray) and state_val.dtype in [np.float32, np.float64]:
                            # Flatten to 1D if needed
                            flat_state = state_val.flatten()
                            state_arrays.append(flat_state)
                    
                    if state_arrays:
                        # Stack into (batch_size, features) shape
                        state_array = np.stack(state_arrays)
                        state_components.append(state_array)
                        logger.info(f"Using explicit 'state' key with shape {state_array.shape}")
                except Exception as e:
                    logger.warning(f"Failed to process 'state' key: {e}")

        # Fallback: use dataset-specific state fields for known datasets
        if not state_components and dataset_name in dataset_state_fields:
            logger.info(f"No explicit 'state' key found, using dataset-specific fields for {dataset_name}")
            target_fields = dataset_state_fields[dataset_name]
            
            for field_name in target_fields:
                if field_name in batch:
                    values = batch[field_name]
                    if isinstance(values, list) and len(values) > 0:
                        if (isinstance(values[0], np.ndarray) 
                            and values[0].dtype in [np.float32, np.float64]):
                            try:
                                state_array = np.stack(values)
                                # Flatten multi-dimensional arrays to 2D (batch, features)
                                if len(state_array.shape) > 2:
                                    state_array = state_array.reshape(batch_size, -1)
                                elif len(state_array.shape) == 1:
                                    state_array = state_array[:, np.newaxis]
                                state_components.append(state_array)
                                logger.info(f"Added dataset-specific state field '{field_name}' with shape {state_array.shape}")
                            except ValueError as e:
                                logger.warning(f"Skipping state field '{field_name}' due to dimension mismatch: {e}")
                                continue
                else:
                    logger.warning(f"Expected state field '{field_name}' not found in batch for dataset {dataset_name}")

        # Use real state data if available, otherwise fallback to minimal dummy state
        if state_components:
            # Concatenate all state components
            state = np.concatenate(state_components, axis=-1)
            logger.info(f"Using real state data with shape {state.shape}")
        else:
            logger.warning(f"No valid state data found for dataset {dataset_name}, using minimal dummy state")
            state = jax.numpy.zeros((batch_size, 1))

        # Handle state dimension overflow/underflow relative to action dimension
        current_state_dim = state.shape[-1]
        target_action_dim = self.config.action_dim
        
        if current_state_dim > target_action_dim:
            # Truncate if state is larger than action dimension
            logger.warning(f"State dimension {current_state_dim} exceeds action dimension {target_action_dim}, truncating")
            state = state[:, :target_action_dim]
        elif current_state_dim < target_action_dim:
            # Pad if state is smaller than action dimension
            logger.info(f"Padding state from {current_state_dim} to {target_action_dim} dimensions")
            state = pad_to_dim(state, target_action_dim, axis=-1)
            
        return jax.numpy.array(state)

    def _cleanup_memory(self, *objects):
        """Clean up memory between batches."""
        try:
            # Delete references to large objects
            for obj in objects:
                if hasattr(obj, '__del__'):
                    del obj

            # Force garbage collection
            gc.collect()

            # Clear JAX caches
            jax.clear_caches()
        except Exception as e:
            logger.warning(f"Error during memory cleanup: {e}")

    def process_output(self, actions, dataset_stats: dict, dataset_name: str):
        """Process model outputs with dynamic action dimension handling."""
        actions = np.array(actions)

        # Flatten if needed (remove extra dimensions like timesteps)
        if len(actions.shape) == 3 and actions.shape[1] == 1:
            actions = actions[:, 0, :]  # Remove timestep dimension

        # Get the actual action dimension from dataset stats
        if hasattr(dataset_stats['action'], 'mean'):
            dataset_action_dim = len(dataset_stats['action'].mean)
        else:
            dataset_action_dim = dataset_stats['action']['size'][0]

        # Dynamic clipping: use min of model output dim and dataset dim
        model_action_dim = actions.shape[-1]
        effective_action_dim = min(model_action_dim, dataset_action_dim)

        logger.info(f"Dataset: {dataset_name}, Model dim: {model_action_dim}, "
                   f"Dataset dim: {dataset_action_dim}, Using: {effective_action_dim}")

        # Slice to effective dimension
        clipped_actions = actions[:, :effective_action_dim]

        # Pad dataset stats if needed for unnormalization
        if effective_action_dim < dataset_action_dim:
            # Use only the first N dimensions of dataset stats
            clipped_stats = self._clip_normalization_stats(dataset_stats, effective_action_dim)
        else:
            clipped_stats = dataset_stats

        unnormalizer = Unnormalize(norm_stats=clipped_stats)
        unnormalized_actions = unnormalizer({'action': clipped_actions})['action']

        # Return raw unnormalized actions for comparison (no structured parsing)
        return unnormalized_actions

    def _clip_normalization_stats(self, dataset_stats: dict, target_dim: int) -> dict:
        """Clip normalization statistics to target dimension."""
        original_stats = dataset_stats['action']

        if hasattr(original_stats, 'mean'):
            # NormStats object
            clipped_mean = original_stats.mean[:target_dim]
            clipped_std = original_stats.std[:target_dim]
            clipped_q01 = original_stats.q01[:target_dim] if original_stats.q01 is not None else None
            clipped_q99 = original_stats.q99[:target_dim] if original_stats.q99 is not None else None

            clipped_stats = normalize.NormStats(
                mean=clipped_mean,
                std=clipped_std,
                q01=clipped_q01,
                q99=clipped_q99
            )
        else:
            # Dictionary stats
            clipped_stats = normalize.NormStats(
                mean=np.array(original_stats['mean'][:target_dim]),
                std=np.array(original_stats['std'][:target_dim]),
                q01=np.array(original_stats['q01'][:target_dim]) if original_stats.get('q01') else None,
                q99=np.array(original_stats['q99'][:target_dim]) if original_stats.get('q99') else None
            )

        return {'action': clipped_stats}

    def evaluate_model(self, model, key, config, dataset_stats: dict, dataloader: tf.data.Dataset, dataset: str, dataset_stats_dict: dict = None) -> dict[any]:
        counter = 0
        dataset_results = DatasetResults()
        start_time = time.perf_counter()

        for batch in dataloader:
            actual_batch_size = len(batch['image_observation'])
            obs = self.prepare_observation(batch, actual_batch_size, max_token_length=config.max_token_len, dataset_name=dataset)
            observation = Observation.from_dict(obs)

            actions = model.sample_actions(key, observation, num_steps=ModelConfig.DEFAULT_NUM_STEPS)
            processed_actions = self.process_output(actions, dataset_stats, dataset)

            # Process ground truth actions - check for action_dict first
            gt_actions = batch['action']
            
            # Check if action_dict is available and dataset requires special processing
            if ('action_dict' in batch and batch['action_dict'] is not None and 
                len(batch['action_dict']) > 0):
                action_dicts = batch['action_dict']
                
                # Check if this is a dataset that requires action dict processing
                if dataset in ['openx_mobile_manipulation', 'openx_single_arm']:
                    logger.info(f"Processing action_dict for dataset {dataset}")
                    processed_gt_actions = []
                    
                    for i, action_dict in enumerate(action_dicts):
                        if action_dict is not None and len(action_dict) > 0:
                            # Create action tensor from dictionary
                            action_tensor = _create_action_tensor_from_dict(action_dict, dataset)
                            if action_tensor is not None:
                                processed_gt_actions.append(action_tensor)
                                logger.info(f"Sample {i}: Created action tensor with shape {action_tensor.shape}")
                            else:
                                # Fallback to original action if dict processing fails
                                processed_gt_actions.append(np.array(gt_actions[i]))
                                logger.warning(f"Sample {i}: Falling back to original action")
                        else:
                            # Use original action if dict is None or empty
                            processed_gt_actions.append(np.array(gt_actions[i]))
                    
                    if processed_gt_actions:
                        gt_actions = processed_gt_actions
                        logger.info(f"Successfully processed {len(processed_gt_actions)} actions from action_dict")
                    else:
                        logger.warning("No valid actions processed from action_dict, using original actions")
            
            # Convert to numpy array format
            gt_actions = np.array([np.array(action) for action in gt_actions])

            # Dynamic action comparison - clip GT to match prediction dimension
            pred_actions = np.array(processed_actions)
            effective_dim = min(pred_actions.shape[1], gt_actions.shape[1])

            # Clip both to effective dimension for fair comparison
            pred_clipped = pred_actions[:, :effective_dim]
            gt_clipped = gt_actions[:, :effective_dim]

            logger.info(f"Comparing actions - Pred shape: {pred_clipped.shape}, GT shape: {gt_clipped.shape}")

            # Store predictions and ground truth
            dataset_results.all_preds.extend(pred_clipped.tolist())
            dataset_results.all_gt.extend(gt_clipped.tolist())

            # Calculate MSE and MAE metrics for this batch directly
            # Use dataset stats dictionary directly
            if dataset_stats_dict is None:
                raise ValueError("dataset_stats_dict is required for OpenX evaluation")
            
            mses, maes, invalid_preds = _calculate_batch_metrics(pred_clipped.tolist(), gt_clipped.tolist(), dataset_stats_dict)
            dataset_results.total_invalid_predictions += invalid_preds
            
            # Store the MSE and MAE values for this batch
            dataset_results.timestep_mses.extend(mses)
            dataset_results.timestep_maes.extend(maes)
            
            # Check for action success (exact match for actions)
            # For continuous actions, we use exact match as success criterion
            for i in range(pred_clipped.shape[0]):
                if np.array_equal(pred_clipped[i], gt_clipped[i]):
                    dataset_results.action_success.append(1)
                else:
                    dataset_results.action_success.append(0)
            

            dataset_results.total_batches += 1
            dataset_results.total_timesteps += actual_batch_size

            counter += 1
            logger.info(f"Processed batch {counter}")

            # Memory cleanup between batches
            self._cleanup_memory(obs, processed_actions, pred_clipped, gt_clipped)

        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time

        # Calculate final comprehensive metrics using the accumulated MSE/MAE data
        final_metrics = _calculate_final_metrics(dataset_results.timestep_mses, dataset_results.timestep_maes, dataset_results.action_success)
        
        # Calculate invalid predictions percentage
        if dataset_results.total_timesteps > 0:
            invalid_percentage = (dataset_results.total_invalid_predictions / dataset_results.total_timesteps) * 100
        else:
            invalid_percentage = 0.0
        
        # Add evaluation metadata
        final_metrics['eval_time'] = dataset_results.eval_time
        final_metrics['total_invalid_preds'] = dataset_results.total_invalid_predictions
        final_metrics['invalid_predictions_percentage'] = invalid_percentage
        final_metrics['total_batches'] = dataset_results.total_batches
        final_metrics['total_timesteps'] = dataset_results.total_timesteps

        return final_metrics


def _get_sorted_shard_paths(dataset_dir: str) -> list[str]:
    """Get sorted shard paths using robust regex parsing."""
    shard_pattern = re.compile(r'translated_shard_(\d+)$')
    shard_dirs = []

    try:
        # Check if data is in a test subdirectory (common OpenX structure)
        test_dir = os.path.join(dataset_dir, 'test')
        search_dir = test_dir if os.path.exists(test_dir) else dataset_dir

        for dirname in os.listdir(search_dir):
            match = shard_pattern.match(dirname)
            if match:
                shard_num = int(match.group(1))
                full_path = os.path.join(search_dir, dirname)
                shard_dirs.append((shard_num, full_path))
    except OSError as e:
        logger.error(f"Error reading directory {dataset_dir}: {e}")
        raise

    # Sort by shard number and return paths
    shard_dirs.sort(key=lambda x: x[0])
    return [path for _, path in shard_dirs]

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with output_dir and dataset_dir
    """
    parser = argparse.ArgumentParser(
        description="Run inference on OpenX datasets"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to store results and dataset statistics'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Root directory containing the openx datasets'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num_shards',
        type=int,
        default=None,
        help='Number of shards to process. If None, all shards are processed.'
    )

    args = parser.parse_args()

    # Validate paths exist
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    return args

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\nResults will be stored in: {args.output_dir}')
    print(f'\nReading datasets from: {args.dataset_dir}')
    if args.num_shards:
        print(f'\nProcessing up to {args.num_shards} shards per dataset.')

    config = pi0.Pi0Config(action_horizon=ModelConfig.DEFAULT_ACTION_HORIZON)
    tokenizer = PaligemmaTokenizer()
    key = jax.random.key(0)
    model = config.load(_model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")))
    logger.info('Model loaded')
    openx_inference = OpenXInference(model, tokenizer, config)

    results_file = os.path.join(args.output_dir, DatasetConfig.RESULTS_FILENAME)

    # Extract dataset name from the dataset directory path
    dataset_name = os.path.basename(args.dataset_dir.rstrip('/'))
    print(f'\n ---- EVALUATING {dataset_name} ---- \n')

    openx_dataset_dir = args.dataset_dir

    if not os.path.exists(openx_dataset_dir):
        print(f"Dataset directory not found: {openx_dataset_dir}")
        return

    shard_paths = _get_sorted_shard_paths(openx_dataset_dir)
    if args.num_shards:
        shard_paths = shard_paths[:args.num_shards]

    dataset, dataloader = get_openx_dataloader(shard_paths, args.batch_size, dataset_name)

    # Get dataset stats from the dataset object
    dataset_stats_dict = dataset.action_stats
    dataset_stats = {'action': normalize.NormStats(**dataset_stats_dict)}

    # Determine dynamic action dimension based on ground truth actions
    dynamic_action_dim = None
    sample_batch = None
    try:
        # Get first batch to analyze action dimensions
        for batch in dataloader:
            sample_batch = batch
            break
        
        if sample_batch is not None:
            # Process sample actions to determine correct action dimension
            sample_gt_actions = sample_batch['action']
            
            # Check if action_dict processing is needed for this dataset
            if ('action_dict' in sample_batch and sample_batch['action_dict'] is not None and 
                len(sample_batch['action_dict']) > 0 and 
                dataset_name in ['openx_mobile_manipulation', 'openx_single_arm']):
                logger.info(f"Analyzing action_dict for dynamic action dimension for dataset {dataset_name}")
                
                # Process first action_dict to get dimension
                action_dicts = sample_batch['action_dict']
                for action_dict in action_dicts:
                    if action_dict is not None and len(action_dict) > 0:
                        action_tensor = _create_action_tensor_from_dict(action_dict, dataset_name)
                        if action_tensor is not None:
                            dynamic_action_dim = len(action_tensor)
                            logger.info(f"Determined dynamic action dimension: {dynamic_action_dim} from action_dict")
                            break
            
            # Fallback to using regular action dimension
            if dynamic_action_dim is None:
                sample_action = np.array(sample_gt_actions[0])
                dynamic_action_dim = len(sample_action)
                logger.info(f"Determined dynamic action dimension: {dynamic_action_dim} from regular action")
            
            logger.info(f"Using dynamic action dimension: {dynamic_action_dim}")
            
        # Reload model with dynamic action dimension if different from default
        if dynamic_action_dim is not None and dynamic_action_dim != config.action_dim:
            logger.info(f"Reloading model with action_dim={dynamic_action_dim} (was {config.action_dim})")
            config = pi0.Pi0Config(action_horizon=ModelConfig.DEFAULT_ACTION_HORIZON, action_dim=dynamic_action_dim)
            model = config.load(_model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")))
            openx_inference = OpenXInference(model, tokenizer, config)
            
        # Recreate dataloader since we consumed the first batch
        dataset, dataloader = get_openx_dataloader(shard_paths, args.batch_size, dataset_name)
        
    except Exception as e:
        logger.warning(f"Failed to determine dynamic action dimension: {e}")
        logger.info("Using default model configuration")

    # Save stats to a file (with comprehensive numpy array conversion)
    def convert_numpy_arrays(obj):
        """Recursively convert numpy arrays and TensorFlow tensors to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'numpy'):  # Handle TensorFlow EagerTensor objects
            return obj.numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_arrays(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # Handle other array-like objects
            return obj.tolist()
        else:
            return obj

    stats_output_path = os.path.join(args.output_dir,
                                   DatasetConfig.STATS_FILENAME_TEMPLATE.format(dataset_name=dataset_name))
    with open(stats_output_path, 'w') as f:
        json.dump(convert_numpy_arrays(dataset_stats_dict), f, indent=4)
    logger.info(f'Dataset stats saved to {stats_output_path}')

    results = openx_inference.evaluate_model(model, key, config, dataset_stats, dataloader, dataset_name, dataset_stats_dict)

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                dataset_results = json.load(f)
        except Exception as e:
            raise Exception(f"Result file might be corrupted. Please delete it and run the script again. {e}")
    else:
        dataset_results = {}

    dataset_results[dataset_name] = results

    with open(results_file, 'w') as f:
        json.dump(dataset_results, f, indent=4)



if __name__ == "__main__":
    main()
