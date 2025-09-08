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
from src.eval_utils import (get_exact_match_rate,
                            calculate_tp_fp_fn_counts,
                            get_micro_precision_from_counts,
                            get_micro_recall_from_counts,
                            get_micro_f1)

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

class ActionComponents:
    WORLD_VECTOR_DIM = 3
    ROTATION_DELTA_DIM = 3
    OPEN_GRIPPER_DIM = 1
    TERMINATE_EPISODE_DIM = 1

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetResults:
    all_preds: list[list[float]] = field(default_factory=list)
    all_gt: list[list[float]] = field(default_factory=list)

    total_batches: int = 0
    total_timesteps: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    invalid_predictions_percentage: float = 0
    total_emr: float = 0
    total_micro_precision: float = 0
    total_micro_recall: float = 0
    total_micro_f1: float = 0
    avg_emr: float = 0
    avg_micro_precision: float = 0
    avg_micro_recall: float = 0
    avg_micro_f1: float = 0
    total_clipped_emr: float = 0
    total_clipped_micro_precision: float = 0
    total_clipped_micro_recall: float = 0
    total_clipped_micro_f1: float = 0
    avg_clipped_emr: float = 0
    avg_clipped_micro_precision: float = 0
    avg_clipped_micro_recall: float = 0
    avg_clipped_micro_f1: float = 0
    total_micro_precision_without_invalids: float = 0
    total_micro_f1_without_invalids: float = 0
    avg_micro_precision_without_invalids: float = 0
    avg_micro_f1_without_invalids: float = 0

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
                           max_token_length: int = ModelConfig.DEFAULT_MAX_TOKEN_LENGTH) -> dict:
        """Prepare observation dictionary for model inference."""
        try:
            # Process images
            base_image, zero_image = self._prepare_images(batch)

            # Process text observations
            tokens, token_mask = self._prepare_text_tokens(batch)

            # Process state
            state = self._prepare_state(batch, batch_size)

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

    def _prepare_state(self, batch: dict, batch_size: int) -> jax.Array:
        """Prepare state vector for model input with robust dimension handling."""
        state_components = []
        excluded_keys = {'image_observation', 'text_observation', 'action', 'reward', 'is_last', 'text_answer'}
        
        logger.info(f"Batch keys: {list(batch.keys())}")

        # First check for 'state' key specifically (from current dataloader)
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
                        logger.info(f"Using 'state' key with shape {state_array.shape}")
                except Exception as e:
                    logger.warning(f"Failed to process 'state' key: {e}")

        # Fallback: process other non-excluded keys that contain state data
        if not state_components:
            for key, values in batch.items():
                if key not in excluded_keys and isinstance(values, list):
                    if (len(values) > 0 and isinstance(values[0], np.ndarray)
                        and values[0].dtype in [np.float32, np.float64]):
                        try:
                            state_array = np.stack(values)
                            # Flatten multi-dimensional arrays to 2D (batch, features)
                            if len(state_array.shape) > 2:
                                state_array = state_array.reshape(batch_size, -1)
                            elif len(state_array.shape) == 1:
                                state_array = state_array[:, np.newaxis]
                            state_components.append(state_array)
                            logger.info(f"Added state component '{key}' with shape {state_array.shape}")
                        except ValueError as e:
                            logger.warning(f"Skipping state component '{key}' due to dimension mismatch: {e}")
                            continue

        # Use real state data if available, otherwise fallback to minimal dummy state
        if state_components:
            # Concatenate all state components
            state = np.concatenate(state_components, axis=-1)
            logger.info(f"Using real state data with shape {state.shape}")
        else:
            logger.warning("No valid state data found, using minimal dummy state")
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

    def evaluate_model(self, model, key, config, dataset_stats: dict, dataloader: tf.data.Dataset, dataset: str) -> dict[any]:
        counter = 0
        dataset_results = DatasetResults()
        start_time = time.perf_counter()

        for batch in dataloader:
            actual_batch_size = len(batch['image_observation'])
            obs = self.prepare_observation(batch, actual_batch_size, max_token_length=config.max_token_len)
            observation = Observation.from_dict(obs)

            actions = model.sample_actions(key, observation, num_steps=ModelConfig.DEFAULT_NUM_STEPS)
            processed_actions = self.process_output(actions, dataset_stats, dataset)

            gt_actions = batch['action']
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

            # Calculate metrics on clipped actions
            mse = np.mean((pred_clipped - gt_clipped) ** 2)
            mae = np.mean(np.abs(pred_clipped - gt_clipped))

            # Use normalized metrics
            gt_variance = np.var(gt_clipped) + 1e-8
            normalized_mse = mse / gt_variance
            pseudo_accuracy = np.exp(-normalized_mse)  # Convert to [0,1] range where 1 is perfect

            dataset_results.total_emr += pseudo_accuracy
            dataset_results.total_micro_precision += pseudo_accuracy
            dataset_results.total_micro_recall += pseudo_accuracy
            dataset_results.total_micro_f1 += pseudo_accuracy

            dataset_results.total_batches += 1
            dataset_results.total_timesteps += actual_batch_size

            counter += 1
            logger.info(f"Processed batch {counter}")

            # Memory cleanup between batches
            self._cleanup_memory(obs, processed_actions, pred_clipped, gt_clipped)

        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time

        # Calculate average metrics
        if dataset_results.total_timesteps > 0:
            dataset_results.avg_emr = dataset_results.total_emr / dataset_results.total_timesteps
            dataset_results.avg_micro_precision = dataset_results.total_micro_precision / dataset_results.total_timesteps
            dataset_results.avg_micro_recall = dataset_results.total_micro_recall / dataset_results.total_timesteps
            dataset_results.avg_micro_f1 = dataset_results.total_micro_f1 / dataset_results.total_timesteps

        return dataset_results.to_dict()


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

    results = openx_inference.evaluate_model(model, key, config, dataset_stats, dataloader, dataset_name)

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
