import argparse
import datetime
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from src.eval.profiling.openpi.src.openpi.models import pi0
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.models.model import Observation
from src.eval.profiling.openpi.src.openpi.models.tokenizer import PaligemmaTokenizer
from src.eval.profiling.openpi.src.openpi.transforms import pad_to_dim, Unnormalize
from src.data_utils.overcooked_dataloader import get_overcooked_dataloader, OvercookedDataset
from src.eval.profiling.openpi.src.openpi.shared import download
from src.eval.profiling.openpi.src.openpi.shared.normalize import NormStats
from src.eval_utils import *
import jax
import numpy as np
import tensorflow as tf
import gc
from dataclasses import dataclass, field, fields
import time


# Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")
# Configure JAX memory settings
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


@dataclass
class DatasetResults:
    all_preds: list[int] = field(default_factory=list)
    all_gt: list[int] = field(default_factory=list)
    
    total_batches: int = 0
    total_timesteps: int = 0
    eval_time: float = 0
    exact_match_rate: float = 0
    player0_accuracy: float = 0
    player1_accuracy: float = 0
    joint_accuracy: float = 0
    
    # Additional metrics (will be set dynamically)
    total_micro_precision: float = 0
    total_micro_recall: float = 0
    total_micro_f1: float = 0
    total_macro_precision: float = 0
    total_macro_recall: float = 0
    total_macro_f1: float = 0
    total_mae: float = 0
    total_mse: float = 0
    total_invalid_predictions: int = 0
    
    # Final averaged metrics (calculated at the end)
    micro_precision: float = 0
    micro_recall: float = 0
    micro_f1: float = 0
    macro_precision: float = 0
    macro_recall: float = 0
    macro_f1: float = 0
    mae: float = 0
    mse: float = 0
    invalid_predictions_percentage: float = 0
    
    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class OvercookedInference:
    def __init__(self, model, tokenizer: PaligemmaTokenizer, config: pi0.Pi0Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Create action mapping for continuous to discrete conversion
        self._create_action_mapping()

    def _create_action_mapping(self):
        temp_dataset = OvercookedDataset.__new__(OvercookedDataset)
        temp_dataset._create_discrete_action_mapping()
        
        self.discrete_to_joint = temp_dataset.discrete_to_joint
        self.joint_to_discrete = temp_dataset.joint_to_discrete
        self.num_discrete_actions = temp_dataset.num_discrete_actions

    def prepare_observation(self, obs_dict: dict, batch_size: int, max_token_length: int = 48) -> dict:
        """Prepare observation dictionary for pi0 model inference."""
        tokenizer = self.tokenizer
        
        # Process image observation
        base_image = obs_dict["image_observation"]
        if isinstance(base_image[0], np.ndarray):
            base_image = np.array(base_image)
        
        # Add batch dimension if needed
        if len(base_image.shape) == 3:
            base_image = base_image[None, ...]
            
        # Convert to jax array
        base_image = jax.numpy.array(base_image)
        zero_image = jax.numpy.zeros_like(base_image)
        
        # Process text observation (layout name)
        text_obs = obs_dict["text_observation"]
        
        # Tokenize text prompt/observation
        if isinstance(text_obs, list):
            tokens = [0] * len(text_obs)
            token_mask = [0] * len(text_obs)
            for i in range(len(text_obs)):
                tokens[i], token_mask[i] = tokenizer.tokenize(text_obs[i])
                tokens[i] = jax.numpy.array(tokens[i])
                token_mask[i] = jax.numpy.array(token_mask[i])
        else:
            tokens, token_mask = tokenizer.tokenize(text_obs)
        
        if not isinstance(tokens, list) and len(tokens.shape) == 1:  
            tokens = jax.numpy.array(tokens)[None, ...]
            token_mask = jax.numpy.array(token_mask)[None, ...]
        else:
            tokens = jax.numpy.array(tokens)
            token_mask = jax.numpy.array(token_mask)
        
        # Create observation dictionary (Pi0 expects 32-dimensional state)
        transformed_dict = {
            "state": jax.numpy.zeros((batch_size, 32)),  # Dummy 32-dim state for Pi0
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
        
        return transformed_dict

    def process_output(self, actions: jax.numpy.ndarray, dataset_stats: dict) -> np.ndarray:
        """
        Process Pi0 action outputs using normalization statistics.
        
        Args:
            actions: Pi0 continuous actions of shape (batch_size, action_horizon, 32)
            dataset_stats: Dataset normalization statistics for unnormalization
            
        Returns:
            np.ndarray: Discrete action indices (0-35)
        """
        # Convert to numpy
        actions = np.array(actions)
        
        # Get only first dimension and preserve batch dimensions 
        actions = actions[..., 0:1]
        
        # Apply unnormalization using dataset statistics
        if 'action' in dataset_stats:
            unnormalizer = Unnormalize(norm_stats=dataset_stats)
            unnormalized_actions = unnormalizer({'action': actions})['action']
        else:
            unnormalized_actions = actions
        
        # Discretize the actions 
        discrete_actions = np.round(unnormalized_actions, 0).astype(int)
        
        # Clip to valid range [0, num_discrete_actions-1]
        discrete_actions = np.clip(discrete_actions, 0, self.num_discrete_actions - 1)
        
        return discrete_actions

    
    def get_dataset_stats(self, dataloader, dataset_name: str = "overcooked"):
        """Calculate normalization statistics for Overcooked discrete actions."""
        # For Overcooked, we need to create normalization stats for our 36 discrete actions (0-35)
        # This maps the continuous model output to our discrete action space
        
        # Collect all discrete actions from the dataset
        all_actions = []
        for batch in dataloader:
            all_actions.extend(batch['action'])
        
        all_actions = np.array(all_actions)
        print(f"Collected {len(all_actions)} actions for statistics")
        print(f"Action range in dataset: {all_actions.min()}-{all_actions.max()}")
        print(f"Unique actions: {len(np.unique(all_actions))}")
        
        # Create normalization statistics that will map continuous outputs to discrete range
        # Structure needs to match what Unnormalize expects (action key with NormStats)
        
        action_mean = all_actions.mean()
        action_std = all_actions.std() if all_actions.std() > 0 else 1.0  # Avoid division by zero
        
        norm_stats = NormStats(
            mean=np.array([action_mean]),
            std=np.array([action_std])
        )
        
        dataset_stats = {
            'action': norm_stats
        }
        
        print(f"Normalization stats: mean={action_mean:.2f}, std={action_std:.2f}")
        
        return dataset_stats, {}

    def evaluate_model(self, model, key, config, dataloader, dataset_name: str, output_dir: str = None, max_samples: int = None) -> dict:
        """Evaluate the model on the Overcooked dataset."""
        counter = 0
        dataset_results = DatasetResults()

        # Calculate normalization statistics from dataset
        print("Calculating dataset normalization statistics...")
        dataset_stats, _ = self.get_dataset_stats(dataloader, dataset_name)

        # Calculate total batches for progress tracking
        total_batches = len(dataloader)
        print(f"Starting evaluation: {total_batches} total batches")

        start_time = time.perf_counter()
        samples_processed = 0

        for batch in dataloader:
            actual_batch_size = len(batch['image_observation'])
            # Check if we've reached max_samples limit
            if max_samples is not None and samples_processed + actual_batch_size > max_samples:
                # Process only remaining samples from this batch
                remaining = max_samples - samples_processed
                if remaining <= 0:
                    break
                # Truncate batch to remaining samples
                for key in batch:
                    if isinstance(batch[key], list):
                        batch[key] = batch[key][:remaining]
                    elif hasattr(batch[key], '__len__'):
                        batch[key] = batch[key][:remaining]
                actual_batch_size = remaining
            obs = {
                'image_observation': batch['image_observation'],
                'text_observation': batch['text_observation']
            }
            # Transform observation
            transformed_dict = self.prepare_observation(obs, max_token_length=config.max_token_len, batch_size=actual_batch_size)
            observation = Observation.from_dict(transformed_dict)
            
            # Sample actions for entire batch
            actions = model.sample_actions(key, observation, num_steps=1)
            normalized_actions = self.process_output(actions, dataset_stats)
            
            counter += 1
            samples_processed += actual_batch_size
            
            # Progress reporting
            if counter % 100 == 0 or counter == total_batches or (max_samples and samples_processed >= max_samples):
                elapsed_time = time.perf_counter() - start_time
                print(f"Progress: {counter}/{total_batches} batches ({counter/total_batches*100:.1f}%) - {elapsed_time:.1f}s elapsed")
            
            # Get ground truth actions from batch
            gt_actions = np.array(batch['action'])
            
            # Flatten normalized_actions for metric calculation
            flat_predictions = normalized_actions.flatten() if normalized_actions.ndim > 1 else normalized_actions
            flat_gt = gt_actions.flatten() if gt_actions.ndim > 1 else gt_actions
            
            # Calculate basic metrics
            emr = get_exact_match_rate(flat_predictions, flat_gt)
            action_space = list(range(self.num_discrete_actions))
            
            # Calculate micro metrics
            total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
                flat_predictions, flat_gt, action_space
            )
            micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
            micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
            micro_f1 = get_micro_f1(micro_precision, micro_recall)
            
            # Calculate macro metrics
            class_precisions = get_precision_per_class(flat_predictions, flat_gt, action_space)
            class_recalls = get_recall_per_class(flat_predictions, flat_gt, action_space)
            class_f1s = get_f1_per_class(class_precisions, class_recalls)
            macro_precision = get_macro_precision(class_precisions)
            macro_recall = get_macro_recall(class_recalls)
            macro_f1 = get_macro_f1(class_f1s)
            
            # Calculate MAE and MSE
            mae = calculate_mae(flat_predictions, flat_gt)
            mse = calculate_mse(flat_predictions, flat_gt)
            
            # Calculate per-player accuracy for Overcooked-specific analysis
            batch_p0_matches = 0
            batch_p1_matches = 0
            batch_joint_matches = 0
            
            for i in range(len(flat_predictions)):
                pred_action_idx = flat_predictions[i]
                gt_action_idx = flat_gt[i]
                
                if pred_action_idx == gt_action_idx:
                    batch_joint_matches += 1
                
                # Convert to joint actions for per-player analysis
                if pred_action_idx < self.num_discrete_actions and gt_action_idx < self.num_discrete_actions:
                    pred_joint = self.discrete_to_joint[pred_action_idx]
                    gt_joint = self.discrete_to_joint[gt_action_idx]
                    
                    # Per-player accuracy
                    if pred_joint[0] == gt_joint[0]:  # Player 0
                        batch_p0_matches += 1
                    if pred_joint[1] == gt_joint[1]:  # Player 1
                        batch_p1_matches += 1
            
            total_predictions = len(flat_predictions)
            
            # Store batch results
            dataset_results.all_preds.extend([int(x) for x in flat_predictions.tolist()])
            dataset_results.all_gt.extend([int(x) for x in flat_gt.tolist()])
            dataset_results.total_batches = counter
            dataset_results.total_timesteps += total_predictions
            dataset_results.exact_match_rate += emr * total_predictions
            dataset_results.player0_accuracy += batch_p0_matches
            dataset_results.player1_accuracy += batch_p1_matches
            dataset_results.joint_accuracy += batch_joint_matches
            
            # Store additional metrics
            dataset_results.total_micro_precision += micro_precision * total_predictions
            dataset_results.total_micro_recall += micro_recall * total_predictions
            dataset_results.total_micro_f1 += micro_f1 * total_predictions
            dataset_results.total_macro_precision += macro_precision * total_predictions
            dataset_results.total_macro_recall += macro_recall * total_predictions
            dataset_results.total_macro_f1 += macro_f1 * total_predictions
            dataset_results.total_mae += mae * total_predictions
            dataset_results.total_mse += mse * total_predictions
            dataset_results.total_invalid_predictions += int(invalid_fp)

            # Save intermediate results every 500 batches
            if counter % 500 == 0:
                intermediate_results = dataset_results.to_dict()
                intermediate_results['eval_time'] = time.perf_counter() - start_time
                
                # Convert raw counts to rates for the main fields (to match final output format)
                total_ts = max(1, intermediate_results['total_timesteps'])
                intermediate_results['exact_match_rate'] = intermediate_results['exact_match_rate'] / total_ts
                intermediate_results['player0_accuracy'] = intermediate_results['player0_accuracy'] / total_ts
                intermediate_results['player1_accuracy'] = intermediate_results['player1_accuracy'] / total_ts
                intermediate_results['joint_accuracy'] = intermediate_results['joint_accuracy'] / total_ts
                
                # Calculate additional metrics
                intermediate_results['micro_precision'] = intermediate_results.get('total_micro_precision', 0) / total_ts
                intermediate_results['micro_recall'] = intermediate_results.get('total_micro_recall', 0) / total_ts
                intermediate_results['micro_f1'] = intermediate_results.get('total_micro_f1', 0) / total_ts
                intermediate_results['macro_precision'] = intermediate_results.get('total_macro_precision', 0) / total_ts
                intermediate_results['macro_recall'] = intermediate_results.get('total_macro_recall', 0) / total_ts
                intermediate_results['macro_f1'] = intermediate_results.get('total_macro_f1', 0) / total_ts
                intermediate_results['mae'] = intermediate_results.get('total_mae', 0) / total_ts
                intermediate_results['mse'] = intermediate_results.get('total_mse', 0) / total_ts
                intermediate_results['invalid_predictions_percentage'] = intermediate_results.get('total_invalid_predictions', 0) / total_ts * 100
                
                # Keep avg_ fields for backward compatibility
                intermediate_results['avg_emr'] = intermediate_results['exact_match_rate']
                intermediate_results['avg_player0_accuracy'] = intermediate_results['player0_accuracy']
                intermediate_results['avg_player1_accuracy'] = intermediate_results['player1_accuracy']
                intermediate_results['avg_joint_accuracy'] = intermediate_results['joint_accuracy']
                
                intermediate_file = os.path.join(output_dir, f'intermediate_results_batch_{counter}.json')
                with open(intermediate_file, 'w') as f:
                    json.dump({'overcooked': intermediate_results}, f, indent=4)
                print(f"Saved intermediate results at batch {counter}")

            # Memory management
            del transformed_dict, observation, actions, normalized_actions, gt_actions
            gc.collect()
            jax.clear_caches()
            
            # Break if we've reached max_samples
            if max_samples is not None and samples_processed >= max_samples:
                print(f"Reached max_samples limit: {samples_processed}/{max_samples}")
                break

        end_time = time.perf_counter()
        eval_duration = end_time - start_time
        dataset_results.eval_time = eval_duration
        
        # Calculate final averages
        if dataset_results.total_timesteps > 0:
            dataset_results.exact_match_rate /= dataset_results.total_timesteps
            dataset_results.player0_accuracy /= dataset_results.total_timesteps
            dataset_results.player1_accuracy /= dataset_results.total_timesteps
            dataset_results.joint_accuracy /= dataset_results.total_timesteps
            
            # Calculate additional metrics
            dataset_results.micro_precision = dataset_results.total_micro_precision / dataset_results.total_timesteps
            dataset_results.micro_recall = dataset_results.total_micro_recall / dataset_results.total_timesteps
            dataset_results.micro_f1 = dataset_results.total_micro_f1 / dataset_results.total_timesteps
            dataset_results.macro_precision = dataset_results.total_macro_precision / dataset_results.total_timesteps
            dataset_results.macro_recall = dataset_results.total_macro_recall / dataset_results.total_timesteps
            dataset_results.macro_f1 = dataset_results.total_macro_f1 / dataset_results.total_timesteps
            dataset_results.mae = dataset_results.total_mae / dataset_results.total_timesteps
            dataset_results.mse = dataset_results.total_mse / dataset_results.total_timesteps
            dataset_results.invalid_predictions_percentage = dataset_results.total_invalid_predictions / dataset_results.total_timesteps * 100

        return dataset_results.to_dict()


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Pi0 inference on Overcooked datasets"
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to store results'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='Path to Overcooked pickle data file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate paths exist
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
        
    return args


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\nResults will be stored in: {args.output_dir}')
    print(f'Reading data from: {args.data_file}')
    
    # Initialize model and inference object
    config = pi0.Pi0Config(action_horizon=1)
    tokenizer = PaligemmaTokenizer()
    key = jax.random.key(0)
    checkpoint_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
    params = _model.restore_params(checkpoint_path)
    model = config.load(params)
    overcooked_inference = OvercookedInference(model, tokenizer, config)
    
    # Enable diagonal move logging for testing
    overcooked_inference._log_raw_outputs = True
    overcooked_inference._diagonal_count = 0

    results_file = os.path.join(args.output_dir, 'pi0_base_overcooked_results.json')
    raw_data_file = os.path.join(args.output_dir, 'raw_predictions_gt.json')

    # Create dataloader
    dataset_obj, dataloader = get_overcooked_dataloader(args.data_file, batch_size=args.batch_size, by_episode=False)
    
    # Run evaluation
    dataset_name = "overcooked"
    print(f'\n ---- EVALUATING {dataset_name} ---- \n')
    if args.max_samples:
        print(f'Limiting evaluation to {args.max_samples} samples')
    results = overcooked_inference.evaluate_model(model, key, config, dataloader, dataset_name, args.output_dir, args.max_samples)
    
    # Save results
    results_data = {dataset_name: results}
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=4)
    
    # Save raw predictions and ground truth for analysis
    raw_data = {
        'predictions': results['all_preds'],
        'ground_truth': results['all_gt'],
        'metadata': {
            'total_timesteps': results['total_timesteps'],
            'total_batches': results['total_batches'],
            'eval_time': results['eval_time'],
            'dataset_name': dataset_name,
            'batch_size': args.batch_size
        }
    }
    
    with open(raw_data_file, 'w') as f:
        json.dump(raw_data, f, indent=4)
    
    print(f'\nResults saved to: {results_file}')
    print(f'Raw data saved to: {raw_data_file}')
    print(f'Exact Match Rate: {results["exact_match_rate"]:.4f}')
    print(f'Player 0 Accuracy: {results["player0_accuracy"]:.4f}')
    print(f'Player 1 Accuracy: {results["player1_accuracy"]:.4f}')
    print(f'Joint Accuracy: {results["joint_accuracy"]:.4f}')
    if 'micro_precision' in results:
        print(f'Micro Precision: {results["micro_precision"]:.4f}')
        print(f'Micro Recall: {results["micro_recall"]:.4f}')
        print(f'Micro F1: {results["micro_f1"]:.4f}')
        print(f'Macro Precision: {results["macro_precision"]:.4f}')
        print(f'Macro Recall: {results["macro_recall"]:.4f}')
        print(f'Macro F1: {results["macro_f1"]:.4f}')
        print(f'MAE: {results["mae"]:.4f}')
        print(f'MSE: {results["mse"]:.4f}')
        print(f'Invalid Predictions %: {results["invalid_predictions_percentage"]:.2f}%')
    print(f'Total timesteps: {results["total_timesteps"]}')
    print(f'Evaluation time: {results["eval_time"]:.2f} seconds')


if __name__ == "__main__":
    main()
