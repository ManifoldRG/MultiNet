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
from src.eval.profiling.openpi.src.openpi.transforms import pad_to_dim
from src.data_utils.overcooked_dataloader import get_overcooked_dataloader
from src.eval.profiling.openpi.src.openpi.shared import download
import jax
import numpy as np
import tensorflow as tf
import gc
from dataclasses import dataclass, field, fields
import time


#Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")
# Configure JAX memory settings
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
        """Create mapping from continuous Pi0 output to discrete Overcooked actions."""
        # Define individual actions for each player (same as in overcooked_dataloader.py)
        single_actions = [
            (0, -1),   # NORTH
            (0, 1),    # SOUTH  
            (1, 0),    # EAST
            (-1, 0),   # WEST
            (0, 0),    # STAY
            (1, 1)     # INTERACT
        ]
        
        # Create discrete to joint action mapping
        self.discrete_to_joint = {}
        self.joint_to_discrete = {}
        
        action_idx = 0
        for p0_action in single_actions:
            for p1_action in single_actions:
                joint_action = (tuple(p0_action), tuple(p1_action))
                self.discrete_to_joint[action_idx] = joint_action
                self.joint_to_discrete[joint_action] = action_idx
                action_idx += 1

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

    def process_output(self, actions: jax.numpy.ndarray) -> np.ndarray:
        """
        Convert continuous Pi0 action outputs to discrete Overcooked joint actions.
        
        Args:
            actions: Pi0 continuous actions of shape (batch_size, action_horizon, 32)
            
        Returns:
            np.ndarray: Discrete joint action indices
        """
        # Convert to numpy and extract first 4 dimensions
        actions = np.array(actions)
        # Extract first 4 dimensions: [p0_x, p0_y, p1_x, p1_y]
        relevant_actions = actions[..., :4]
        
        discrete_actions = []
        for batch_idx in range(relevant_actions.shape[0]):
            batch_discrete_actions = []
            for step_idx in range(relevant_actions.shape[1]):
                # Extract player actions
                p0_continuous = relevant_actions[batch_idx, step_idx, :2]  # [x, y] for player 0
                p1_continuous = relevant_actions[batch_idx, step_idx, 2:4]  # [x, y] for player 1
                
                # Convert continuous to discrete actions
                p0_discrete = self._continuous_to_discrete_action(p0_continuous)
                p1_discrete = self._continuous_to_discrete_action(p1_continuous)
                
                # Convert to joint action index
                joint_action = (tuple(p0_discrete), tuple(p1_discrete))
                discrete_action_idx = self.joint_to_discrete.get(joint_action, 0)  # Default to (STAY, STAY)
                
                batch_discrete_actions.append(discrete_action_idx)
            discrete_actions.append(batch_discrete_actions)
        
        return np.array(discrete_actions)

    def _continuous_to_discrete_action(self, continuous_action: np.ndarray) -> tuple:
        """Convert continuous 2D action to discrete Overcooked action."""
        x, y = continuous_action
        
        # Define thresholds for action mapping
        threshold = 0.5
        
        # Check for interact action first (high magnitude in both dimensions)
        if abs(x) > threshold and abs(y) > threshold:
            return (1, 1)   # INTERACT
        
        # Map to discrete actions based on thresholds
        if abs(x) > abs(y):  # Horizontal movement dominant
            if x > threshold:
                return (1, 0)   # EAST
            elif x < -threshold:
                return (-1, 0)  # WEST
            else:
                return (0, 0)   # STAY
        else:  # Vertical movement dominant
            if y > threshold:
                return (0, 1)   # SOUTH
            elif y < -threshold:
                return (0, -1)  # NORTH
            else:
                return (0, 0)   # STAY

    def evaluate_model(self, model, key, config, dataloader, dataset_name: str, output_dir: str = None) -> dict:
        """Evaluate the model on the Overcooked dataset."""
        counter = 0
        dataset_results = DatasetResults()

        # Calculate total batches for progress tracking
        total_batches = len(dataloader)
        print(f"Starting evaluation: {total_batches} total batches")

        start_time = time.perf_counter()

        for batch in dataloader:
            # Process entire batch at once
            actual_batch_size = len(batch['image_observation'])
            obs = {
                'image_observation': batch['image_observation'],
                'text_observation': batch['text_observation']
            }
            # Transform observation
            transformed_dict = self.prepare_observation(obs, max_token_length=config.max_token_len, batch_size=actual_batch_size)
            observation = Observation.from_dict(transformed_dict)
            
            # Sample actions for entire batch
            actions = model.sample_actions(key, observation, num_steps=10)
            discrete_actions = self.process_output(actions)
            
            counter += 1
            
            # Progress reporting
            if counter % 100 == 0 or counter == total_batches:
                elapsed_time = time.perf_counter() - start_time
                print(f"Progress: {counter}/{total_batches} batches ({counter/total_batches*100:.1f}%) - {elapsed_time:.1f}s elapsed")
            
            # Get ground truth actions from batch
            gt_actions = np.array(batch['action'])
            
            # Calculate metrics
            batch_exact_matches = 0
            batch_p0_matches = 0
            batch_p1_matches = 0
            batch_joint_matches = 0
            
            for i in range(len(discrete_actions)):
                for j in range(len(discrete_actions[i])):
                    pred_action_idx = discrete_actions[i][j]
                    gt_action_idx = gt_actions[i]  # Ground truth is single value per timestep
                    
                    # Exact match
                    if pred_action_idx == gt_action_idx:
                        batch_exact_matches += 1
                        batch_joint_matches += 1
                    
                    # Convert to joint actions for per-player analysis
                    pred_joint = self.discrete_to_joint[pred_action_idx]
                    gt_joint = self.discrete_to_joint[gt_action_idx]
                    
                    # Per-player accuracy
                    if pred_joint[0] == gt_joint[0]:  # Player 0
                        batch_p0_matches += 1
                    if pred_joint[1] == gt_joint[1]:  # Player 1
                        batch_p1_matches += 1
            
            total_predictions = len(discrete_actions) * len(discrete_actions[0]) if len(discrete_actions) > 0 else 0
            
            # Store batch results (convert to Python int to avoid JSON serialization issues)
            dataset_results.all_preds.extend([int(action) for batch_actions in discrete_actions for action in batch_actions])
            dataset_results.all_gt.extend([int(action) for action in gt_actions.tolist()])
            dataset_results.total_batches = counter
            dataset_results.total_timesteps += total_predictions
            dataset_results.exact_match_rate += batch_exact_matches
            dataset_results.player0_accuracy += batch_p0_matches
            dataset_results.player1_accuracy += batch_p1_matches
            dataset_results.joint_accuracy += batch_joint_matches

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
            del transformed_dict, observation, actions, discrete_actions, gt_actions
            gc.collect()
            jax.clear_caches()

        end_time = time.perf_counter()
        eval_duration = end_time - start_time
        dataset_results.eval_time = eval_duration
        
        # Calculate final averages
        if dataset_results.total_timesteps > 0:
            dataset_results.exact_match_rate /= dataset_results.total_timesteps
            dataset_results.player0_accuracy /= dataset_results.total_timesteps
            dataset_results.player1_accuracy /= dataset_results.total_timesteps
            dataset_results.joint_accuracy /= dataset_results.total_timesteps

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

    results_file = os.path.join(args.output_dir, 'pi0_base_overcooked_results.json')
    raw_data_file = os.path.join(args.output_dir, 'raw_predictions_gt.json')

    # Create dataloader
    dataset_obj, dataloader = get_overcooked_dataloader(args.data_file, batch_size=args.batch_size, by_episode=False)
    
    # Run evaluation
    dataset_name = "overcooked"
    print(f'\n ---- EVALUATING {dataset_name} ---- \n')
    results = overcooked_inference.evaluate_model(model, key, config, dataloader, dataset_name, args.output_dir)
    
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
    print(f'Total timesteps: {results["total_timesteps"]}')
    print(f'Evaluation time: {results["eval_time"]:.2f} seconds')


if __name__ == "__main__":
    main()
