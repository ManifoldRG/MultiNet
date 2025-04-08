import argparse
import datetime
import json
import os
import sys
# Adjust the relative path to go up FIVE levels to reach the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
# Update imports to be relative to the project root (prefix with 'src.')
from src.eval.profiling.openpi.src.openpi.models import pi0_fast
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.models.model import Observation
from src.eval.profiling.openpi.src.openpi.models.tokenizer import FASTTokenizer, PaligemmaTokenizer
from src.data_utils.procgen_dataloader import get_procgen_dataloader
# 'definitions' import is now handled correctly because project root is in sys.path
# No change needed for the import within procgen_dataloader.py itself
from src.eval.profiling.openpi.src.openpi.shared import download
import jax
import numpy as np
import tensorflow as tf
import gc
# Update imports to be relative to the project root (prefix with 'src.')
from src.eval.profiling.openpi.src.openpi.shared import normalize
from src.eval.profiling.openpi.src.openpi.transforms import Unnormalize, ExtractFASTActions, pad_to_dim
from src.eval.profiling.openpi.src.openpi.shared.normalize import RunningStats
from src.eval.profiling.openpi.scripts.procgen_utils import ActionUtils, MetricUtils
from definitions.procgen import ProcGenDefinitions
import jax.numpy as jnp
import jax.tree_util as jtu

#Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")
# Configure JAX memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class ProcGenInferenceFast:
    def __init__(self, model, tokenizer: FASTTokenizer, config: pi0_fast.Pi0FASTConfig, max_decoding_steps: int = 10):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_decoding_steps = max_decoding_steps

    def prepare_observation(self, obs_dict: dict, action_dim: int = 1, max_token_length: int = 48) -> dict:
        # Prepare observation dictionary for pi0 model inference
        tokenizer = self.tokenizer

        # Process image observation
        base_image = obs_dict["image_observation"]
        if isinstance(base_image, list) and isinstance(base_image[0], tf.Tensor): # Handle list of tensors
            base_image = [img.numpy() for img in base_image]
            base_image = np.stack(base_image, axis=0) # Stack into a single numpy array
        elif isinstance(base_image, tf.Tensor): # Handle single tensor
             base_image = base_image.numpy()

        # Add batch dimension if needed (e.g., if processing a single timestep)
        if isinstance(base_image, np.ndarray) and len(base_image.shape) == 3:
            base_image = base_image[None, ...] # Add batch dimension

        # Ensure base_image is a jax array
        base_image = jax.numpy.array(base_image)
        # Determine the actual batch size from the image data
        current_batch_size = base_image.shape[0]

        # Create zero images for missing views based on the actual batch size
        zero_image = jax.numpy.zeros_like(base_image)

        # Process text observation
        text_obs = obs_dict["text_observation"]
        # Ensure text_obs is a list of strings, matching the batch size
        if isinstance(text_obs, tf.Tensor):
            # If it's a single tensor, decode and replicate for the batch
            decoded_text = text_obs.numpy().decode('utf-8')
            text_obs = [decoded_text] * current_batch_size
        elif isinstance(text_obs, list) and all(isinstance(t, tf.Tensor) for t in text_obs):
            # If it's a list of tensors, decode each one
            text_obs = [t.numpy().decode('utf-8') for t in text_obs]
        elif isinstance(text_obs, str):
             # If it's a single string, replicate for the batch
             text_obs = [text_obs] * current_batch_size
        # Add checks for other potential input types if necessary
        
        # Initialize empty lists for batch processing
        tokens_list = []
        token_mask_list = []
        token_ar_mask_list = []  # Added for auto-regressive mask
        token_loss_mask_list = [] # Added for loss mask
        
        # Process each text in the batch
        if len(text_obs) > 1:
            for text in text_obs:
                # For inference, we don't have actions yet, so pass None
                # We also don't have a proprio state for Procgen (or pass zeros if needed)
                state = pad_to_dim(jnp.zeros(1), action_dim)  # zeros for state
                tokens, token_mask, token_ar_mask, token_loss_mask = self.tokenizer.tokenize(
                    prompt=text,
                    state=state,
                    actions = None
                )
                
                tokens_list.append(jax.numpy.array(tokens))
                token_mask_list.append(jax.numpy.array(token_mask))
                token_ar_mask_list.append(jax.numpy.array(token_ar_mask))
                token_loss_mask_list.append(jax.numpy.array(token_loss_mask))
        else:
            state = pad_to_dim(jnp.zeros(1), action_dim)  # zeros for state
            tokens, token_mask, token_ar_mask, token_loss_mask = self.tokenizer.tokenize(
                prompt=text_obs[0], # Pass the first element of the list
                state=state,
                actions = None            
            )
            
            tokens_list.append(jax.numpy.array(tokens))
            token_mask_list.append(jax.numpy.array(token_mask))
            token_ar_mask_list.append(jax.numpy.array(token_ar_mask))
            token_loss_mask_list.append(jax.numpy.array(token_loss_mask))

        # Stack the lists of arrays into single batch arrays
        tokens = jax.numpy.stack(tokens_list)
        token_mask = jax.numpy.stack(token_mask_list)
        token_ar_mask = jax.numpy.stack(token_ar_mask_list)
        token_loss_mask = jax.numpy.stack(token_loss_mask_list)

        # Create observation dictionary matching Pi0FAST schema
        transformed_dict = {
            # Use current_batch_size derived from image data
            "state": jnp.asarray(pad_to_dim(jnp.zeros((current_batch_size, 1), dtype=jnp.float32), action_dim)),
            "image": {
                "base_0_rgb": base_image,
                "base_1_rgb": zero_image,      # Added missing view (zeros)
                "wrist_0_rgb": zero_image       # Renamed from left/right_wrist (zeros)
            },
            "image_mask": {
                # Use current_batch_size for masks
                "base_0_rgb": jax.numpy.ones(current_batch_size, dtype=bool),
                "base_1_rgb": jax.numpy.zeros(current_batch_size, dtype=bool), # Added mask for base_1
                "wrist_0_rgb": jax.numpy.zeros(current_batch_size, dtype=bool) # Renamed mask for wrist
            },
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": token_ar_mask,
            "token_loss_mask": token_loss_mask
        }

        return transformed_dict


    def get_dataset_stats(self, root_dir: str, dataset_name: str):
        
        running_stats = RunningStats()

        print('Calculating dataset stats...')
        #Load the dataset shards and calculate the stats
        for shard in root_dir:
            print(f'Processing shard: {shard}')
            dataset = tf.data.Dataset.load(shard)
            actions = []
            for elem in dataset:
                float_actions = ActionUtils.set_procgen_unused_special_action_to_stand_still(
                    elem['actions'][0].numpy(), dataset_name) # Procgen has 1D action space so only first dimension is used
                actions.append(float_actions)
            
            # Update running statistics
            actions = np.array(actions)
            running_stats.update(actions)

            # Free memory from actions list and dataset
            del actions
            del dataset
            import gc
            gc.collect()
    
        # Get final statistics
        stats = running_stats.get_statistics()
        # Convert NormStats to dictionary format
        stats_dict = {
            'mean': stats.mean.tolist(),
            'std': stats.std.tolist(),
            'q01': stats.q01.tolist(),
            'q99': stats.q99.tolist()
        }
        return {'action': stats_dict}, {'action': stats}
    
    def process_output(self, action_tokens, dataset_stats: dict):
        """
        Decode the model's action tokens and round them.

        Args:
            action_tokens (jax.numpy.ndarray): Action tokens predicted by the pi0_fast model.
                                               Shape: (batch_size, num_action_tokens)
            dataset_stats (dict): Dataset statistics containing mean and std for unnormalization.
                                  Expected format: {'action': NormStats(mean=..., std=..., ...)}


        Returns:
            np.ndarray: Decoded and rounded actions. Shape: (batch_size, 1)
        """

        # Convert tokens to numpy
        actions_from_tokens = np.array(action_tokens)

        if actions_from_tokens.ndim > 1:
             actions = actions_from_tokens[:, 0:1] # Take the first token's value as the action as Procgen has 1D action space
        else:
             actions = actions_from_tokens[:, None] # Add dimension if needed

        # Get only first dimension since Procgen uses 1D action space
        actions = actions[..., 0:1]

        # Load normalization statistics
        norm_stats = dataset_stats.get('action') # Get the NormStats object for 'action'
        if norm_stats is None:
            raise ValueError("Dataset statistics for 'action' not found or invalid.")


        print('\nAction before unnormalization: ', actions)
        # Create and apply unnormalize transform
        unnormalizer = Unnormalize(norm_stats={'action': norm_stats}, use_quantiles=True)
        unnormalized_actions = unnormalizer({'action': actions})['action']
        print('Action after unnormalization: ', unnormalized_actions)

        """Discretize the actions after scaling them back to the original action space"""
        # Procgen actions are typically integers representing discrete choices.
        # Rounding might be appropriate, or casting to int if they map directly.
        discretized_actions = np.round(unnormalized_actions).astype(int) # Round and cast to int

        return discretized_actions
    
    def evaluate_model(self, key, config, dataset_stats: dict, dataloader: tf.data.Dataset, dataset: str, output_dir: str):
        """Evaluate the model on the dataset"""
        counter = 0
        results_file = os.path.join(output_dir, f'{dataset}_results.json')
        
        # Create CPU device for data preparation
        cpu_device = jax.devices("cpu")[0]
        
        for batch in dataloader:
            # Process entire batch at once, keeping data on CPU
            with jax.default_device(cpu_device):
                obs = {
                    'image_observation': batch['image_observation'],
                    'text_observation': batch['text_observation']
                }
                
                # Transform observation (will stay on CPU)
                transformed_dict = self.prepare_observation(obs, max_token_length=config.max_token_len)
                if "token_loss_mask" not in transformed_dict:
                    transformed_dict["token_loss_mask"] = jax.numpy.zeros_like(
                        transformed_dict["tokenized_prompt_mask"], 
                        dtype=bool
                    )

                # Create observation object on CPU
                observation = Observation.from_dict(transformed_dict)

            # Transfer necessary data to accelerator only for model inference
            # The model's sample_actions will handle the device transfer internally
            action_tokens = self.model.sample_actions(key, observation, max_decoding_steps = self.max_decoding_steps, temperature=0.0)
            #print('\nAction tokens before decoding: ', action_tokens)
            decoder = ExtractFASTActions(tokenizer=self.tokenizer, action_horizon=config.action_horizon, action_dim=config.action_dim)
            
            # Process all action tokens at once instead of individually
            decoded_actions = []
            for each_action in action_tokens:
                action = decoder({'actions': each_action})['actions']
                decoded_actions.append(action)

            # Process each action to ensure consistent shape and size
            processed_actions = []
            for action in decoded_actions:
                # Convert to numpy and squeeze any extra dimensions
                action = np.array(action)
                try:
                    if len(action.shape) > 1:
                        action = np.squeeze(action)
                    # Trim to 1 dimensions if needed
                    if len(action) > self.max_decoding_steps:
                        print('\nAction is longer than {} dimensions: '.format(self.max_decoding_steps), action)
                        action = action[:self.max_decoding_steps]
                    elif len(action) < self.max_decoding_steps:
                        print('\nAction is shorter than {} dimensions: '.format(self.max_decoding_steps), action)
                        action = np.pad(action, (0, self.max_decoding_steps - len(action)))
                    else:
                        print('\nAction is of correct size: ', action)
                except:
                    print('\n Scalar action detected: ', action)
                
                processed_actions.append(action)
            
            # Stack the actions into a single array
            actions = np.stack(processed_actions)
            #print('\nDecoded Actions: ', actions)

            # Process outputs back on CPU
            with jax.default_device(cpu_device):
                unnormalized_discrete_actions = self.process_output(actions, dataset_stats)
                counter += 1
                
                # Get ground truth actions and compute metrics on CPU
                gt_actions = np.array(batch['action'])
                gt_actions = ActionUtils.set_procgen_unused_special_action_to_stand_still(gt_actions, dataset)

                action_space = ProcGenDefinitions.get_valid_action_space(dataset)
                total_tp, total_fp, total_fn = MetricUtils._calculate_metrics_counts(gt_actions, unnormalized_discrete_actions, action_space)
                micro_precision = MetricUtils.get_micro_precision_from_counts(total_tp, total_fp)
                micro_recall = MetricUtils.get_micro_recall_from_counts(total_tp, total_fn)
                micro_f1 = MetricUtils.get_micro_f1(micro_precision, micro_recall)

                batch_results = {
                    "dataset": dataset,
                    "batch_id": counter,
                    "metrics": {
                        "micro_precision": float(micro_precision),
                        "micro_recall": float(micro_recall),
                        "micro_f1_score": float(micro_f1)
                    },
                    "predictions": {
                        "ground_truth": gt_actions.tolist(),
                        "predicted": unnormalized_discrete_actions.tolist()
                    }
                }

            # Handle results file I/O
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    dataset_results = json.load(f)
            else:
                dataset_results = {
                    "dataset": dataset,
                    "batches": []
                }
            
            dataset_results["batches"].append(batch_results)
            with open(results_file, 'w') as f:
                json.dump(dataset_results, f, indent=4)
            
            print(f"Dataset: {dataset}, Batch {counter} metrics - Micro Precision: {micro_precision:.4f}, Micro Recall: {micro_recall:.4f}, Micro F1: {micro_f1:.4f}")
            

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with output_dir and dataset_dir
    """
    parser = argparse.ArgumentParser(
        description="Run inference on ProcGen datasets using pi0-FAST model" # Updated description
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        # Updated help text to mention stats, like original script
        help='Directory to store results and dataset statistics'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Root directory containing the procgen datasets'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='Batch size for inference'
    )


    args = parser.parse_args()

    # Validate paths exist
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    return args

def main():
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\nResults will be stored in: {args.output_dir}')
    print(f'\nReading datasets from: {args.dataset_dir}')

    # Initialize model and inference object for pi0_fast
    # Use pi0_fast config and checkpoint
    config = pi0_fast.Pi0FASTConfig(action_horizon=1, action_dim=1) 
    tokenizer = FASTTokenizer()
    key = jax.random.key(0)

    params = _model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_base/params"))
    model = config.load(params)
    del params
    gc.collect()
    jax.clear_caches()
    print('Pi0-FAST Model loaded')
    procgen_inference = ProcGenInferenceFast(model,tokenizer, config, max_decoding_steps=4)  # 4 becasue of "Action", ":", and " " before action tokens


    # Get dataset shards
    procgen_dataset_list = os.listdir(args.dataset_dir) # Update path as needed
    for dataset in procgen_dataset_list:
        print(f'\n ---- EVALUATING {dataset} ---- \n')
        dataset_path = os.path.join(args.dataset_dir, dataset) #Update path as needed
        if not os.path.isdir(dataset_path):
            print(f"Skipping {dataset}, not a directory.")
            continue

        tfds_shards = os.listdir(dataset_path) # Update path as needed
        # Filter out non-directory files if any
        tfds_shards = [s for s in tfds_shards if os.path.isdir(os.path.join(dataset_path, s))]
        if not tfds_shards:
            print(f"No data shards found in {dataset_path}. Skipping.")
            continue

        # Attempt to sort shards by timestamp, handle potential errors
        try:
            tfds_sorted_shards = sorted(tfds_shards, key=lambda x: datetime.datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
        except (ValueError, IndexError):
            print(f"Warning: Could not sort shards by timestamp for {dataset}. Using unsorted order.")
            tfds_sorted_shards = tfds_shards

        # Add path to shards
        tfds_sorted_shard_paths = [os.path.join(dataset_path, shard) for shard in tfds_sorted_shards]


        #Dataset stats loading/calculation
        stats_output_path = os.path.join(args.output_dir, f'{dataset}_dataset_stats.json')
        if os.path.exists(stats_output_path):
            print(f'Loading existing dataset stats from {stats_output_path}')
            with open(stats_output_path, 'r') as f:
                dataset_stats_dict = json.load(f)
            # Create NormStats object from the loaded dictionary
            # Ensure the structure matches what NormStats expects (mean, std, q01, q99)
            try:
                 # Assuming the dict has an 'action' key containing the stats dict
                 action_stats_dict = dataset_stats_dict.get('action', {})
                 # Convert lists back to numpy arrays if needed by NormStats constructor
                 for stat_key in action_stats_dict:
                     action_stats_dict[stat_key] = np.array(action_stats_dict[stat_key])
                 dataset_stats = {'action': normalize.NormStats(**action_stats_dict)}
            except TypeError as e:
                 print(f"Error creating NormStats from loaded dict: {e}. Check format in {stats_output_path}")
                 print("Skipping stats loading for this dataset.")
                 dataset_stats = None # Indicate stats are unavailable
            except KeyError as e:
                 print(f"Missing key {e} in loaded stats dict: {stats_output_path}")
                 print("Skipping stats loading for this dataset.")
                 dataset_stats = None # Indicate stats are unavailable

        else:
            print(f'Calculating dataset stats for {dataset}...')
            # Pass the list of full shard paths to get_dataset_stats
            dataset_stats_dict, dataset_stats = procgen_inference.get_dataset_stats(tfds_sorted_shard_paths, dataset_name=dataset)
            print('Dataset stats calculated.')
            print(f'Saving dataset stats to {stats_output_path}')
            # Ensure the dictionary saved contains serializable lists (handled by get_dataset_stats)
            with open(stats_output_path, 'w') as f:
                json.dump(dataset_stats_dict, f, indent=4)

        # Skip evaluation if stats could not be loaded or calculated
        if dataset_stats is None:
             print(f"Cannot proceed with evaluation for {dataset} due to missing/invalid stats. Skipping.")
             continue

        # Create dataloader
        try:
            # Pass batch_size from args
            dataset_obj, dataloader = get_procgen_dataloader(tfds_sorted_shard_paths, batch_size=args.batch_size)
            del dataset_obj
            gc.collect()
            jax.clear_caches()
        except Exception as e:
            raise Exception(f"Error creating dataloader for {dataset}: {e}")

        
        # Call evaluate_model with dataset_stats
        procgen_inference.evaluate_model(key, config, dataset_stats, dataloader, dataset, args.output_dir)




if __name__ == "__main__":
    main() 