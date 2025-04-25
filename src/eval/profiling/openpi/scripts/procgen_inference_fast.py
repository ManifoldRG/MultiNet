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
from src.eval.profiling.openpi.src.openpi.models.tokenizer import FASTTokenizer
from src.data_utils.procgen_dataloader import get_procgen_dataloader
# 'definitions' import is now handled correctly because project root is in sys.path
# No change needed for the import within procgen_dataloader.py itself
from src.eval.profiling.openpi.src.openpi.shared import download
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import gc
# Update imports to be relative to the project root (prefix with 'src.')
from src.eval.profiling.openpi.src.openpi.shared import normalize
from src.eval.profiling.openpi.src.openpi.transforms import Unnormalize, ExtractFASTActions, pad_to_dim
from src.eval.profiling.openpi.src.openpi.shared.normalize import RunningStats
from src.eval.profiling.openpi.scripts.procgen_utils import ActionUtils
from definitions.procgen import ProcGenDefinitions
from src.eval_utils import (
    calculate_brier_mae,
    calculate_mean,
    quantile_filter,
    min_max_normalize,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
    get_exact_match_rate,
    calculate_tp_fp_fn_counts,
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1
)
import time
import functools

from dataclasses import dataclass, field, fields
import jax.numpy as jnp
import jax.tree_util as jtu

#Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")
# Configure JAX memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
MAX_BRIER_ERROR = 2.0


@dataclass
class DatasetResults:
    all_preds: list[list[float]] = field(default_factory=list)
    all_gt: list[list[float]] = field(default_factory=list)
    
    eval_time: float = 0
    total_batches: int = 0
    total_timesteps: int = 0
    total_invalids: int = 0
    invalid_predictions_percentage: float = 0
    emr: float = 0
    total_brier_mae: float = 0;
    total_quantile_filtered_brier_mae: float = 0
    micro_precision: float = 0
    micro_recall: float = 0
    micro_f1: float = 0

    avg_brier_mae: float = 0
    avg_normalized_brier_mae: float = 0
    avg_quantile_filtered_brier_mae: float = 0
    avg_quantile_filtered_normalized_brier_mae: float = 0
    max_rel_brier_mae: float = 0
    prop_beyond_threshold_brier_mae: float = 0

    clipped_emr: float = 0
    clipped_micro_precision: float = 0
    clipped_micro_recall: float = 0
    clipped_micro_f1: float = 0

    micro_precision_without_invalids: float = 0
    micro_f1_without_invalids: float = 0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


class ProcGenInferenceFast:
    def __init__(self, model, tokenizer: FASTTokenizer, config: pi0_fast.Pi0FASTConfig, max_decoding_steps: int):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_decoding_steps = max_decoding_steps
        with open("src/eval/profiling/openpi/scripts/bpe_token_to_action_value_mappings.json", "r") as f:
            self.bpe_to_action_value_mappings = json.load(f)

        self.paligemma_tokens_to_action_values = {}

    def prepare_observation(self, obs_dict: dict, action_dim: int = 1, max_token_length: int = 48) -> dict:
        # Prepare observation dictionary for pi0 model inference
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
                "base_1_rgb": jax.numpy.ones(current_batch_size, dtype=bool), # Added mask for base_1
                "wrist_0_rgb": jax.numpy.ones(current_batch_size, dtype=bool) # Renamed mask for wrist
            },
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": token_ar_mask,
            "token_loss_mask": token_loss_mask
        }
        
        # Clear original image data
        obs_dict["image_observation"] = None
        del base_image
        gc.collect()

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
                float_actions = ProcGenDefinitions.set_procgen_unused_special_action_to_stand_still(
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

        """Discretize the actions after scaling them back to the original action space"""
        # Procgen actions are typically integers representing discrete choices.
        # Rounding might be appropriate, or casting to int if they map directly.
        discretized_actions = np.round(unnormalized_actions).astype(int) # Round and cast to int
        print('Action after unnormalization: ', discretized_actions)

        return discretized_actions
    
    def evaluate_model(self, key, config, dataset_stats: dict, dataloader: tf.data.Dataset, dataset: str, output_dir: str):
        """Evaluate the model on the dataset"""
        counter = 0
        dataset_results = DatasetResults()
        all_brier_maes = []

        start_time = time.perf_counter()
        decoder = ExtractFASTActions(tokenizer=self.tokenizer, action_horizon=config.action_horizon, action_dim=config.action_dim)
        unnormalizer = Unnormalize(norm_stats={'action': dataset_stats['action']}, use_quantiles=True)

        # def calculate_brier_mae(predicted_probabilities, one_hot_label) -> float:
        #     """Calculate mean absolute error from absolute errors using JAX operations"""
        #     return jnp.sum(jnp.abs(predicted_probabilities - one_hot_label))

        # @functools.partial(jax.jit, static_argnames=['action_space_size'])
        # def calculate_batch_brier_mae(action_probs_batch, gt_actions_batch, action_space_size):
        #     # Create one-hot encoded ground truth actions for entire batch at once
        #     gt_one_hot = jax.nn.one_hot(gt_actions_batch, action_space_size)
        #     return jax.vmap(lambda x, y: calculate_brier_mae(x, y))(action_probs_batch, gt_one_hot)

        for batch in dataloader:
            batch_start_time = time.perf_counter()
            # Process entire batch at once, keeping data on CPU
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
            action_tokens, action_probs = self.model.sample_actions(key, observation, max_decoding_steps = self.max_decoding_steps, temperature=0.0)

            # Move results back to CPU immediately
            action_tokens = jax.device_get(action_tokens)
            action_probs = jax.device_get(action_probs)

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
            dataset_results.total_timesteps += len(actions)

            action_space = sorted(ProcGenDefinitions.get_valid_action_space(dataset, "default"))
            unnormalized_discrete_actions = self.process_output(actions, dataset_stats)
            # Get ground truth actions and compute metrics on CPU
            gt_actions = np.array(batch['action'])
            gt_actions = ProcGenDefinitions.set_procgen_unused_special_action_to_stand_still(gt_actions, dataset)

            dataset_results.all_preds.extend(unnormalized_discrete_actions.tolist())
            dataset_results.all_gt.extend(gt_actions.tolist())

            # Calculate Brier MAE
            for action_idx in range(len(actions)):
                # get one hot encoded gt action
                gt_action_one_hot = np.zeros((len(action_space)))
                gt_action_one_hot[gt_actions[action_idx]] = 1
                if unnormalized_discrete_actions[action_idx] not in action_space:
                    all_brier_maes.append(MAX_BRIER_ERROR)
                else:
                    unnormalized_action_values_to_probs = self.get_unnormalized_action_values_to_probs(
                        dataset, action_probs[action_idx], unnormalizer, action_space
                    )

                    all_brier_maes.append(calculate_brier_mae(unnormalized_action_values_to_probs, gt_action_one_hot))

            time_per_timestep = (time.perf_counter() - batch_start_time)/len(actions)
            del obs, transformed_dict, observation, action_tokens, action_probs, decoded_actions, processed_actions, \
                unnormalized_discrete_actions, gt_actions
                # unnormalized_probs_batch, batch_brier_maes, 
            gc.collect()
            jax.clear_caches()
            tf.keras.backend.clear_session()

            print(f"Processed {counter} episodes, cleared memory, took {time_per_timestep} seconds per timestep")
            counter += 1
            # Uncomment to stop after 2 batches
            # if counter == 10:
            #     break

        end_time = time.perf_counter()
        dataset_results.eval_time = end_time - start_time
        print(f"Time taken for {counter} batches: {dataset_results.eval_time} seconds")

        # Calculate metrics
        total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
            dataset_results.all_preds, dataset_results.all_gt, action_space
        )

        micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
        micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
        micro_f1 = get_micro_f1(micro_precision, micro_recall)
        
        print(f"Unclipped Micro Precision: {micro_precision}, Micro Recall: {micro_recall}, Micro F1: {micro_f1}")
        
        total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
            dataset_results.all_preds, dataset_results.all_gt, action_space
        )

        micro_precision_without_invalids = get_micro_precision_from_counts(total_tp, valid_fp)
        micro_f1_without_invalids = get_micro_f1(micro_precision_without_invalids, micro_recall) # micro_recall is not affected

        print(f"Unclipped Micro Precision without invalids: {micro_precision_without_invalids}, \
            Unclipped Micro F1 without invalids: {micro_f1_without_invalids}")

        clipped_predictions = np.clip(dataset_results.all_preds, action_space[0], action_space[-1])
        clipped_emr = get_exact_match_rate(clipped_predictions, dataset_results.all_gt)
        clipped_total_tp, clipped_total_fp, clipped_total_fn, _, _ = calculate_tp_fp_fn_counts(
            clipped_predictions, dataset_results.all_gt, action_space
        )
        clipped_micro_precision = get_micro_precision_from_counts(clipped_total_tp, clipped_total_fp)
        clipped_micro_recall = get_micro_recall_from_counts(clipped_total_tp, clipped_total_fn)
        clipped_micro_f1 = get_micro_f1(clipped_micro_precision, clipped_micro_recall)

        print(f"Clipped Micro Precision: {clipped_micro_precision}, Clipped Micro Recall: {clipped_micro_recall}, Clipped Micro F1: {clipped_micro_f1}")

        # Store results for this batch
        dataset_results.total_invalids = int(invalid_fp)  # invalid_fp is the same as the number of invalid predictions
        dataset_results.invalid_predictions_percentage = dataset_results.total_invalids / dataset_results.total_timesteps * 100

        dataset_results.total_batches = counter
        dataset_results.total_brier_mae = sum(all_brier_maes)
        dataset_results.emr = get_exact_match_rate(dataset_results.all_preds, dataset_results.all_gt)
        dataset_results.micro_precision = micro_precision
        dataset_results.micro_recall = micro_recall
        dataset_results.micro_f1 = micro_f1

        dataset_results.clipped_emr = clipped_emr
        dataset_results.clipped_micro_precision = clipped_micro_precision
        dataset_results.clipped_micro_recall = clipped_micro_recall
        dataset_results.clipped_micro_f1 = clipped_micro_f1

        dataset_results.avg_brier_mae = calculate_mean(all_brier_maes)
        dataset_results.avg_normalized_brier_mae = calculate_mean(min_max_normalize(all_brier_maes).tolist())

        quantile_filtered_brier_maes = quantile_filter(all_brier_maes)
        dataset_results.total_quantile_filtered_brier_mae = sum(quantile_filtered_brier_maes)
        dataset_results.avg_quantile_filtered_brier_mae = calculate_mean(quantile_filtered_brier_maes)
        dataset_results.avg_quantile_filtered_normalized_brier_mae = calculate_mean(min_max_normalize(quantile_filtered_brier_maes).tolist())

        dataset_results.max_rel_brier_mae = calculate_max_relative_mae(all_brier_maes)
        dataset_results.prop_beyond_threshold_brier_mae = calculate_proportion_beyond_mae_threshold(all_brier_maes)

        dataset_results.micro_precision_without_invalids = micro_precision_without_invalids
        dataset_results.micro_f1_without_invalids = micro_f1_without_invalids

        return dataset_results.to_dict()

    def get_unnormalized_action_values_to_probs(
            self, dataset: str, action_probs_for_last_token: np.ndarray, unnormalizer: Unnormalize, action_space: list[int]
        ) -> np.ndarray:
            if dataset not in self.paligemma_tokens_to_action_values.keys():
                self.paligemma_tokens_to_action_values[dataset] = {}
                # Vectorize the unnormalization operation
                all_action_values = np.array([float(v) for v in self.bpe_to_action_value_mappings["mappings"].values()])  # all action values in the json file
                unnormalized_values = unnormalizer({'action': all_action_values})['action']
                rounded_values = np.round(unnormalized_values).astype(int)

                # Create the mapping in one go
                for token_id, final_action_value in zip(self.bpe_to_action_value_mappings["mappings"].keys(), rounded_values):  # loop through the paligemma tokens and rounded action values
                    if final_action_value not in action_space:
                        continue
                    self.paligemma_tokens_to_action_values[dataset][token_id] = final_action_value

            unnormalized_action_values_to_probs = np.zeros(len(action_space))


            # only get the probs of the valid actions
            token_ids = np.fromiter(self.paligemma_tokens_to_action_values[dataset].keys(), dtype=int)
            valid_action_probs = np.array(action_probs_for_last_token[token_ids])

            # Accumulate probabilities
            np.add.at(
                unnormalized_action_values_to_probs,
                np.array(list(self.paligemma_tokens_to_action_values[dataset].values())),
                valid_action_probs
            )

            # Vectorized normalization
            total_prob = np.sum(unnormalized_action_values_to_probs)
            if total_prob > 0:
                unnormalized_action_values_to_probs /= total_prob

            return unnormalized_action_values_to_probs


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
        '--dataset_stats_dir',
        type=str,
        required=True,
        help='Directory to store dataset statistics'
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
    config = pi0_fast.Pi0FASTConfig(action_horizon=1, action_dim=1, batch_size=args.batch_size)
    tokenizer = FASTTokenizer()
    key = jax.random.key(0)

    params = _model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_base/params"))
    model = config.load(params)
    del params
    gc.collect()
    jax.clear_caches()
    print('Pi0-FAST Model loaded')
    procgen_inference = ProcGenInferenceFast(model,tokenizer, config, max_decoding_steps=4)  # 4 becasue of "Action", ":", and " " before action tokens

    results_file = os.path.join(args.output_dir, 'pi0_fast_procgen_results.json')

    # Get dataset shards
    procgen_dataset_list = os.listdir(args.dataset_dir) # Update path as needed
    for dataset in procgen_dataset_list:
        print(f'\n ---- EVALUATING {dataset} ---- \n')
        dataset_path = os.path.join(args.dataset_dir, dataset) #Update path as needed
        if not os.path.isdir(dataset_path):
            print(f"Skipping {dataset}, not a directory.")
            continue

        tfds_shards = os.listdir(f'{args.dataset_dir}/{dataset}') # Update path as needed
        tfds_sorted_shards = sorted(
            tfds_shards,
            key=lambda x: (
                datetime.datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"),  # primary sort by timestamp
                *(int(n) for n in x.split('_')[1:-1]),  # middle numbers as integers
                float(x.split('_')[-1])  # last number as float
            )
        )
        if not tfds_shards:
            print(f"No data shards found in {dataset_path}. Skipping.")
            continue

        # Add path to shards
        tfds_sorted_shard_paths = [os.path.join(f'{args.dataset_dir}/{dataset}', shard) for shard in tfds_sorted_shards]

        #Dataset stats loading/calculation
        stats_output_path = os.path.join(args.dataset_stats_dir, 'procgen_dataset_statistics_prod.json')
        if os.path.exists(stats_output_path):
            print(f'Loading existing dataset stats from {stats_output_path}')
            with open(stats_output_path, 'r') as f:
                dataset_stats_dict = json.load(f)
            # Create NormStats object from the loaded dictionary
            # Ensure the structure matches what NormStats expects (mean, std, q01, q99)
            try:
                 # Assuming the dict has an 'action' key containing the stats dict
                 action_stats_dict = dataset_stats_dict[dataset].get('action', {})
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
            dataset_obj, dataloader = get_procgen_dataloader(tfds_sorted_shard_paths, dataset_name=dataset, batch_size=args.batch_size)
            del dataset_obj
            gc.collect()
            jax.clear_caches()
        except Exception as e:
            raise Exception(f"Error creating dataloader for {dataset}: {e}")

        
        # Call evaluate_model with dataset_stats
        results = procgen_inference.evaluate_model(key, config, dataset_stats, dataloader, dataset, args.output_dir)

        # Load existing results if file exists, otherwise create new
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    dataset_results = json.load(f)
            except Exception as e:
                raise Exception(f"Result file might be corrupted. Please delete it and run the script again. {e}")
        else:
            dataset_results = {}

        dataset_results[dataset] = results
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(dataset_results, f, indent=4)


if __name__ == "__main__":
    main()