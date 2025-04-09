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
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from definitions.procgen import ProcGenDefinitions
from src.eval.profiling.openpi.src.openpi.shared import download
import jax
import numpy as np
import tensorflow as tf
import gc
from src.eval.profiling.openpi.src.openpi.shared import normalize
from src.eval.profiling.openpi.src.openpi.transforms import Unnormalize
from src.eval.profiling.openpi.src.openpi.shared.normalize import RunningStats
from src.eval.profiling.openpi.scripts.procgen_utils import ActionUtils
from definitions.procgen import ProcGenDefinitions
from src.eval_utils import (get_exact_match_rate,
                            calculate_tp_fp_fn_counts,
                            get_micro_precision_from_counts, 
                            get_micro_recall_from_counts, 
                            get_micro_f1)
from dataclasses import dataclass, fields


#Restrict tf to CPU
tf.config.set_visible_devices([], "GPU")
# Configure JAX memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'


@dataclass
class DatasetResults:
    total_batches: int = 0
    total_timesteps: int = 0
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

class ProcGenInference:
    def __init__(self, model, tokenizer: PaligemmaTokenizer, config: pi0.Pi0Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def prepare_observation(self, obs_dict: dict, batch_size: int = 5, action_dim: int = 32, max_token_length: int = 48) -> dict:
    #Prepare observation dictionary for pi0 model inference
        tokenizer = self.tokenizer
        
        # Process image observation
        base_image = obs_dict["image_observation"]
        if isinstance(base_image[0], tf.Tensor):
            for img in range(len(base_image)):
                base_image[img] = base_image[img].numpy()
        
        # Add batch dimension if needed
        if isinstance(base_image, np.ndarray) and len(base_image.shape) == 3:
            base_image = base_image[None, ...] # Add batch dimension
            
        # Create zero images for missing views and convert to jax array
        base_image = jax.numpy.array(base_image)
        zero_image = jax.numpy.zeros_like(base_image)
        
        # Process text observation - Procgen does not have a text observation per timestep, so we use the description of the environment/task as the prompt
        text_obs = obs_dict["text_observation"]
        if isinstance(text_obs, tf.Tensor):
            text_obs = text_obs.numpy().decode('utf-8')
            
        # Tokenize text prompt/observation
        if isinstance(text_obs, list):
            tokens = [0]*len(text_obs)
            token_mask = [0]*len(text_obs)
            for i in range(len(text_obs)):
                tokens[i], token_mask[i] = tokenizer.tokenize(text_obs[i])
                tokens[i] = jax.numpy.array(tokens[i])
                token_mask[i] = jax.numpy.array(token_mask[i])
        else:
            tokens, token_mask = tokenizer.tokenize(text_obs)
        
        if not isinstance(tokens, list) and len(tokens.shape) == 1:  
            tokens = jax.numpy.array(tokens)[None, ...] # Add batch dimension
            token_mask = jax.numpy.array(token_mask)[None, ...] # Add batch dimension
        else:
            tokens = jax.numpy.array(tokens)
            token_mask = jax.numpy.array(token_mask)
        
        # Create observation dictionary
        transformed_dict = {
            "state": jax.numpy.zeros((batch_size, action_dim)),  # Dummy state since Procgen doesn't have a proprioceptive state
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

    def get_dataset_stats(self, root_dir: str, dataset_name: str):
        
        running_stats = RunningStats()

        print('Calculating dataset stats...')
        #Load the dataset shards and calculate the stats
        for shard in root_dir:
            print(f'Processing shard: {shard}')
            dataset = tf.data.Dataset.load(shard)
            actions = []
            for elem in dataset:          
                float_action_tensor = ActionUtils.set_procgen_unused_special_action_to_stand_still(   # Procgen has 1D action space so only first dimension is used
                    np.array(elem['actions'][0].numpy()), dataset_name)
                actions.append(float_action_tensor)

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

    def process_output(self, actions, dataset_stats: dict):
        """
        Unnormalize the model's action outputs using stored normalization statistics.
        
        Args:
            actions (jax.numpy.ndarray): Normalized actions from the model
            dataset_stats (dict): Dictionary containing normalization statistics
            
        Returns:
            np.ndarray: Unnormalized actions scaled back to the original action space
        """
        # Convert to numpy array if actions is a jax array
        actions = np.array(actions)
        # Get only first dimension since Procgen uses 1D action space
        actions = actions[..., 0:1]  # Keep the first dimension while preserving batch dimensions
        
        # Load normalization statistics
        norm_stats = dataset_stats
        
        print('Raw model actions:', actions)
        print('Normalization stats:', norm_stats)
        # Create and apply unnormalize transform
        unnormalizer = Unnormalize(norm_stats=norm_stats)
        unnormalized_actions = unnormalizer({'action': actions})['action']
        print('Action after unnormalization: ', unnormalized_actions)

        """Discretize the actions after scaling them back to the original action space"""
        unnormalized_actions = np.round(unnormalized_actions, 0)
        
        return unnormalized_actions

    def evaluate_model(self, model, key, config, dataset_stats: dict, dataloader: tf.data.Dataset, dataset: str) -> dict[any]:
        """Evaluate the model on the dataset"""
        counter = 0
        dataset_results = DatasetResults()

        for batch in dataloader:
            # Process entire batch at once
            obs = {
                'image_observation': batch['image_observation'],  # Full batch
                'text_observation': batch['text_observation']     # Full batch
            }
            print(f"Batch {counter} obs size: {len(obs['image_observation'])}")
            # Transform observation
            transformed_dict = self.prepare_observation(obs, max_token_length=config.max_token_len)
            observation = Observation.from_dict(transformed_dict)
            
            # Sample actions for entire episode
            actions = model.sample_actions(key, observation, num_steps=10)
            unnormalized_discrete_actions = self.process_output(actions, dataset_stats)
            counter += 1
            print(f"Batch {counter} actions:", unnormalized_discrete_actions)

            #Compare to gt actions and calculate error value
            # Get ground truth actions from batch
            gt_actions = np.array(batch['action'])
            gt_actions = ActionUtils.set_procgen_unused_special_action_to_stand_still(gt_actions, dataset)
            
            print('Ground truth actions: ', gt_actions)
            print('Predicted actions: ', unnormalized_discrete_actions)
            
            # Calculate metrics
            emr = get_exact_match_rate(unnormalized_discrete_actions, gt_actions)
            action_space = sorted(ProcGenDefinitions.get_valid_action_space(dataset, 'default'))
            
            # Calculate metrics counts once and reuse
            total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
                unnormalized_discrete_actions, gt_actions, action_space
            )

            # Calculate all metrics using the same counts
            """
            using micro to avoid minority class distortion since we expect the action predictions to be imbalanced.
            e.g. in one episode, the agent might take the same action consecutively to reach a goal in a straight line.
            """
            micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
            micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
            micro_f1 = get_micro_f1(micro_precision, micro_recall)
            
            print(f"Unclipped Micro Precision: {micro_precision}, Micro Recall: {micro_recall}, Micro F1: {micro_f1}")
            
            # Calculate metrics that count invalid predictions as both false positives and false negatives
            total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
                unnormalized_discrete_actions, gt_actions, action_space
            )

            micro_precision_without_invalids = get_micro_precision_from_counts(total_tp, valid_fp)
            micro_f1_without_invalids = get_micro_f1(micro_precision_without_invalids, micro_recall) # micro_recall is not affected

            print(f"Unclipped Micro Precision without invalids: {micro_precision_without_invalids}, \
                  Unclipped Micro F1 without invalids: {micro_f1_without_invalids}")

            clipped_predictions = np.clip(unnormalized_discrete_actions, action_space[0], action_space[-1])
            clipped_emr = get_exact_match_rate(clipped_predictions, gt_actions)
            clipped_total_tp, clipped_total_fp, clipped_total_fn, _, _ = calculate_tp_fp_fn_counts(
                clipped_predictions, gt_actions, action_space
            )
            clipped_micro_precision = get_micro_precision_from_counts(clipped_total_tp, clipped_total_fp)
            clipped_micro_recall = get_micro_recall_from_counts(clipped_total_tp, clipped_total_fn)
            clipped_micro_f1 = get_micro_f1(clipped_micro_precision, clipped_micro_recall)

            print(f"Clipped Micro Precision: {clipped_micro_precision}, Clipped Micro Recall: {clipped_micro_recall}, Clipped Micro F1: {clipped_micro_f1}")

            # Store results for this batch
            dataset_results.total_invalid_predictions += int(invalid_fp)  # invalid_fp is the same as the number of invalid predictions
            dataset_results.total_batches = counter
            dataset_results.total_timesteps += len(actions)
            dataset_results.total_emr += emr
            dataset_results.total_micro_precision += micro_precision
            dataset_results.total_micro_recall += micro_recall
            dataset_results.total_micro_f1 += micro_f1

            dataset_results.total_clipped_emr += clipped_emr
            dataset_results.total_clipped_micro_precision += clipped_micro_precision
            dataset_results.total_clipped_micro_recall += clipped_micro_recall
            dataset_results.total_clipped_micro_f1 += clipped_micro_f1

            dataset_results.total_micro_precision_without_invalids += micro_precision_without_invalids
            dataset_results.total_micro_f1_without_invalids += micro_f1_without_invalids

            # Memory management
            del transformed_dict, observation, actions, unnormalized_discrete_actions, \
                gt_actions, total_tp, total_fp, total_fn, clipped_predictions, \
                clipped_total_tp, clipped_total_fp, clipped_total_fn, \
                micro_precision, micro_recall, micro_f1, clipped_micro_precision, \
                clipped_micro_recall, clipped_micro_f1, emr, clipped_emr, \
                micro_precision_without_invalids, micro_f1_without_invalids
            gc.collect()
            jax.clear_caches()
            print(f"Processed {counter} episodes, cleared memory")

            # Uncomment to stop after 2 batches
            # if counter == 1:
            #     break

        dataset_results.avg_emr = dataset_results.total_emr / dataset_results.total_timesteps
        dataset_results.invalid_predictions_percentage = dataset_results.total_invalid_predictions / dataset_results.total_timesteps * 100
        dataset_results.avg_micro_precision = dataset_results.total_micro_precision / dataset_results.total_timesteps
        dataset_results.avg_micro_recall = dataset_results.total_micro_recall / dataset_results.total_timesteps
        dataset_results.avg_micro_f1 = dataset_results.total_micro_f1 / dataset_results.total_timesteps
        dataset_results.avg_clipped_emr = dataset_results.total_clipped_emr / dataset_results.total_timesteps
        dataset_results.avg_clipped_micro_precision = dataset_results.total_clipped_micro_precision / dataset_results.total_timesteps
        dataset_results.avg_clipped_micro_recall = dataset_results.total_clipped_micro_recall / dataset_results.total_timesteps
        dataset_results.avg_clipped_micro_f1 = dataset_results.total_clipped_micro_f1 / dataset_results.total_timesteps
        dataset_results.avg_micro_precision_without_invalids = dataset_results.total_micro_precision_without_invalids / dataset_results.total_timesteps
        dataset_results.avg_micro_f1_without_invalids = dataset_results.total_micro_f1_without_invalids / dataset_results.total_timesteps

        return dataset_results.to_dict()

def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with output_dir and dataset_dir
    """
    parser = argparse.ArgumentParser(
        description="Run inference on ProcGen datasets"
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
        help='Root directory containing the procgen datasets'
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
    
    # Initialize model and inference object
    config = pi0.Pi0Config(action_horizon=1) #We want to predict only for the next timestep
    tokenizer = PaligemmaTokenizer()
    key = jax.random.key(0)
    model = config.load(_model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")))
    print('Model loaded')
    procgen_inference = ProcGenInference(model, tokenizer, config)

    results_file = os.path.join(args.output_dir, 'pi0_base_procgen_results.json')

    # Get dataset shards
    procgen_dataset_list = os.listdir(args.dataset_dir) # Update path as needed
    for dataset in procgen_dataset_list:
        print(f'\n ---- EVALUATING {dataset} ---- \n')
        tfds_shards = os.listdir(f'{args.dataset_dir}/{dataset}') # Update path as needed
        tfds_sorted_shards = sorted(tfds_shards, key=lambda x: datetime.datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
        # Add path to shards
        tfds_sorted_shard_paths = [os.path.join(f'{args.dataset_dir}/{dataset}', shard) for shard in tfds_sorted_shards]


        # Get dataset stats and save to JSON file
        stats_output_path = os.path.join(args.output_dir, f'{dataset}_dataset_stats.json')
        if os.path.exists(stats_output_path):
            print(f'Dataset stats already exist at {stats_output_path}')
            with open(stats_output_path, 'r') as f:
                dataset_stats_dict = json.load(f)
            dataset_stats = {}
            dataset_stats['action'] = normalize.NormStats(**dataset_stats_dict['action'])
            
        else:
            dataset_stats_dict, dataset_stats = procgen_inference.get_dataset_stats(tfds_sorted_shard_paths, dataset_name=dataset)
            print('Dataset stats calculated: ', dataset_stats_dict)
            print(f'Saving dataset stats to {stats_output_path}')
            with open(stats_output_path, 'w') as f:
                json.dump(dataset_stats_dict, f, indent=4)

        # Create dataloader
        dataset_obj, dataloader = get_procgen_dataloader(tfds_sorted_shard_paths, batch_size=5)

        results = procgen_inference.evaluate_model(model, key, config, dataset_stats, dataloader, dataset)
    
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
