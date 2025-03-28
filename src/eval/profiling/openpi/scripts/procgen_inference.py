import argparse
import datetime
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from openpi.models import pi0
from openpi.models import model as _model
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.transforms import pad_to_dim
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from definitions.procgen import ProcGenDefinitions
from openpi.shared import download
import jax
import numpy as np
import tensorflow as tf
import gc
from openpi.shared import normalize
from openpi.transforms import Unnormalize
from openpi.shared.normalize import RunningStats
# Configure JAX memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

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

    def get_dataset_stats(self, root_dir: str):
        
        running_stats = RunningStats()

        print('Calculating dataset stats...')
        #Load the dataset shards and calculate the stats
        for shard in root_dir:
            print(f'Processing shard: {shard}')
            dataset = tf.data.Dataset.load(shard)
            actions = []
            for elem in dataset:
                actions.append(elem['actions'][0].numpy()) # Procgen has 1D action space
            
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
        return {'action': stats}

    def process_output(self, actions, dataset_stats: dict):
        """
        Unnormalize the model's action outputs using stored normalization statistics.
        
        Args:
            actions (jax.numpy.ndarray): Normalized actions from the model
            
        Returns:
            np.ndarray: Unnormalized actions scaled back to the original action space
        """
        # Convert to numpy array if actions is a jax array
        actions = np.array(actions)
        # Get only first dimension since Procgen uses 1D action space
        actions = actions[..., 0:1]  # Keep the first dimension while preserving batch dimensions
        
        # Load normalization statistics
        norm_stats = dataset_stats
        

        print('Action before unnormalization: ', actions)
        # Create and apply unnormalize transform
        unnormalizer = Unnormalize(norm_stats=norm_stats)
        unnormalized_actions = unnormalizer({'action': actions})['action']
        print('Action after unnormalization: ', unnormalized_actions)

        """Discretize the actions after scaling them back to the original action space"""
        unnormalized_actions = np.round(unnormalized_actions, 0)
        
        return unnormalized_actions

    def evaluate_model(self, model, key, config, dataset_stats: dict, dataloader: tf.data.Dataset):
        """Evaluate the model on the dataset"""
        counter = 0
        for batch in dataloader:
            '''# Process each timestep in episode
            for timestep_idx in range(len(batch['image_observation'])):
                obs = {
                    'image_observation': batch['image_observation'][0][timestep_idx],
                    'text_observation': batch['text_observation'][0][timestep_idx]
                }
                
                # Transform observation
                transformed_dict = procgen_inference.prepare_observation(obs, max_token_length=config.max_token_len)
                observation = Observation.from_dict(transformed_dict) # Process according to model input format
                
                # Sample actions
                actions = model.sample_actions(key, observation, num_steps=10)
                procgen_action = actions[0] #Procgen has an action space of 1 dimension and discrete 
                print(f"Timestep {timestep_idx} actions:", procgen_action)


                
                # Memory management
                del transformed_dict, observation, actions

            # Clear memory every 10 episodes
            counter += 1
            if counter % 10 == 0:
                gc.collect()
                jax.clear_caches()
                print(f"Processed {counter} episodes, cleared memory")'''
            
            # Process entire batch at once
            obs = {
                'image_observation': batch['image_observation'],  # Full batch
                'text_observation': batch['text_observation']     # Full batch
            }
            
            # Transform observation
            transformed_dict = self.prepare_observation(obs, max_token_length=config.max_token_len)
            observation = Observation.from_dict(transformed_dict)
            
            # Sample actions for entire episode
            actions = model.sample_actions(key, observation, num_steps=10)
            unnormalized_actions = self.process_output(actions, dataset_stats)
            counter += 1
            print(f"Batch {counter} actions:", unnormalized_actions)
            
            # Memory management
            del transformed_dict, observation, actions, unnormalized_actions
            
            # Clear memory every 10 episodes
            #counter += 1
            #if counter % 10 == 0:
            gc.collect()
            jax.clear_caches()
            print(f"Processed {counter} episodes, cleared memory")
        
        
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
   

    # Get dataset shards
    procgen_dataset_list = os.listdir(args.dataset_dir) # Update path as needed
    for dataset in procgen_dataset_list:
        print(f'\n ---- EVALUATING {dataset} ---- \n')
        tfds_shards = os.listdir(f'{args.dataset_dir}/{dataset}') # Update path as needed
        tfds_sorted_shards = sorted(tfds_shards, key=lambda x: datetime.datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
        # Add path to shards
        tfds_sorted_shard_paths = [os.path.join(f'{args.dataset_dir}/{dataset}', shard) for shard in tfds_sorted_shards]


        # Get dataset stats
        dataset_stats = procgen_inference.get_dataset_stats(tfds_sorted_shard_paths)
        print('Dataset stats calculated: ', dataset_stats)

        # Save dataset stats to JSON file
        stats_output_path = os.path.join(args.output_dir, f'{dataset}_dataset_stats.json')
        print(f'Saving dataset stats to {stats_output_path}')
        with open(stats_output_path, 'w') as f:
            json.dump(dataset_stats, f, indent=4)
        

        # Create dataloader
        dataset, dataloader = get_procgen_dataloader(tfds_sorted_shard_paths, batch_size=5)

        procgen_inference.evaluate_model(model, key, config, dataset_stats, dataloader)
    
    

if __name__ == "__main__":
    main()
