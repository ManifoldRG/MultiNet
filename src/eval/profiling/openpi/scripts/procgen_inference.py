import datetime
import os
import sys
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

def main():
    # Initialize model
    config = pi0.Pi0Config(action_horizon=1) #We want to predict only for the next timestep
    tokenizer = PaligemmaTokenizer()
    key = jax.random.key(0)
    model = config.load(_model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")))
    print('Model loaded')
    procgen_inference = ProcGenInference(model, tokenizer, config)
   

    # Get dataset shards
    tfds_shards = os.listdir('../../../../../../bigfish') # Update path as needed
    tfds_sorted_shards = sorted(tfds_shards, key=lambda x: datetime.datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
    # Add path to shards
    tfds_sorted_shard_paths = [os.path.join('../../../../../../bigfish', shard) for shard in tfds_sorted_shards]

    # Create dataloader
    dataset, dataloader = get_procgen_dataloader(tfds_sorted_shard_paths, batch_size=8)
    
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
        transformed_dict = procgen_inference.prepare_observation(obs, max_token_length=config.max_token_len)
        observation = Observation.from_dict(transformed_dict)
        
        # Sample actions for entire episode
        actions = model.sample_actions(key, observation, num_steps=10)
        print(f"Batch {counter} actions:", actions)
        
        # Memory management
        del transformed_dict, observation, actions
        
        # Clear memory every 10 episodes
        #counter += 1
        #if counter % 10 == 0:
        gc.collect()
        jax.clear_caches()
        print(f"Processed {counter} episodes, cleared memory")

if __name__ == "__main__":
    main()
