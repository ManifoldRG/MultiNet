import datetime
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
import os
import gc

# Configure JAX memory settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

class ProcGenInference:
    def __init__(self, model: _model.Model, tokenizer: PaligemmaTokenizer, config: pi0.Pi0Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def prepare_observation(self, obs_dict: dict, batch_size: int = 1, action_dim: int = 32, max_token_length: int = 48) -> dict:
    #Prepare observation dictionary for pi0 model inference
        tokenizer = self.tokenizer
        
        # Process image observation
        base_image = obs_dict["image_observation"]
        if isinstance(base_image, tf.Tensor):
            base_image = base_image.numpy()
        
        # Add batch dimension if needed
        if len(base_image.shape) == 3:
            base_image = base_image[None, ...] # Add batch dimension
            
        # Create zero images for missing views and convert to jax array
        base_image = jax.numpy.array(base_image)
        zero_image = jax.numpy.zeros_like(base_image)
        
        # Process text observation - Procgen does not have a text observation per timestep, so we use the description of the environment/task as the prompt
        text_obs = obs_dict["text_observation"]
        if isinstance(text_obs, tf.Tensor):
            text_obs = text_obs.numpy().decode('utf-8')
            
        # Tokenize text prompt/observation
        tokens, token_mask = tokenizer.tokenize(text_obs)
        tokens = jax.numpy.array(tokens)[None, ...] # Add batch dimension
        token_mask = jax.numpy.array(token_mask)[None, ...] # Add batch dimension
        
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
    config = pi0.Pi0Config(action_dim=32, action_horizon=50, max_token_len=48)
    tokenizer = PaligemmaTokenizer()
    key = jax.random.key(0)
    model = config.load(_model.restore_params(download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")))
    print('Model loaded')
    procgen_inference = ProcGenInference(model, tokenizer, config)
   

    # Get dataset shards
    dataset_name = "coinrun"  # Example Procgen environment
    tfds_shards = [f"path/to/procgen/{dataset_name}/translated_shard_0"]  # Update path as needed
    tfds_sorted_shards = sorted(tfds_shards, key=lambda x: datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
    
    # Create dataloader
    dataset, dataloader = get_procgen_dataloader(tfds_sorted_shards, batch_size=1, by_episode=True)
    
    counter = 0
    for batch in dataloader:
        # Process each timestep in episode
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
            print(f"Processed {counter} episodes, cleared memory")

if __name__ == "__main__":
    main()
