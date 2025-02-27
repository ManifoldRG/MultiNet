import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class ProcgenDataset(Dataset):
    def __init__(self, tfds_shards: List[str]):
        """
        Initialize the Procgen dataset.
        
        Args:
            tfds_shards: List of paths to the translated Procgen dataset shards
        """
        self.tfds_shards = tfds_shards
        self.current_elem_idx = 0
        self.current_shard_idx = 0

    def _process_shards(self):
        """
        Process the shards of the dataset, yielding episodes.
        
        Each episode consists of a sequence of timesteps with observations, actions, and rewards.
        """
        current_episode = []

        for shard_idx, shard in enumerate(self.tfds_shards):
            # Skip shards that have already been processed
            if shard_idx < self.current_shard_idx:
                continue
                
            dataset = tf.data.Dataset.load(shard)

            # Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):
                # Skip elements that have already been processed
                if shard_idx == self.current_shard_idx and elem_idx < self.current_elem_idx:
                    continue

                # Process observations (images)
                if 'observations' in elem:
                    # Procgen observations are RGB images
                    observation = elem['observations'].numpy()
                    # Ensure the observation is in the right format (H, W, C) and uint8
                    if observation.dtype != np.uint8:
                        observation = (observation * 255).astype(np.uint8)
                else:
                    observation = None

                # Process actions
                if 'actions' in elem:
                    action = elem['actions'].numpy()
                else:
                    action = None

                # Process rewards
                if 'rewards' in elem:
                    reward = elem['rewards'].numpy()
                else:
                    reward = 0.0

                # Determine if this is the last step in an episode
                is_last = False
                if 'dones' in elem:
                    is_last = bool(elem['dones'].numpy())
                
                # Create a dictionary for this timestep
                step_data = {
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'is_last': is_last
                }
                
                current_episode.append(step_data)
                
                # If this is the last step in an episode, yield the episode
                if is_last:
                    if elem_idx + 1 == len(dataset):
                        self.current_elem_idx = 0
                        self.current_shard_idx = shard_idx + 1
                    else:
                        self.current_elem_idx = elem_idx + 1
                        self.current_shard_idx = shard_idx
                    yield current_episode
                    current_episode = []

        # Yield any remaining episode
        if current_episode:
            self.current_shard_idx = 0
            self.current_elem_idx = 0
            yield current_episode
            current_episode = []

    def __len__(self) -> int:
        """
        Return the number of episodes in the dataset.
        """
        return sum(1 for _ in self._process_shards())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an episode by index.
        
        Args:
            idx: Index of the episode to retrieve
            
        Returns:
            A dictionary containing the processed episode data
        """
        if idx == 0:    
            self.current_shard_idx = 0
            self.current_elem_idx = 0
        for i, episode in enumerate(self._process_shards()):
            return self._process_episode(episode)
        raise IndexError("Episode index out of range")

    def _process_episode(self, episode: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process an episode into a format suitable for training or evaluation.
        
        Args:
            episode: List of timesteps in the episode
            
        Returns:
            A dictionary containing the processed episode data
        """
        observations = []
        actions = []
        rewards = []
        is_last = []

        for timestep in episode:
            observations.append(timestep['observation'])
            actions.append(timestep['action'])
            rewards.append(timestep['reward'])
            is_last.append(timestep['is_last'])

        return {
            'observation': observations,
            'action': actions,
            'reward': rewards,
            'is_last': is_last
        }

def procgen_custom_collate(batch):
    """
    Custom collate function for the Procgen dataset.
    
    Args:
        batch: A batch of episodes
        
    Returns:
        A dictionary containing the collated batch data
    """
    # Initialize dictionaries to store the collected data
    collected_data = {
        'observation': [],
        'action': [],
        'reward': [],
        'is_last': []
    }

    # Collect data from the batch
    for item in batch:
        for key in collected_data:
            if item[key] is not None:
                collected_data[key].append(item[key])

    result = {}
    for key, value in collected_data.items():
        if value:  # Check if the list is not empty and is not None
            result[key] = value  # Keep as list for non-numeric data
        else:
            result[key] = None  # If no valid data, set to None

    return result

def get_procgen_dataloader(tfds_shards: List[str], batch_size: int, num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for the Procgen dataset.
    
    Args:
        tfds_shards: List of paths to the translated Procgen dataset shards
        batch_size: Batch size for the DataLoader
        num_workers: Number of worker processes for the DataLoader
        
    Returns:
        A PyTorch DataLoader for the Procgen dataset
    """
    dataset = ProcgenDataset(tfds_shards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=procgen_custom_collate
    )