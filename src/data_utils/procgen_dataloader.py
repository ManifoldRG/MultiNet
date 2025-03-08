from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any
from collections import defaultdict
from pathlib import Path
from definitions.procgen import ProcGenDefinitions
import tensorflow as tf
import numpy as np


class ProcGenDataset(Dataset):
    def __init__(self, tfds_shards: List[str], by_episode: bool = False):
        self.tfds_shards = tfds_shards
        self.action_tensor_size = None
        self.action_stats = None
        self.action_stats_populated = False
        self.cur_shard = None
        self.cur_shard_idx = None
        self.by_episode = by_episode
        
        self.samples_per_shard = []
        for shard in self.tfds_shards:
            dataset = tf.data.Dataset.load(shard)
            self.samples_per_shard.append(len(dataset))
        
    def _process_shard(self, shard_idx):
        current_shard = []

        shard  = self.tfds_shards[shard_idx]
        dataset = tf.data.Dataset.load(shard)
        dataset_name = Path(shard).parts[-2]
        #Process the input data for each element in the shard
        for elem_idx, elem in enumerate(dataset):                
            action = elem['actions'].numpy()
            
            image_observation = elem['observations'].numpy().astype(np.uint8)
            image_observation = np.moveaxis(image_observation, 0, 2)
            
            # Extract relevant features from the example
            step_data = {
                'text_observation': list(ProcGenDefinitions.DESCRIPTIONS[dataset_name].keys())[0],
                'image_observation': image_observation,
                'action': action,
                'reward': elem['rewards'].numpy(),
                'is_last': elem['dones'].numpy()
            }

            if self.action_stats is None and self.action_stats_populated == False:
                self.action_stats = {
                    'size': action.shape
                }
                self.action_stats_populated = True
            
            current_shard.append(step_data)
        return current_shard

    def _get_action_stats(self):
        if self.action_stats is None:
            raise AttributeError("action_stats is None, it has not been populated yet")
        if self.action_stats_populated == False:
            raise ValueError("Action stats population is not finished yet.")
        return self.action_stats
        
    def _get_feature(self, elem, feature_name: str) -> Any:
        # Implement feature extraction based on your TFDS structure
        # This is a placeholder and needs to be adjusted based on your data
        try:
            return elem[feature_name]
        except KeyError:
            return None

    def __len__(self) -> int:
        if self.by_episode:
            return len(self.tfds_shards)
        return sum(self.samples_per_shard)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.by_episode:
            if idx < 0:
                idx = len(self.tfds_shards) + idx
            
            if self.cur_shard_idx != idx:
                self.cur_shard = self._process_shard(idx)
                self.cur_shard_idx = idx
            return self._process_episode(self.cur_shard)
        
        if idx < 0:
            idx = sum(self.samples_per_shard) + idx
        
        samples_so_far = 0
        for i in range(len(self.tfds_shards)):
            samples_so_far += self.samples_per_shard[i]
            if idx < samples_so_far:
                break
            
        if self.cur_shard_idx != i:
            self.cur_shard = self._process_shard(i)
            self.cur_shard_idx = i
        
        return self.cur_shard[idx - (samples_so_far - self.samples_per_shard[i])]
        
    def _process_episode(self, episode: List[Dict[str, Any]]) -> Dict[str, Any]:
        image_observation = []
        etc_observations = {}
        concatenated_action_float = []
        reward = []
        is_last = []

        for timestep in episode:
            text_observation.append(timestep['text_observation'])
            timestep.pop('text_observation')
            image_observation.append(timestep['image_observation'])
            timestep.pop('image_observation')
            concatenated_action_float.append(timestep['action'])
            timestep.pop('action')
            reward.append(timestep['reward'])
            timestep.pop('reward')
            is_last.append(timestep['is_last'])
            timestep.pop('is_last')

            for key, value in timestep.items():
                if key not in etc_observations:
                    etc_observations[key] = []
                etc_observations[key].append(value)

        result = {
            'text_observation': text_observation,
            'image_observation': image_observation,
            'action': concatenated_action_float,
            'reward': reward,
            'is_last': is_last
        }
        result.update(etc_observations)

        return result

def custom_collate(batch):
    result = defaultdict(list)
    for item in batch: 
        for key, value in item.items(): # image_observation, action, reward, is_last..
            result[key].append(value)
    return result

def get_procgen_dataloader(tfds_shards: List[str], batch_size: int, num_workers: int = 0, by_episode=False) -> DataLoader:
    dataset = ProcGenDataset(tfds_shards, by_episode=by_episode)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn= custom_collate
    )
