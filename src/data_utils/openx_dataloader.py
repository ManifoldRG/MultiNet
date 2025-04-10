from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict, Any

import tensorflow as tf
import numpy as np


class OpenXDataset(Dataset):
    def __init__(self, tfds_shards: List[str], dataset_name: str, by_episode: bool = False):
        self.tfds_shards = tfds_shards
        self.action_tensor_size = None
        self._action_stats = None
        self.cur_shard = None
        self.cur_shard_idx = None
        self.by_episode = by_episode
        self.dataset_name = dataset_name
        
        self.samples_per_shard = []
        for shard in self.tfds_shards:
            dataset = tf.data.Dataset.load(shard)
            self.samples_per_shard.append(len(dataset))
        
    def _process_shard(self, shard_idx):
        current_shard = []

        shard  = self.tfds_shards[shard_idx]
        dataset = tf.data.Dataset.load(shard)

        #Process the input data for each element in the shard
        for elem_idx, elem in enumerate(dataset):                
            concatenated_action_float = elem['action']
            float_action_tensors = []
            if isinstance(elem['action'], dict):
                elem['action'] = dict(sorted(elem['action'].items()))
                #Input processing
                for key, tensor in elem['action'].items():
                    if (tensor.dtype == tf.float32 or tensor.dtype==tf.float64):
                        if tensor.shape.ndims >= 2:
                        # Flatten the 2D tensor
                            tensor = tf.reshape(tensor, (-1,))
                        elif tensor.shape.ndims == 1:
                            tensor = tf.expand_dims(tensor, axis=0)
                            tensor = tf.reshape(tensor, (-1,))
                        elif tensor.shape.ndims == 0:
                            tensor = tf.reshape(tensor, (1, ))
                        float_action_tensors.append(tensor)

                #Concatenate all fields of continuous action space
                if float_action_tensors:
                    concatenated_action_float = tf.concat(float_action_tensors, axis=0).numpy()

            float_obs = {}
            if isinstance(elem['observation'], dict):
                #Input processing
                elem['observation'] = dict(sorted(elem['observation'].items()))
                for key, tensor in elem['observation'].items():
                    if 'language' not in key and 'image' not in key and 'pointcloud' not in key and 'rgb' not in key and 'instruction' not in key:
                        if (tensor.dtype == tf.float32 or tensor.dtype==tf.float64) and tensor.shape.ndims>=1 and not tf.reduce_any(tf.math.is_inf(tensor)):
                            float_obs[key] = tensor.numpy()

            #Processing image observation
            image_observation = None
            if 'image' in elem['observation']:
                image_observation = elem['observation']['image'].numpy().astype(np.uint8)
            elif 'hand_color_image' in elem['observation']:
                image_observation = elem['observation']['hand_color_image'].numpy().astype(np.uint8)
            elif 'agentview_rgb' in elem['observation']:
                image_observation = elem['observation']['agentview_rgb'].numpy().astype(np.uint8)
            elif 'hand_image' in elem['observation']:
                image_observation = elem['observation']['hand_image'].numpy().astype(np.uint8)
            elif 'wrist_image' in elem['observation']:
                image_observation = elem['observation']['wrist_image'].numpy().astype(np.uint8)
            elif 'rgb' in elem['observation']:
                image_observation = elem['observation']['rgb'].numpy().astype(np.uint8)

            #Processing text observation
            text_observation = None
            if 'instruction' in elem['observation'] and elem['observation']['instruction'].dtype == tf.int32:
                #Decode language table's tokenized instructions
                instruction = elem['observation']['instruction'].numpy()
                text_observation = bytes(instruction[np.where(instruction != 0)].tolist()).decode("utf-8")
            elif 'natural_language_instruction' in elem['observation']:
                elem['observation']['natural_language_instruction'] = elem['observation']['natural_language_instruction'].numpy()
                if isinstance(elem['observation']['natural_language_instruction'], bytes):
                    text_observation = elem['observation']['natural_language_instruction'].decode('utf-8')
                else:
                    text_observation = elem['observation']['natural_language_instruction'].numpy().decode('utf-8')
            elif 'language_instruction' in elem:
                if isinstance(elem['language_instruction'], bytes):
                    text_observation = elem['language_instruction'].decode('utf-8')
                else:
                    text_observation = elem['language_instruction'].numpy().decode('utf-8')
            elif 'natural_language_instruction' in elem:
                if isinstance(elem['natural_language_instruction'], bytes):
                    text_observation = elem['natural_language_instruction'].decode('utf-8')
                else:
                    text_observation = elem['natural_language_instruction'].numpy().decode('utf-8')
            elif 'action_instruct' in elem:
                if isinstance(elem['action_instruct'], bytes):
                    text_observation = elem['action_inst'].decode('utf-8')
                else:
                    text_observation = elem['action_instruct'].numpy().decode('utf-8')

            # Extract relevant features from the example
            step_data = {
                'text_observation': text_observation,
                'image_observation': image_observation,
                'action': concatenated_action_float,
                'reward': elem['reward'].numpy(),
                'is_last': elem['is_last'].numpy()
            }
            step_data.update(float_obs)                
            current_shard.append(step_data)
        return current_shard

    def _populate_action_stats(self):
        for shard in self.tfds_shards:
            dataset = tf.data.Dataset.load(shard)

            #Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):                
                concatenated_action_float = elem['action']
                float_action_tensors = []
                if isinstance(elem['action'], dict):
                    elem['action'] = dict(sorted(elem['action'].items()))
                    #Input processing
                    for key, tensor in elem['action'].items():
                        if (tensor.dtype == tf.float32 or tensor.dtype==tf.float64):
                            if tensor.shape.ndims >= 2:
                            # Flatten the 2D tensor
                                tensor = tf.reshape(tensor, (-1,))
                            elif tensor.shape.ndims == 1:
                                tensor = tf.expand_dims(tensor, axis=0)
                                tensor = tf.reshape(tensor, (-1,))
                            elif tensor.shape.ndims == 0:
                                tensor = tf.reshape(tensor, (1, ))
                            float_action_tensors.append(tensor)
                #Concatenate all fields of continuous action space
                if float_action_tensors:
                    concatenated_action_float = tf.concat(float_action_tensors, axis=0).numpy()
                    
                #Track the min, max, sum, and count of the action space
                if self.action_tensor_size is None:
                    self.action_tensor_size = concatenated_action_float.shape
                if self._action_stats is None:
                    self._action_stats = {
                        'min': np.full(self.action_tensor_size, np.inf),
                        'max': np.full(self.action_tensor_size, -np.inf),
                        'sum': np.zeros(self.action_tensor_size),
                        'count': 0
                    }
                self._action_stats['min'] = np.minimum(self._action_stats['min'], concatenated_action_float)
                self._action_stats['max'] = np.maximum(self._action_stats['max'], concatenated_action_float)
                self._action_stats['sum'] += concatenated_action_float
                self._action_stats['count'] += 1
    
    @property
    def action_stats(self):
        if self._action_stats is None:
            self._populate_action_stats()
            self._action_stats['mean'] = self._action_stats['sum'] / self._action_stats['count']
            self._action_stats['size'] = self.action_tensor_size
        return self._action_stats
        

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
        text_observation = []
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
        for key, value in item.items(): # text_observation, image_observation, action, reward, is_last..
            result[key].append(value)
    return result

def get_openx_dataloader(tfds_shards: List[str], batch_size: int, dataset_name: str, num_workers: int = 0, by_episode=False) -> DataLoader:
    dataset = OpenXDataset(tfds_shards, dataset_name, by_episode=by_episode)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate
    )
