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
            if self.dataset_name == "robot_vqa":
                elem["action"] = None
                elem['reward'] = None                
            concatenated_action_float = elem['action']
            float_action_tensors = []
            # Store the original action dictionary if it's a dict, otherwise None
            action_dict = None
            if isinstance(elem['action'], dict):
                # Store the original action dictionary
                action_dict = {key: tensor.numpy() for key, tensor in elem['action'].items()}
                
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
                    if 'action' not in key and 'act' not in key and'language' not in key and 'image' not in key and 'pointcloud' not in key and 'rgb' not in key and 'instruction' not in key:
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
                    text_observation = elem['action_instruct'].decode('utf-8')
                else:
                    text_observation = elem['action_instruct'].numpy().decode('utf-8')

            #Check for RoboVQA and include additional features
            text_answer = None
            if 'raw_text_question' in elem['observation'] and 'raw_text_answer' in elem['observation']:
                #Pick only the last image from the list of images depicting the progression of the scene - this is the image reqd for the question
                image_observation = elem['observation']['images'][-1].numpy().astype(np.uint8)
                text_observation = elem['observation']['raw_text_question'].numpy().decode('utf-8')
                text_answer = elem['observation']['raw_text_answer'].numpy().decode('utf-8')

            if self.dataset_name == "robot_vqa":
                parts = text_observation.split('Q:', 1)
                if len(parts) == 1:
                    text_observation_multi_embodiment = text_observation
                elif len(parts) == 2 and text_observation.startswith("Q:"):
                    text_observation_multi_embodiment = f"Question: {parts[1].strip()}"
                else:
                    text_observation_multi_embodiment = f"Task and Context: {parts[0].strip()}\n Question: {parts[1].strip()}"

            # Extract relevant features from the example
            step_data = {
                'text_observation': text_observation if self.dataset_name != "robot_vqa" else text_observation_multi_embodiment,
                'image_observation': image_observation,
                'action': concatenated_action_float,
                'action_dict': action_dict,
                'reward': elem['reward'].numpy() if self.dataset_name != "robot_vqa" else elem['reward'],
                'is_last': elem['is_last'].numpy(),
                'text_answer': text_answer
            }
            step_data.update(float_obs)                
            current_shard.append(step_data)
        return current_shard

    def _populate_action_stats(self):
        all_actions = []  # Store all actions for quantile calculation
        
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
                        'sum_of_squares': np.zeros(self.action_tensor_size),
                        'count': 0
                    }
                self._action_stats['min'] = np.minimum(self._action_stats['min'], concatenated_action_float)
                self._action_stats['max'] = np.maximum(self._action_stats['max'], concatenated_action_float)
                self._action_stats['sum'] += concatenated_action_float
                self._action_stats['count'] += 1
                
                # Store action for quantile calculation
                all_actions.append(concatenated_action_float)
        
        # Calculate quantiles if we have actions
        if all_actions:
            stacked_actions = np.stack(all_actions)
            self._action_stats['q01'] = np.quantile(stacked_actions, 0.01, axis=0)
            self._action_stats['q99'] = np.quantile(stacked_actions, 0.99, axis=0)


    @property
    def action_stats(self):
        if self._action_stats is None:
            self._populate_action_stats()
            # Check if _populate_action_stats actually populated data
            if self._action_stats is None:
                # If no action data was found, return a default structure
                return {
                    'min': [],
                    'max': [],
                    'sum': [],
                    'mean': [],
                    'std': [],
                    'count': 0,
                    'size': (0,)
                }

            self._action_stats['mean'] = self._action_stats['sum'] / self._action_stats['count']
            # Compute variance using E[X²] - E[X]²
            mean_of_squares = self._action_stats['sum_of_squares'] / self._action_stats['count']
            variance = mean_of_squares - (self._action_stats['mean'] ** 2)
            self._action_stats['std'] = np.sqrt(variance + 1e-8)
            self._action_stats['size'] = self.action_tensor_size

            # Convert numpy arrays to lists for JSON serialization
            for key in ['min', 'max', 'sum', 'mean', 'std', 'q01', 'q99']:
                if key in self._action_stats and hasattr(self._action_stats[key], 'tolist'):
                    self._action_stats[key] = self._action_stats[key].tolist()

            # Convert action_tensor_size tuple to list for JSON compatibility
            if hasattr(self.action_tensor_size, '__iter__'):
                self._action_stats['size'] = list(self.action_tensor_size)
            else:
                self._action_stats['size'] = [self.action_tensor_size] if self.action_tensor_size is not None else []
        return self._action_stats
        

    @action_stats.setter
    def action_stats(self, value):
        self._action_stats = value
        
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
        action_dict = []
        reward = []
        is_last = []
        text_answer = []

        for timestep in episode:
            text_observation.append(timestep['text_observation'])
            timestep.pop('text_observation')
            image_observation.append(timestep['image_observation'])
            timestep.pop('image_observation')
            concatenated_action_float.append(timestep['action'])
            timestep.pop('action')
            action_dict.append(timestep['action_dict'])
            timestep.pop('action_dict')
            reward.append(timestep['reward'])
            timestep.pop('reward')
            is_last.append(timestep['is_last'])
            timestep.pop('is_last')
            text_answer.append(timestep['text_answer'])
            timestep.pop('text_answer')

            for key, value in timestep.items():
                if key not in etc_observations:
                    etc_observations[key] = []
                etc_observations[key].append(value)

        result = {
            'text_observation': text_observation,
            'image_observation': image_observation,
            'action': concatenated_action_float,
            'action_dict': action_dict,
            'reward': reward,
            'is_last': is_last,
            'text_answer': text_answer
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
