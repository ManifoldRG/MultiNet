import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

class OpenXDataset(Dataset):
    def __init__(self, tfds_shards: List[str]):
        self.tfds_shards = tfds_shards
        self.current_elem_idx = 0
        self.current_shard_idx = 0

    def _process_shards(self):
        
        current_episode = []

        for shard_idx, shard in enumerate(self.tfds_shards):

            # To prevent input processing overhead for shards that have already been processed
            if shard_idx < self.current_shard_idx:
                continue
            #print(shard)
            dataset = tf.data.Dataset.load(shard)

            #Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):

                # To prevent input processing overhead for elements of shardsthat have already been processed
                if shard_idx == self.current_shard_idx and elem_idx < self.current_elem_idx:
                    continue
                    

                discrete_observations = None
                concatenated_action_float = elem['action']
                float_action_tensors = []
                if isinstance(elem['action'], dict):
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
                        concatenated_action_float = tf.concat(float_action_tensors, axis=0)
                    
                    #Final concatenated action space shape
                    #if isinstance(concatenated_action_float, tf.Tensor):
                        #print('Concatenated action space shape')
                        #print(concatenated_action_float.shape)
                    
                        #print('\n')

                concatenated_obs_float = tf.Variable([0.0, 0.0], dtype=tf.float32).numpy()
                float_obs_tensors = []
                if isinstance(elem['observation'], dict):
                    #Input processing
                    for key, tensor in elem['observation'].items():
                        if 'language' not in key and 'image' not in key and 'pointcloud' not in key and 'rgb' not in key and 'instruction' not in key:
                            if (tensor.dtype == tf.float32 or tensor.dtype==tf.float64) and tensor.shape.ndims>=1 and not tf.reduce_any(tf.math.is_inf(tensor)):
                                if tensor.shape.ndims >= 2:
                                # Flatten the 2D tensor
                                    tensor = tf.reshape(tensor, (-1,))
                                elif tensor.shape.ndims == 1:
                                    tensor = tf.expand_dims(tensor, axis=0)
                                    tensor = tf.reshape(tensor, (-1,))
                                elif tensor.shape.ndims == 0:
                                    tensor = tf.reshape(tensor, (1, ))
                                float_obs_tensors.append(tensor)

                    #Concatenate all fields of continuous observation space
                    if float_obs_tensors:
                        #print(float_obs_tensors)
                        concatenated_obs_float = tf.concat(float_obs_tensors, axis=0) 


                    else:
                        #Dummy observation space if no observation other than images or language instruction
                        concatenated_obs_float = tf.Variable([0.0, 0.0], dtype=tf.float32).numpy()

                    #Final concatenated observation space shape
                    if isinstance(concatenated_obs_float, tf.Tensor):
                        #print('Concatenated observation Space shape')
                        #print(concatenated_obs_float.shape)
                        concatenated_obs_float = concatenated_obs_float.numpy()

                #Processing image observation
                img_obs_pil = None
                if 'image' in elem['observation']:
                    img_obs = elem['observation']['image'].numpy().astype(np.uint8)
                    if img_obs.shape[2] == 3:
                        fourth_channel = img_obs[:,:,0]  # Use the red channel as the fourth channel
                        img_4channel = np.dstack((img_obs, fourth_channel))
                        img_obs_pil = Image.fromarray(img_4channel)
                    elif img_obs.shape[2] == 2:
                        # If the image has 2 channels, duplicate channels to create a 4-channel image
                        img_4channel = np.dstack((img_obs, img_obs))
                        img_obs_pil = Image.fromarray(img_4channel)
                    else:
                        img_obs_pil = Image.fromarray(img_obs)
                elif 'rgb' in elem['observation']:
                    img_obs = elem['observation']['rgb'].numpy().astype(np.uint8)
                    if img_obs.shape[2] == 3:
                        fourth_channel = img_obs[:,:,0]  # Use the red channel as the fourth channel
                        img_4channel = np.dstack((img_obs, fourth_channel))
                        img_obs_pil = Image.fromarray(img_4channel)
                    elif img_obs.shape[2] == 2:
                        # If the image has 2 channels, duplicate channels to create a 4-channel image
                        img_4channel = np.dstack((img_obs, img_obs))
                        img_obs_pil = Image.fromarray(img_4channel)
                    else:
                        img_obs_pil = Image.fromarray(img_obs)

                #Processing text observation
                text_observation = None
                if 'instruction' in elem['observation'] and elem['observation']['instruction'].dtype == tf.int32:
                    #Dummy discrete observations for text observations to work
                    discrete_observations = tf.constant([0]).numpy()
                    #Decode language table's tokenized instructions
                    instruction = elem['observation']['instruction'].numpy()
                    text_observation = bytes(instruction[np.where(instruction != 0)].tolist()).decode("utf-8")
                elif 'natural_language_instruction' in elem['observation']:
                    #Dummy discrete observations for text observations to work
                    discrete_observations = tf.constant([0]).numpy()
                    elem['observation']['natural_language_instruction'] = elem['observation']['natural_language_instruction'].numpy()
                    if isinstance(elem['observation']['natural_language_instruction'], bytes):
                        text_observation = elem['observation']['natural_language_instruction'].decode('utf-8')
                    else:
                        text_observation = elem['observation']['natural_language_instruction'].numpy().decode('utf-8')
                elif 'language_instruction' in elem:
                    #Dummy discrete observations for text observations to work
                    discrete_observations = tf.constant([0]).numpy()
                    if isinstance(elem['language_instruction'], bytes):
                        text_observation = elem['language_instruction'].decode('utf-8')
                    else:
                        text_observation = elem['language_instruction'].numpy().decode('utf-8')
                elif 'natural_language_instruction' in elem:
                    #Dummy discrete observations for text observations to work
                    discrete_observations = tf.constant([0]).numpy()
                    if isinstance(elem['natural_language_instruction'], bytes):
                        text_observation = elem['natural_language_instruction'].decode('utf-8')
                    else:
                        text_observation = elem['natural_language_instruction'].numpy().decode('utf-8')
                elif 'action_instruct' in elem:
                    if isinstance(elem['action_instruct'], bytes):
                        text_observation = elem['action_inst'].decode('utf-8')
                    else:
                        text_observation = elem['action_instruct'].numpy().decode('utf-8')

                #print('\nText observation')
                #print(text_observation)
                # Extract relevant features from the example
                step_data = {
                    'continuous_observation': concatenated_obs_float,
                    'text_observation': text_observation,
                    'image_observation': img_obs_pil,
                    'discrete_observation': discrete_observations,
                    'action': concatenated_action_float,
                    'reward': elem['reward'],
                    'is_last': elem['is_last']
                }
                
                current_episode.append(step_data)
                
                if step_data['is_last']:
                    #episodes.append(current_episode)
                    #print(len(current_episode))
                    #print(current_episode[-1])
                    if elem_idx+1 == len(dataset):
                        self.current_elem_idx = 0
                        self.current_shard_idx = shard_idx+1
                    else:
                        self.current_elem_idx = elem_idx+1
                        self.current_shard_idx = shard_idx
                    yield current_episode
                    current_episode = []

        if current_episode:  # Add the last episode if it's not empty
            #print(len(current_episode))
            #print(current_episode[-1])
            self.current_shard_idx = 0
            self.current_elem_idx = 0
            yield current_episode
            current_episode = []
            #episodes.append(current_episode)

        #return episodes

    def _get_feature(self, elem, feature_name: str) -> Any:
        # Implement feature extraction based on your TFDS structure
        # This is a placeholder and needs to be adjusted based on your data
        try:
            return elem[feature_name]
        except KeyError:
            return None

    def __len__(self) -> int:
        return sum(1 for _ in self._process_shards())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx == 0:    
            self.current_shard_idx = 0
            self.current_elem_idx = 0
        for i, episode in enumerate(self._process_shards()):
            #if i == idx:
                #print(self.current_shard_idx, self.current_elem_idx)
            return self._process_episode(episode)
        raise IndexError("Episode index out of range")

    def _process_episode(self, episode: List[Dict[str, Any]]) -> Dict[str, Any]:

        concatenated_obs_float = []
        text_observation = []
        img_obs_pil = [] 
        discrete_observations = []
        concatenated_action_float = []
        reward = []
        is_last = []

        for timestep in episode:

            concatenated_obs_float.append(timestep['continuous_observation'])
            text_observation.append(timestep['text_observation'])
            img_obs_pil.append(timestep['image_observation'])
            discrete_observations.append(timestep['discrete_observation'])
            concatenated_action_float.append(timestep['action'])
            reward.append(timestep['reward'])
            is_last.append(timestep['is_last'])


        return {
                    'continuous_observation': concatenated_obs_float,
                    'text_observation': text_observation,
                    'image_observation': img_obs_pil ,
                    'discrete_observation': discrete_observations,
                    'action': concatenated_action_float,
                    'reward': reward,
                    'is_last': is_last
                }
    

def custom_collate(batch):
    # Initialize dictionaries to store the collected data
    collected_data = {
        'continuous_observation': [],
        'text_observation': [],
        'image_observation': [],
        'discrete_observation': [],
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

def get_openx_dataloader(tfds_shards: List[str], batch_size: int, num_workers: int = 0) -> DataLoader:
    dataset = OpenXDataset(tfds_shards)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn= custom_collate
    )

