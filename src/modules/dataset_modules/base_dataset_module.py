from dataclasses import dataclass
from abc import abstractmethod, ABC
from src.modules.modality_modules.vlm_module import VLMModule

from typing import Any
from pathlib import Path

import tensorflow as tf
import numpy as np
import json
import string
import os
import warnings

class DatasetModule(ABC):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0) -> None:
        self._definitions_class = None
        self.get_dataloader_fn = None 
        self.dataset_family = None
        self.format_instruction_prompt_fn = None

        self.disk_root_dir = disk_root_dir
        self.batch_size = batch_size
        self.modality_module = None
        if modality == 'vlm':
            self.modality_module = VLMModule(source, model, max_concurrent_prompts=self.batch_size)
        assert self.modality_module is not None, "The modality module has not been set correcly. Check required."

        self.k_shots = k_shots
        self.action_stats = None
        self._datasets = []
        
    @property
    def datasets(self):
        if len(self._datasets) == 0:
            for dataset in list(self.descriptions.keys()):
                tfds_shards = self._find_shards(dataset)
                if len(tfds_shards) != 0:
                    self._datasets.append(dataset)
        return self._datasets
        
    @property
    def descriptions(self):
        return self._definitions_class.DESCRIPTIONS

    @property
    def action_spaces(self):
        return self._definitions_class.ACTION_SPACES

    @property
    def action_exclusiveness(self):
        return self._definitions_class.ACTION_EXCLUSIVENESS
    
    @property
    def additional_instructions(self):
        if self._definitions_class.ADDITIONAL_INSTRUCTIONS is None:
            return {}
        return self._definitions_class.ADDITIONAL_INSTRUCTIONS

    # Take np array of int labels and output one hot encoding
    def _get_one_hot(self, targets: np.ndarray, nb_classes: int):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    # Main evaluation function.
    def run_eval(self) -> None:
        # Since OpenX consists of multiple datasets, a modality module should be initialized per every evaluation step for each dataset.

        total_results = {}
        for dataset in self.datasets:
            
            if os.path.exists('<path to results>'):
                with open('<path to results>', 'r') as f:
                    completed_datasets = json.load(f)
            
                if dataset in completed_datasets:
                    print(f'\nSkipping dataset: {dataset} (already evaluated)\n')
                    continue
        
            result = self._run_eval_dataset(dataset)
            total_results[dataset] = result
            
            if os.path.exists('<path to results>'):
                # If it exists, load the existing data
                with open('<path to results>', 'r') as f:
                    existing_results = json.load(f)
                # Append new data to existing data
                existing_results.update(total_results)
            else:
                # If it doesn't exist, use the current eval_results
                existing_results = total_results

            # Write the updated or new results to the file
            with open('<path to results>', 'w') as f:
                json.dump(existing_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            self.action_stats = None

    def _process_episode(self, episode):
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
    
    @abstractmethod
    # Each dataset needs its own eval scheme
    def _run_eval_dataset(self, dataset) -> dict:
        pass
    
    @abstractmethod
    # Each dataset needs its own finding and sorting logic for shard files
    def _find_shards(self, dataset: str) -> list[str]:
        pass

    # Forming the batch.
    def _process_batch(self, batch: dict[str, list[Any]], dataset: str):
        # TODO: this probably doesnt make sense anymore since we're only getting 1 episode, 
        #   and batching 1 episode vertically just gives us 1 timestep every time
        # Getting the maxmimum length of episodes.
        text_obs = batch['text_observation']
        batch_size = len(text_obs)
        max_timesteps = 0
        for b in range(batch_size):
            max_timesteps = max(max_timesteps, len(text_obs[b]))
        
        for t in range(max_timesteps):
            cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts = [], [], [], [], [], [], []
            
            for b in range(batch_size):
                if t < len(text_obs[b]):
                    # This batch is consumed.
                    idxs.append(b)
                    cur_inputs.append([])

                    # First, setting the instructions and output types.
                    env_name = text_obs[b][t].strip().strip(string.punctuation).lower()
                    instruction = self._get_vlm_instruction(dataset, env_name)
                    instructions.append(instruction)

                    output_type = self._get_output_type(dataset, env_name)
                    output_types.append(output_type)

                    labels.append(batch['action'][b][t])
                    is_lasts.append(batch['is_last'][b][t])

                    if 'image_observation' in batch and batch['image_observation'][b][t] is not None:
                        image_obs = batch['image_observation'][b][t]
                        if len(image_obs.shape) == 4:
                            image_obs = [('image_observation', image) for image in image_obs]
                            cur_inputs[-1] += image_obs
                        else:
                            cur_inputs[-1].append(('image_observation', image_obs))

                    # Processing additional observations.
                    for key, value in batch.items():
                        if key not in ['action', 'reward', 'is_last', 'image_observation', 'text_observation'] and value[b][t] is not None:
                            cur_inputs[-1].append((key, value[b][t]))

                    cur_inputs[-1].append(('text_observation', text_obs[b][t]))
            yield cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts

    def _get_action_space(self, dataset, env_name):
        if dataset not in self.action_spaces:
            dataset = 'default'
        
        assert dataset in self.action_spaces, f"The dataset {dataset} is not included in the action spaces."
        
        if env_name in self.action_spaces[dataset]:
            # If env_name exists, the action space of that environment is defined specifically.
            action_space = self.action_spaces[dataset][env_name]
        else:
            # If not, the action space is the one shared by all environments.
            action_space = self.action_spaces[dataset]['default']
        return action_space
    
    # Generating the instruction text for VLMModule.
    def _get_vlm_instruction(self, dataset: str, env_name: str):
        assert dataset in self.descriptions, f"The dataset {dataset} is not included in the OpenX group."

        if env_name in self.descriptions[dataset]:
            # If env_name exists, the description of that environment is defined specifically.
            env_desc = ' '.join(self.descriptions[dataset][env_name])
        else:
            # If not, the env_name itself becomes the description.
            env_desc = env_name.capitalize() + "."

        action_space = self._get_action_space(dataset, env_name)
        
        # Handle the cases where the action space does not have a verbal description, and stats need to be used instead.
        if len(action_space) == 1:
            # If there is a placeholder 'None' in the action space, it means that the action space is not given a verbal description.
            if action_space[0] == None:
                action_space = {}
                for i in range(self.action_stats['size'][0]):
                    action_space[i] = ("The action space statistics of this dimension of the action space over the entire dataset", self.action_stats['min'][i], self.action_stats['max'][i], self.action_stats['mean'][i])
        
        elif len(action_space) != 1:
            # For cases where the verbal description is present but not the ranges, so we augment the information given with the stats
            for i in range(self.action_stats['size'][0]):
                if not isinstance (action_space[i], tuple):
                    action_space[i] = (action_space[i]+". In addition to this verbal description, here are the action space statistics of this dimension over the entire dataset", self.action_stats['min'][i], self.action_stats['max'][i], self.action_stats['mean'][i])
        
        if dataset not in self.action_exclusiveness:
            dataset = 'default'
            
        only_one_action = self.action_exclusiveness[dataset][env_name] if env_name in self.action_exclusiveness[dataset] else self.action_exclusiveness[dataset]['default']
        additional_inst = None
        if dataset in self.additional_instructions:
            if env_name in self.additional_instructions[dataset]:
                additional_inst = ' '.join(self.additional_instructions[dataset][env_name])
            else:
                additional_inst = self.additional_instructions[dataset]['default']

        instruction = self.format_instruction_prompt_fn(env_name, env_desc, action_space, only_one_action, additional_inst)
        return instruction
    
    # Getting the output type for VLMModule.
    def _get_output_type(self, dataset: str, env_name: str):
        if dataset not in self.action_exclusiveness:
            dataset = 'default'
            
        only_one_action = self.action_exclusiveness[dataset][env_name] if env_name in self.action_exclusiveness[dataset] else self.action_exclusiveness[dataset]['default']
        if only_one_action:
            return list
        else:
            return list

@dataclass
class BatchInfo:
    dataset_family: str
    dataset_name: str
    batch_num: int
    batch_id: str
    output_types: list[str]
    token_count: int
    is_lasts: list[int]
    labels: list
    num_inputs: int
    save_root: str
    
    def save_to_file(self) -> str:
        save_dir = f"{self.save_root}/batch_info/{self.dataset_family}/{self.dataset_name}_size_{self.num_inputs}"
        # Create folders if they dont exist        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        file_name = f"batch_{self.batch_num}.npz"
        run = 0
        while Path(f'{save_dir}/run_{run}/{file_name}').exists():
            run += 1
        Path(f'{save_dir}/run_{run}').mkdir()
                    
        np.savez(f'{save_dir}/run_{run}/{file_name}', 
                 dataset_name=self.dataset_name, batch_num=self.batch_num, batch_id=self.batch_id, 
                 output_types=self.output_types, token_count=self.token_count, is_lasts=self.is_lasts, 
                 labels=self.labels, num_inputs=self.num_inputs)
                
        return Path(f'{save_dir}/run_{run}/{file_name}').absolute()
 

class DatasetBatchModule(DatasetModule, ABC):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self.batch_size = batch_size
        self._batch_list = None
        
    @property
    def batch_list(self):
        if self._batch_list is None:
            self._batch_list = {ds : [] for ds in self.datasets}
        return self._batch_list
    
    @abstractmethod
    def _run_eval_dataset(self, dataset_batch_info_folder: str) -> dict:
        pass
    
    def _send_batch_job(self, batch, dataset_name, batch_num):
        cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts = self._process_batch(batch, dataset_name)
        num_inputs = len(cur_inputs)
        batch_id, token_count = self.modality_module.send_batch_job(cur_inputs, [], instructions)
        
        is_lasts = [int(is_last) for is_last in is_lasts]
        labels = [label.numpy() if not isinstance(label, np.ndarray) else label for label in labels]

        batch_job = BatchInfo(self.dataset_family, dataset_name, batch_num, batch_id, output_types, token_count, is_lasts, labels, num_inputs, self.disk_root_dir)
        fp = batch_job.save_to_file()
        self.batch_list[dataset_name].append(str(fp))
        
    def _send_batch_jobs_for_dataset(self, dataset):
        tfds_shards = self._find_shards(dataset)
        if len(tfds_shards) == 0:
            return {}

        # Creating the dataloader.
        dataloader_obj, dataloader = self.get_dataloader_fn(tfds_shards, batch_size=self.batch_size, by_episode=False)

        for i, batch in enumerate(dataloader):
            # Action stats need to be retrieved only once for each dataset, after they have been populated.
            if self.action_stats is None:
                self.action_stats = dataloader_obj.action_stats  
            self._send_batch_job(batch, dataset, i)
        return self.batch_list[dataset]
        
    def send_batch_jobs_for_all_datasets(self):
        for dataset in self.datasets:
            self._send_batch_jobs_for_dataset(dataset)
            self.action_stats = None
        return self.batch_list
        
    # Forming the batch.
    def _process_batch(self, batch: dict[str, list[Any]], dataset: str):
        # Getting the maxmimum length of episodes.
        text_obs = batch['text_observation']
        num_timesteps = len(text_obs)
        cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts = [], [], [], [], [], [], []
        for t in range(num_timesteps):
            # This batch is consumed.
            idxs.append(t)
            cur_inputs.append([])

            # First, setting the instructions and output types.
            env_name = text_obs[t].strip().strip(string.punctuation).lower()
            instruction = self._get_vlm_instruction(dataset, env_name)
            instructions.append(instruction)

            output_type = self._get_output_type(dataset, env_name)
            output_types.append(output_type)

            labels.append(batch['action'][t])
            is_lasts.append(batch['is_last'][t])

            if 'image_observation' in batch and batch['image_observation'][t] is not None:
                image_obs = batch['image_observation'][t]
                if len(image_obs.shape) == 4:
                    image_obs = [('image_observation', image) for image in image_obs]
                    cur_inputs[-1] += image_obs
                else:
                    cur_inputs[-1].append(('image_observation', image_obs))

            # Processing additional observations.
            for key, value in batch.items():
                if key not in ['action', 'reward', 'is_last', 'image_observation', 'text_observation'] and value[t] is not None:
                    cur_inputs[-1].append((key, value[t]))

            cur_inputs[-1].append(('text_observation', text_obs[t]))
            
        
        return cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts
    
    # Pass dict output from send_batch_jobs_for_all_datasets() for batch_info_dict
    def run_eval(self, results_path, batch_info_dict) -> None:
        if not batch_info_dict:
            batch_info_dict = self.batch_list
        
        total_results = {}
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                total_results = json.load(f)

        for dataset, batches in batch_info_dict.items():
            if dataset in total_results:
                warnings.warn(f'Skipping dataset: {dataset} (already evaluated)!' 
                                f'Delete the results from the results json for any dataset that should be overwritten.')
                continue
                
            result = self._run_eval_dataset(batches)
            total_results[dataset] = result

            # Write the updated or new results to the file
            with open(results_path, 'w') as f:
                json.dump(total_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)