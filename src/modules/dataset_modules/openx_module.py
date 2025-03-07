from dataclasses import dataclass

import tensorflow as tf
from src.modules.modality_modules.vlm_module import VLMModule
from src.data_utils.openx_dataloader import get_openx_dataloader
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from definitions.prompt import format_instruction_prompt
from definitions.openx import OpenXDefinitions
from definitions.procgen import ProcGenDefinitions
from typing import Any, Union
from glob import glob
from pathlib import Path

import numpy as np
import json
import string
import time
import os
import warnings


class DatasetModule:
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, dataset_family: str, batch_size: int = None, k_shots: int = 0) -> None:
        self.disk_root_dir = disk_root_dir
        self.dataset_family = dataset_family
        
        if 'openx' == self.dataset_family:
            self.datalaoder_fn = get_openx_dataloader
            defs = OpenXDefinitions
        elif 'procgen' == self.dataset_family:
            self.dataloader_fn = get_procgen_dataloader
            defs = ProcGenDefinitions
        self.descriptions = defs.DESCRIPTIONS
        self.action_spaces = defs.ACTION_SPACES
        self.action_exclusiveness = defs.ACTION_EXCLUSIVENESS
        self.additional_instructions = defs.ADDITIONAL_INSTRUCTIONS    
        
        self.datasets = []
        for dataset in list(self.descriptions.keys()):
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) != 0:
                self.datasets.append(dataset)
                
        self.modality_module = None
        if modality == 'vlm':
            self.modality_module = VLMModule(source, model, max_concurrent_prompts=batch_size)
        assert self.modality_module is not None, "The modality module has not been set correcly. Check required."

        self.batch_size = batch_size
        self.k_shots = k_shots
        self.action_stats = None

    def _get_one_hot(self, targets, nb_classes):
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
    
    def _run_eval_dataset(self, dataset) -> dict:
        if self.dataset_family == 'openx':
            return self._run_eval_openx(dataset)
        elif self.dataset_family == 'procgen':
            return self._run_eval_procgen(dataset)
    
    def _run_eval_procgen(self, dataset: str) -> dict:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader_obj, dataloader = self.datalaoder_fn(tfds_shards, batch_size=self.batch_size, by_episode=True)

            action_success = []
            incorrect_preds = 0
            total_preds = 0
            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj._get_action_stats()  

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts in self._process_batch(batch, dataset):
                       
                    outputs = self.modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)
                    if self.dataset_family == 'procgen':
                        outputs = [[output[1]] for output in outputs]
                        
                    if isinstance(outputs[0][0], float)==False:
                        outputs = [[float(item) for item in sublist] for sublist in outputs]
                        
                    # Any invalid output 'None' should be initialized into a random action.
                    outputs = [self._validate_text_output(output, shape=labels[o].shape) for o, output in enumerate(outputs)]
                    # This only works for continuous vector actions. (Okay for OpenX)
                    
                    incorrect_preds += sum(np.array(labels) != np.array(outputs))
                    total_preds += len(labels)
                    for i, idx in enumerate(idxs):
                        # If any episodes are the last, recording the success rate.
                        if is_lasts[i]:
                            #print(outputs[i])
                            #print(labels[i])
                            if np.array_equal(np.array(outputs[i]), np.array(labels[i])):
                                #success_counts[idx] += 1
                                action_success.append(1)
                            else:
                                action_success.append(0)

            action_success_rate = (sum(action_success) / len(action_success)) * 100
            avg_dataset_01error = incorrect_preds / total_preds

            end_time = time.time()
            eval_time = end_time - start_time

            result['action_success_rate'] = action_success_rate
            result['total_dataset_01error'] = avg_dataset_01error
            result['eval_time'] = eval_time

        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result
        
    # Evaluation of one dataset.
    def _run_eval_openx(self, dataset: str) -> dict[str, Union[list, float]]:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader_obj, dataloader = self.datalaoder_fn(tfds_shards, batch_size=self.batch_size, by_episode=True)

            #avg_mse_list = []
            total_dataset_mse = 0.0
            #episode_count = 0
            action_success = []
            timestep_mse = []

            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj._get_action_stats()  
                #episode_mses = [[] for b in range(batch_size)]
                #success_counts = [0 for b in range(batch_size)]

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts in self._process_batch(batch, dataset):
                       
                    
                    outputs = self.modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)
                    
                    # Any invalid output 'None' should be initialized into a random action.
                    outputs = [self._validate_text_output(output, shape=labels[o].shape) for o, output in enumerate(outputs)]
                        
                    if isinstance(outputs[0][0], float)==False:
                        outputs = [[float(item) for item in sublist] for sublist in outputs]

                    # This only works for continuous vector actions. (Okay for OpenX)
                    mses = np.mean((np.array(labels) - np.array(outputs)) ** 2, axis=-1)
                    assert len(mses) == len(idxs), "The calculated MSEs are not matched with the processed inputs."

                    for i, idx in enumerate(idxs):
                        #episode_mses[idx].append(mses[i])
                        timestep_mse.append(mses[i])
                        # If any episodes are the last, recording the success rate.
                        if is_lasts[i]:
                            #print(outputs[i])
                            #print(labels[i])
                            if np.array_equal(np.array(outputs[i]), np.array(labels[i])):
                                #success_counts[idx] += 1
                                action_success.append(1)
                            else:
                                action_success.append(0)

                # Calculate average RMSE for the episode
                #avg_episode_mses = [np.mean(episode_mse) for episode_mse in episode_mses]
                #avg_mse_list += avg_episode_mses
                #total_dataset_amse += np.sum(avg_episode_mses)
                #episode_count += batch_size
                #total_success_counts += success_counts

            action_success_rate = (sum(action_success) / len(action_success)) * 100
            total_dataset_mse = sum(timestep_mse)
            print(f"\nTotal MSE across {len(timestep_mse)} timesteps: {total_dataset_mse:.4f}")
            num_timesteps = len(timestep_mse)
            avg_dataset_mse = total_dataset_mse / num_timesteps

            # Calculate average AMSE over all episodes
            #avg_dataset_amse = total_dataset_amse / episode_count
            
            # Calculate min-max normalized AMSE
            min_mse = min(timestep_mse)
            max_mse = max(timestep_mse)
            normalized_mse = (timestep_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else 0
            normalized_amse = sum(normalized_mse) / len(normalized_mse)

            end_time = time.time()
            eval_time = end_time - start_time

            '''result['action_success_rate'] = (sum(total_success_counts) / len(total_success_counts)) * 100
            result['avg_mse_list'] = avg_mse_list
            result['episode_count'] = episode_count
            result['total_dataset_amse'] = total_dataset_amse

            # Calculating average AMSE over all episodes
            avg_dataset_amse = total_dataset_amse / episode_count
            
            # Calculating min-max normalized AMSE
            min_amse = min(avg_mse_list)
            max_amse = max(avg_mse_list)
            result['normalized_amse'] = (avg_dataset_amse - min_amse) / (max_amse - min_amse) if max_amse != min_amse else 0
            result['eval_time'] = eval_time'''

            result['action_success_rate'] = action_success_rate
            result['total_dataset_amse'] = total_dataset_mse
            result['num_timesteps'] = num_timesteps
            result['avg_dataset_amse'] = avg_dataset_mse
            result['normalized_amse'] = normalized_amse
            result['eval_time'] = eval_time

        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result
    
    # Finding the translated TFDS shards.
    def _find_shards(self, dataset: str) -> list[str]:
        try:
            dataset_dir = glob(f"{self.disk_root_dir}/mount_dir*/{self.dataset_family}_*/{dataset}")[0]
            shard_files = glob(f"{dataset_dir}/translated_shard_*")
            tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
            return tfds_shards
        except IndexError:
            print(f"Cannot identify the directory to the dataset {dataset}. Skipping this dataset.")
            return []

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

        instruction = format_instruction_prompt(env_name, env_desc, action_space, only_one_action, additional_inst)
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
        
    # Validating the final output from the VLM/LLM model.
    def _validate_text_output(self, output: Any, shape: tuple[int]) -> np.array:
        if output is None or not isinstance(output, list) or len(output) != shape[0] or any(isinstance(x, (str, np.string_, set)) for x in output):
            output = np.random.random(size=shape)
        
        return np.array(output)

@dataclass
class BatchInfo:
    dataset_name: str
    dataset_family: str
    batch_num: int
    batch_id: str
    output_types: list[str]
    token_count: int
    is_lasts: list[int]
    labels: list
    num_inputs: int
    save_root: str
    
    def save_to_file(self) -> str:
        save_dir = f"{self.save_root}/batch_info/{self.dataset_name}_size_{self.num_inputs}"
        # Create folders if they dont exist        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        file_name = f"batch_{self.batch_num}.npz"
        run = 0
        while Path(f'{save_dir}/run_{run}/{file_name}').exists():
            run += 1
        Path(f'{save_dir}/run_{run}').mkdir()
                    
        np.savez(f'{save_dir}/run_{run}/{file_name}', 
                 dataset_name=self.dataset_name, dataset_family=self.dataset_family, batch_num=self.batch_num, batch_id=self.batch_id, 
                 output_types=self.output_types, token_count=self.token_count, is_lasts=self.is_lasts, 
                 labels=self.labels, num_inputs=self.num_inputs)
                
        return Path(f'{save_dir}/run_{run}/{file_name}').absolute()
 
class DatasetBatchModule(DatasetModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, dataset_family: str, batch_size: int = None, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, dataset_family, batch_size, k_shots)
        self.batch_list = {ds : [] for ds in self.datasets}
        
    def _send_batch_job(self, batch, dataset_name, batch_num):
        cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts = self._process_batch(batch, dataset_name)
        num_inputs = len(cur_inputs)
        batch_id, token_count = self.modality_module.send_batch_job(cur_inputs, [], instructions)
        
        is_lasts = [int(is_last) for is_last in is_lasts]
        labels = [label.numpy() for label in labels if not isinstance(label, np.ndarray)]

        batch_job = BatchInfo(dataset_name, self.dataset_family, batch_num, batch_id, output_types, token_count, is_lasts, labels, num_inputs, self.disk_root_dir)
        fp = batch_job.save_to_file()
        self.batch_list[dataset_name].append(fp)
        
    def _send_batch_jobs_for_dataset(self, dataset):
        tfds_shards = self._find_shards(dataset)
        if len(tfds_shards) == 0:
            return {}

        # Creating the dataloader.
        dataloader_obj, dataloader = self.datalaoder_fn(tfds_shards, batch_size=self.batch_size, by_episode=False)

        for i, batch in enumerate(dataloader):
            # Action stats need to be retrieved only once for each dataset, after they have been populated.
            if self.action_stats is None:
                self.action_stats = dataloader_obj._get_action_stats()  
            self._send_batch_job(batch, dataset, i)
        return self.batch_list[dataset]
        
    def send_batch_jobs_for_all_datasets(self):
        for dataset in self.datasets:
            self._send_batch_jobs_for_dataset(dataset)
            self.action_stats = None
        return self.batch_list
    
    def _run_eval_dataset(self, dataset_batch_info_folder) -> dict:
        if self.dataset_family == 'openx':
            return self._run_eval_openx(dataset_batch_info_folder)
        elif self.dataset_family == 'procgen':
            return self._run_eval_procgen(dataset_batch_info_folder)
        
    def _run_eval_procgen(self, dataset_batch_info_folder):
        result = {}
        
        timestep_mse = []
        total_preds = 0
        start_time = time.time()
        
        paths = Path(dataset_batch_info_folder).iterdir()
        for fp in paths:
            if Path(fp).exists():
                try:
                    batch_info = np.load(fp, allow_pickle=True)
                except Exception:
                    warnings.warn(f'Could not load file at path {fp}. Skipping...')
                    continue
            else:
                warnings.warn(f'Could not find file at path {fp}. Skipping...')
                continue    

                    
            output_types = list(batch_info['output_types'])
            ds = batch_info['dataset_name'].item()
            batch_num = batch_info['batch_num'].item()
            batch_id = batch_info['batch_id'].item()
            labels = batch_info['labels']
            num_inputs = batch_info['num_inputs'].item()
            is_lasts = [bool(is_last) for is_last in batch_info['is_lasts']]
            
            status = self.modality_module.get_batch_job_status(batch_id)
            if status == 'completed':
                outputs = self.modality_module.retrieve_batch_results(batch_id, output_types)
            else:
                warnings.warn(f'Batch not completed for batch {ds} batch num {batch_num} '
                              f'with batch id {batch_id}. Status: {status}. Skipping...')
                continue
            
            action_space = self._get_action_space(ds, 'default')
            num_actions  = 0
            for action_idx, (_, action_dict) in action_space.items():
                num_actions += len(action_dict)
                
            # validate outputs
            if not isinstance(outputs, list):
                outputs = []
            
            valid_idxs = []
            for i, output in enumerate(outputs):
                valid = True
                if isinstance(output, list) and all([isinstance(d, dict) for d in output]):
                    keys, vals = set(), []
                    for d in output:
                        for k, v in d.items():
                            if not str(k).isdigit() or not 0 <= int(k) < num_actions or k in keys:
                                valid = False
                                break
                            keys.add(int(k))
                            vals.append(float(v))
                        
                    if valid and sum(vals) >= 0.999:
                        valid_idxs.append(i)
            
            valid_probs = []
            for idx in valid_idxs:
                output = outputs[idx]
                output = {int(k): float(v) for d in output for k, v in d.items()}
                probs = [0.0]*num_actions
                for i in range(len(probs)):
                    if i in output:
                        probs[i] = output[i]
                valid_probs.append(probs)
            
            valid_labels = np.array([labels[i][0] for i in valid_idxs])
            valid_labels = self._get_one_hot(valid_labels, num_actions)
            
            brier_mses = np.mean((valid_probs - valid_labels)**2, axis=-1)
            timestep_mse.extend(brier_mses)
            total_preds += len(valid_idxs)
            
        total_dataset_mse = sum(timestep_mse)
        num_timesteps = len(timestep_mse)
        print(f"\nTotal MSE across {num_timesteps} timesteps: {total_dataset_mse:.4f}")
        
        avg_dataset_mse = total_dataset_mse / num_timesteps
        
        # Calculate min-max normalized AMSE
        min_mse = min(timestep_mse)
        max_mse = max(timestep_mse)
        normalized_mse = (timestep_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else 0
        normalized_amse = sum(normalized_mse) / len(normalized_mse)
        
        end_time = time.time()
        eval_time = end_time - start_time

        result['total_dataset_amse'] = total_dataset_mse
        result['num_timesteps'] = num_timesteps
        result['avg_dataset_amse'] = avg_dataset_mse
        result['normalized_amse'] = normalized_amse
        result['eval_time'] = eval_time
            
        return result
        
    def _run_eval_openx(self, dataset_batch_info_folder):
        result = {}
        
        #avg_mse_list = []
        total_dataset_mse = 0.0
        #episode_count = 0
        action_success = []
        timestep_mse = []
        start_time = time.time()
        
        paths = Path(dataset_batch_info_folder).iterdir()
        for fp in paths:
            if Path(fp).exists():
                try:
                    batch_info = np.load(fp, allow_pickle=True)
                except Exception:
                    warnings.warn(f'Could not load file at path {fp}. Skipping...')
                    continue
            else:
                warnings.warn(f'Could not find file at path {fp}. Skipping...')
                continue    

                    
            output_types = list(batch_info['output_types'])
            ds = batch_info['dataset_name'].item()
            batch_num = batch_info['batch_num'].item()
            batch_id = batch_info['batch_id'].item()
            labels = [tf.convert_to_tensor(label) for label in batch_info['labels']]
            num_inputs = batch_info['num_inputs'].item()
            is_lasts = [bool(is_last) for is_last in batch_info['is_lasts']]
            
            status = self.modality_module.get_batch_job_status(batch_id)
            if status == 'completed':
                outputs = self.modality_module.retrieve_batch_results(batch_id, output_types)
            else:
                warnings.warn(f'Batch not completed for batch {ds} batch num {batch_num} '
                              f'with batch id {batch_id}. Status: {status}. Skipping...')
                continue
            
            outputs = [self._validate_text_output(output, shape=labels[o].shape) for o, output in enumerate(outputs)]
                
            if isinstance(outputs[0][0], float)==False:
                outputs = [[float(item) for item in sublist] for sublist in outputs]

            # This only works for continuous vector actions. (Okay for OpenX)
            mses = np.mean((np.array(labels) - np.array(outputs)) ** 2, axis=-1)
            assert len(mses) == num_inputs, "The calculated MSEs are not matched with the processed inputs."

            for i in range(num_inputs):
                #episode_mses[idx].append(mses[i])
                timestep_mse.append(mses[i])
                # If any episodes are the last, recording the success rate.
                if is_lasts[i]:
                    #print(outputs[i])
                    #print(labels[i])
                    if np.array_equal(np.array(outputs[i]), np.array(labels[i])):
                        #success_counts[idx] += 1
                        action_success.append(1)
                    else:
                        action_success.append(0)
        action_success_rate = (sum(action_success) / len(action_success)) * 100
        total_dataset_mse = sum(timestep_mse)
        print(f"\nTotal MSE across {len(timestep_mse)} timesteps: {total_dataset_mse:.4f}")
        num_timesteps = len(timestep_mse)
        avg_dataset_mse = total_dataset_mse / num_timesteps

        # Calculate average AMSE over all episodes
        #avg_dataset_amse = total_dataset_amse / episode_count
        
        # Calculate min-max normalized AMSE
        min_mse = min(timestep_mse)
        max_mse = max(timestep_mse)
        normalized_mse = (timestep_mse - min_mse) / (max_mse - min_mse) if max_mse != min_mse else 0
        normalized_amse = sum(normalized_mse) / len(normalized_mse)

        end_time = time.time()
        eval_time = end_time - start_time

        '''result['action_success_rate'] = (sum(total_success_counts) / len(total_success_counts)) * 100
        result['avg_mse_list'] = avg_mse_list
        result['episode_count'] = episode_count
        result['total_dataset_amse'] = total_dataset_amse

        # Calculating average AMSE over all episodes
        avg_dataset_amse = total_dataset_amse / episode_count
        
        # Calculating min-max normalized AMSE
        min_amse = min(avg_mse_list)
        max_amse = max(avg_mse_list)
        result['normalized_amse'] = (avg_dataset_amse - min_amse) / (max_amse - min_amse) if max_amse != min_amse else 0
        result['eval_time'] = eval_time'''

        result['action_success_rate'] = action_success_rate
        result['total_dataset_amse'] = total_dataset_mse
        result['num_timesteps'] = num_timesteps
        result['avg_dataset_amse'] = avg_dataset_mse
        result['normalized_amse'] = normalized_amse
        result['eval_time'] = eval_time
        
        return result
    
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
    