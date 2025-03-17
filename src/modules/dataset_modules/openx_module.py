from src.modules.dataset_modules.base_dataset_module import DatasetModule, DatasetBatchModule
from definitions.openx import OpenXDefinitions
from src.data_utils.openx_dataloader import get_openx_dataloader
from definitions.openx_prompt import format_instruction_prompt
from pathlib import Path
from typing import Any, Union
import numpy as np
import time
import tensorflow as tf
from glob import glob

class OpenXModule(DatasetModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0) -> None:
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = OpenXDefinitions
        self._dataloader_fn = get_openx_dataloader
        self.dataset_family = 'openx'
        self.format_instruction_prompt_fn = format_instruction_prompt
        
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
        
    def _validate_text_output(self, output: Any, shape: tuple[int]) -> np.array:
        if output is None or not isinstance(output, list) or len(output) != shape[0] or any(isinstance(x, (str, np.string_, set)) for x in output):
            output = np.random.random(size=shape)
        
        return np.array(output)

    # Evaluation of one dataset.
    def _run_eval_dataset(self, dataset: str) -> dict[str, Union[list, float]]:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader_obj, dataloader = self.dataloader_fn(tfds_shards, batch_size=self.batch_size, by_episode=True)

            #avg_mse_list = []
            total_dataset_mse = 0.0
            #episode_count = 0
            action_success = []
            timestep_mse = []

            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj.action_stats  
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

class OpenXBatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = OpenXDefinitions
        self._dataloader_fn = get_openx_dataloader
        self.dataset_family = 'openx'
        self.format_instruction_prompt_fn = format_instruction_prompt

    def _validate_text_output(self, output: Any, shape: tuple[int]) -> np.array:
        if output is None or not isinstance(output, list) or len(output) != shape[0] or any(isinstance(x, (str, np.string_, set)) for x in output):
            output = np.random.random(size=shape)
        
        return np.array(output)
    
    def _run_eval_dataset(self, dataset_batch_info_folder):
        result = {}
        
        #avg_mse_list = []
        total_dataset_mse = 0.0
        #episode_count = 0
        action_success = []
        timestep_mse = []
        start_time = time.time()
        
        paths = Path(dataset_batch_info_folder).iterdir()
        for fp in paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f'Could not find file at path {fp}') 
            
            batch_info = np.load(fp, allow_pickle=True)    

                    
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
                raise Exception(f'Batch not completed for batch {ds} batch num {batch_num} '
                                f'with batch id {batch_id}. Status: {status}. Skipping...')
            
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
        action_success_rate = None
        if len(action_success) > 0:
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