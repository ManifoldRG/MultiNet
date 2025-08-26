from src.modules.dataset_modules.base_dataset_module import DatasetModule, DatasetBatchModule
from definitions.openx import OpenXDefinitions
from src.data_utils.openx_dataloader import get_openx_dataloader
from definitions.openx_prompt import format_instruction_prompt
from src.eval_utils import quantile_filter, calculate_mean, min_max_normalize, calculate_mse, calculate_mae, calculate_max_relative_mae, calculate_proportion_beyond_mae_threshold 
from pathlib import Path
from typing import Any, Union
import numpy as np
import time
import tensorflow as tf
from glob import glob

def _validate_text_output(output: Any, shape: tuple[int]) -> np.array:
    if output is None or not isinstance(output, list) or len(output) != shape[0] or any(isinstance(x, (str, np.string_, set)) for x in output):
        return False
    return True

def _validate_outputs_and_calculate_metrics(outputs, labels, action_stats):
    """Validate outputs and calculate MSE and MAE metrics for OpenX."""
    mses, maes = [], []
    total_invalid_preds = 0
    
    for i, output in enumerate(outputs):
        if _validate_text_output(output, shape=labels[i].shape):
            output = [float(item) for item in output]
            mses.append(calculate_mse(output, labels[i]))
            maes.append(calculate_mae(output, labels[i]))
        else:
            # max value of MSE/MAE for invalid outputs
            max_vals = np.array(action_stats['max'])
            min_vals = np.array(action_stats['min'])
            mse = calculate_mse(max_vals, min_vals)
            mae = calculate_mae(max_vals, min_vals)
            mses.append(mse)
            maes.append(mae)
            total_invalid_preds += 1
    
    return mses, maes, total_invalid_preds

def _calculate_final_metrics(timestep_mses, timestep_maes, action_success):
    """Calculate comprehensive final metrics for OpenX evaluation."""
    result = {}
    
    # Calculate MSE metrics
    total_dataset_mse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_mse = total_dataset_mse / num_timesteps if num_timesteps > 0 else 0.0
    
    # Calculate normalized MSE
    if num_timesteps > 1:
        normalized_mses = min_max_normalize(timestep_mses)
        normalized_amse = calculate_mean(normalized_mses)
    else:
        normalized_amse = 0.0
    
    # Calculate MAE metrics
    total_dataset_mae = sum(timestep_maes)
    avg_dataset_mae = calculate_mean(timestep_maes)
    
    if num_timesteps > 1:
        normalized_maes = min_max_normalize(timestep_maes)
        normalized_amae = calculate_mean(normalized_maes)
        
        # Calculate quantile filtered MAE metrics
        quantile_filtered_maes = quantile_filter(timestep_maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
        normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
        
        # Calculate additional MAE metrics
        max_rel_mae = calculate_max_relative_mae(timestep_maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
    else:
        normalized_amae = 0.0
        normalized_quantile_filtered_amae = 0.0
        max_rel_mae = 0.0
        prop_beyond_threshold_mae = 0.0
    
    # Calculate action success rate
    action_success_rate = None
    if len(action_success) > 0:
        action_success_rate = (sum(action_success) / len(action_success)) * 100
    
    result['action_success_rate'] = action_success_rate
    result['total_dataset_amse'] = total_dataset_mse
    result['total_dataset_amae'] = total_dataset_mae
    result['num_timesteps'] = num_timesteps
    result['avg_dataset_amse'] = avg_dataset_mse
    result['avg_dataset_amae'] = avg_dataset_mae
    result['normalized_amse'] = normalized_amse
    result['normalized_amae'] = normalized_amae
    result['normalized_quantile_filtered_amae'] = normalized_quantile_filtered_amae
    result['max_relative_mae'] = max_rel_mae
    result['proportion_beyond_threshold_mae'] = prop_beyond_threshold_mae
    
    return result

# Finding the translated TFDS shards.
def _find_shards(dataset: str, disk_root_dir: str) -> list[str]:
    try:
        dataset_dir = glob(f"/mnt/disks/mount_dir/MultiNet/src/v1/processed/{dataset}/test/")[0]
        shard_files = glob(f"{dataset_dir}/translated_shard_*")
        tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        return tfds_shards
    except IndexError:
        print(f"Cannot identify the directory to the dataset {dataset}. Skipping this dataset.")
        return []
    
class OpenXModule(DatasetModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0) -> None:
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = OpenXDefinitions
        self.get_dataloader_fn = get_openx_dataloader
        self.dataset_family = 'openx'
        self.format_instruction_prompt_fn = format_instruction_prompt  
    
    def _find_shards(self, dataset: str) -> list[str]:
        return _find_shards(dataset, self.disk_root_dir)
    
    # Evaluation of one dataset.
    def _run_eval_dataset(self, dataset: str) -> dict[str, Union[list, float]]:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader_obj, dataloader = self.get_dataloader_fn(tfds_shards, batch_size=self.batch_size, dataset_name=dataset, by_episode=True)

            #avg_mse_list = []
            total_dataset_mse = 0.0
            #episode_count = 0
            action_success = []
            timestep_mse = []
            total_invalid_preds = 0
            timestep_mae = []
            
            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj.action_stats  
                #episode_mses = [[] for b in range(batch_size)]
                #success_counts = [0 for b in range(batch_size)]

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts in self._process_batch(batch, dataset):
                    outputs = self.modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)
                    
                    mses, maes, invalid_preds = _validate_outputs_and_calculate_metrics(outputs, labels, self.action_stats)
                    total_invalid_preds += invalid_preds
                            
                    # This only works for continuous vector actions. (Okay for OpenX)
                    assert len(mses) == len(idxs), "The calculated MSEs are not matched with the processed inputs."

                    for i, idx in enumerate(idxs):
                        #episode_mses[idx].append(mses[i])
                        timestep_mse.append(mses[i])
                        timestep_mae.append(maes[i])
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

        

            result = _calculate_final_metrics(timestep_mse, timestep_mae, action_success)
            result['eval_time'] = time.time() - start_time
            result['total_invalid_preds'] = total_invalid_preds
            
        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result

class OpenXBatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_info_dir: str, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_info_dir, batch_size, k_shots)
        self._definitions_class = OpenXDefinitions
        self.get_dataloader_fn = get_openx_dataloader
        self.dataset_family = 'openx'
        self.format_instruction_prompt_fn = format_instruction_prompt
    
    def _find_shards(self, dataset: str) -> list[str]:
        return _find_shards(dataset, self.disk_root_dir)
    
    def _run_eval_dataset(self, dataset_batch_info_paths: Union[str, list[str]]):
        result = {}
        
        #avg_mse_list = []
        total_dataset_mse = 0.0
        #episode_count = 0
        action_success = []
        timestep_mse = []
        timestep_mae = []
        start_time = time.time()
        total_invalid_preds = 0
        
        # If it's a folder path, iterate over all files in the folder
        if isinstance(dataset_batch_info_paths, str):
            paths = Path(dataset_batch_info_paths).iterdir()
        elif isinstance(dataset_batch_info_paths, list):
            paths = dataset_batch_info_paths
        else:
            raise ValueError(f"data_batch_info_paths should be a path to a folder or a list of filepaths")
            
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
                                f'with batch id {batch_id}. Status: {status}. Stopping eval.')
            
            mses, maes, invalid_preds = _validate_outputs_and_calculate_metrics(outputs, labels, self.action_stats)
            total_invalid_preds += invalid_preds
                    
            # This only works for continuous vector actions. (Okay for OpenX)
            assert len(mses) == num_inputs, "The calculated MSEs are not matched with the processed inputs."

            for i in range(num_inputs):
                #episode_mses[idx].append(mses[i])
                timestep_mse.append(mses[i])
                timestep_mae.append(maes[i])
                # If any episodes are the last, recording the success rate.
                if is_lasts[i]:
                    #print(outputs[i])
                    #print(labels[i])
                    if np.array_equal(np.array(outputs[i]), np.array(labels[i])):
                        #success_counts[idx] += 1
                        action_success.append(1)
                    else:
                        action_success.append(0)
        

        result = _calculate_final_metrics(timestep_mse, timestep_mae, action_success)
        result['eval_time'] = time.time() - start_time
        result['total_invalid_preds'] = total_invalid_preds
        
        return result