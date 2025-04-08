from src.modules.dataset_modules.base_dataset_module import DatasetModule, DatasetBatchModule
from definitions.procgen import ProcGenDefinitions
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from src.eval_utils import quantile_filter, calculate_brier_mae, min_max_normalize, calculate_brier_mse, calculate_mean
from src.eval_utils import calculate_max_relative_mae, calculate_proportion_beyond_mae_threshold
from src.eval_utils import get_micro_precision_from_counts, get_micro_recall_from_counts, get_micro_f1, calculate_tp_fp_fn_counts
from definitions.procgen_prompt import format_instruction_prompt

from pathlib import Path
from typing import Union
import numpy as np
import time
from glob import glob

MAX_BRIER_MAE_ERROR = 2.0
MAX_BRIER_MSE_ERROR = 2.0
NOOP_ACTION = 4

def _validate_text_output(output, num_actions) -> bool:
    if not isinstance(output, list) or not all([isinstance(d, dict) for d in output]):
        return False
    
    keys, vals = set(), []
    for d in output:
        for k, v in d.items():
            # Check if the key is a digit and within the action space and if it is not a duplicate
            if not str(k).isdigit() or not 0 <= int(k) < num_actions or k in keys:
                return False
            keys.add(int(k))
            vals.append(float(v))
    
    # Check if the sum of the probabilities is 1, avoiding floating point errors
    if sum(vals) < 0.999:
        return False
    
    return True
    
# Finding the translated TFDS shards.
def _find_shards(dataset: str, disk_root_dir: str) -> list[str]:
    try:
        #FIXME: this needs to change when doing final evals depending on the files' naming scheme
        dataset_dir = glob(f"{disk_root_dir}/mount_dir*/procgen_*/{dataset}")[0]
        shard_files = glob(f"{dataset_dir}/translated_shard_*")
        tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        return tfds_shards
    except IndexError:
        print(f"Cannot identify the directory to the dataset {dataset}. Skipping this dataset.")
        return []

def _validate_outputs_and_calculate_metrics(outputs, one_hot_labels, num_actions):
    brier_mses, brier_maes, preds = [], [], []
    total_invalid_preds = 0               
    # Validate outputs and calculate Brier MSEs
    for o, output in enumerate(outputs):
        if _validate_text_output(output, num_actions):
            output = {int(k): float(v) for d in output for k, v in d.items()}
            probs = [0.0]*num_actions
            for i in range(len(probs)):
                if i in output:
                    probs[i] = output[i]
                    
            mae = calculate_brier_mae(probs, one_hot_labels[o])
            brier_maes.append(mae)
            
            mse = calculate_brier_mse(probs, one_hot_labels[o])
            brier_mses.append(mse)
            
            preds.append(np.argmax(probs))
        else:
            # max possible Brier MSE is 2.0
            brier_maes.append(MAX_BRIER_MAE_ERROR)
            brier_mses.append(MAX_BRIER_MSE_ERROR)
            
            total_invalid_preds += 1
            
            preds.append(-1)
    return brier_mses, brier_maes, total_invalid_preds, preds

def _calculate_final_metrics(timestep_mses, timestep_maes, preds, trues, num_actions):
    result = {}
    
    # Calculate MAE metrics
    average_dataset_mae = calculate_mean(timestep_maes)
    normalized_maes = min_max_normalize(timestep_maes)
    average_normalized_mae = calculate_mean(normalized_maes)

    # Calculate quantile filtered MAE metrics
    quantile_filtered_maes = quantile_filter(timestep_maes)
    normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
    average_normalized_quantile_filtered_mae = calculate_mean(normalized_quantile_filtered_maes)
    
    max_rel_mae = calculate_max_relative_mae(timestep_maes)
    prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
    
    # Calculate f1   
    possible_actions = list(range(num_actions))
    tp, fp, fn = calculate_tp_fp_fn_counts(preds, trues, possible_actions)
    precision = get_micro_precision_from_counts(tp, fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    
    # Calculate MSE metrics
    total_dataset_amse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0

    normalized_mses = min_max_normalize(timestep_mses)
    normalized_amse = calculate_mean(normalized_mses)
    
    result['total_dataset_amse'] = total_dataset_amse
    result['total_dataset_amae'] = sum(timestep_maes)
    result['num_timesteps'] = num_timesteps
    result['avg_dataset_amse'] = avg_dataset_amse
    result['avg_dataset_amae'] = average_dataset_mae
    result['normalized_amse'] = normalized_amse
    result['normalized_amae'] = average_normalized_mae
    result['normalized_quantile_filtered_amae'] = average_normalized_quantile_filtered_mae
    result['max_relative_mae'] = max_rel_mae
    result['proportion_beyond_threshold_mae'] = prop_beyond_threshold_mae
    result['recall'] = recall
    result['precision'] = precision
    result['f1'] = f1
    return result
            
        
class ProcGenModule(DatasetModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0) -> None:
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = ProcGenDefinitions
        self.get_dataloader_fn = get_procgen_dataloader
        self.dataset_family = 'procgen'
        self.format_instruction_prompt_fn = format_instruction_prompt
        
    # Finding the translated TFDS shards.
    def _find_shards(self, dataset: str) -> list[str]:
        return _find_shards(dataset, self.disk_root_dir)
            
    def _run_eval_dataset(self, dataset: str) -> dict:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader_obj, dataloader = self.get_dataloader_fn(tfds_shards, batch_size=self.batch_size, by_episode=True)
            result = {}
        
            timestep_mses, timestep_maes, timestep_preds, timestep_trues = [], [], [], []
            total_invalid_preds = 0
            start_time = time.time()
            
            action_space = self._get_action_space(dataset, 'default')
            num_actions  = 0
            for action_idx, (_, action_dict) in action_space.items():
                num_actions += len(action_dict)
                
            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj.action_stats

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts in self._process_batch(batch, dataset):
                       
                    outputs = self.modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)

                    # Check if labels are within the action space, otherwise set to NoOp action
                    labels = np.array([label[0] if label[0] < num_actions else NOOP_ACTION for label in labels])
                    one_hot_labels = self._get_one_hot(labels, num_actions)
                                       
                    if not isinstance(outputs, list):
                        outputs = [None]*len(labels)
                        
                    brier_mses, brier_maes, invalid_preds, preds = _validate_outputs_and_calculate_metrics(outputs, one_hot_labels, num_actions)
                    timestep_mses.extend(brier_mses)
                    timestep_maes.extend(brier_maes)
                    total_invalid_preds += invalid_preds 
                    timestep_preds.extend(preds)
                    timestep_trues.extend(labels)
                    
            result = _calculate_final_metrics(timestep_mses, timestep_maes, timestep_preds, timestep_trues, num_actions)
            result['eval_time'] = time.time() - start_time
            result['total_invalid_preds'] = total_invalid_preds

        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result


class ProcGenBatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = ProcGenDefinitions
        self.get_dataloader_fn = get_procgen_dataloader
        self.dataset_family = 'procgen'
        self.format_instruction_prompt_fn = format_instruction_prompt
        
    # Finding the translated TFDS shards.
    def _find_shards(self, dataset: str) -> list[str]:
        return _find_shards(dataset, self.disk_root_dir)
        
    def _run_eval_dataset(self, dataset_batch_info_paths: Union[str, list[str]]):
        result = {}
        
        timestep_mses, timestep_maes, timestep_preds, timestep_trues = [], [], [], []
        total_invalid_preds = 0
        start_time = time.time()
        
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
            labels = batch_info['labels']
            num_inputs = batch_info['num_inputs'].item()
            is_lasts = [bool(is_last) for is_last in batch_info['is_lasts']]
            
            status = self.modality_module.get_batch_job_status(batch_id)
            if status == 'completed':
                outputs = self.modality_module.retrieve_batch_results(batch_id, output_types)
            else:
                raise Exception(f'Batch not completed for batch {ds} batch num {batch_num} '
                                f'with batch id {batch_id}. Status: {status}. Stopping eval.')
            
            action_space = self._get_action_space(ds, 'default')
            num_actions  = 0
            for action_idx, (_, action_dict) in action_space.items():
                num_actions += len(action_dict)
            
            # Check if labels are within the action space, otherwise set to NoOp action
            labels = np.array([label[0] if label[0] < num_actions else NOOP_ACTION for label in labels])
            one_hot_labels = self._get_one_hot(labels, num_actions)
            
            if not isinstance(outputs, list):
                outputs = [None]*len(labels)
                
            brier_mses, brier_maes, invalid_preds, preds = _validate_outputs_and_calculate_metrics(outputs, one_hot_labels, num_actions)
            timestep_mses.extend(brier_mses)
            timestep_maes.extend(brier_maes)
            total_invalid_preds += invalid_preds
            timestep_preds.extend(preds)
            timestep_trues.extend(labels)
            
        result = _calculate_final_metrics(timestep_mses, timestep_maes, timestep_preds, timestep_trues, num_actions)
        result['eval_time'] = time.time() - start_time
        result['total_invalid_preds'] = total_invalid_preds
            
        return result