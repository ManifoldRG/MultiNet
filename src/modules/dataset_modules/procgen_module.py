from src.modules.dataset_modules.base_dataset_module import DatasetModule, DatasetBatchModule
from definitions.procgen import ProcGenDefinitions
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from definitions.procgen_prompt import format_instruction_prompt
from pathlib import Path
import numpy as np
import time
from glob import glob

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
        
            timestep_mse = []
            total_invalid_preds = 0
            start_time = time.time()
            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj.action_stats

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts in self._process_batch(batch, dataset):
                       
                    outputs = self.modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)
                    
                    action_space = self._get_action_space(dataset, 'default')
                    num_actions  = 0
                    for action_idx, (_, action_dict) in action_space.items():
                        num_actions += len(action_dict)
                    labels = np.array([label[0] for label in labels])
                    one_hot_labels = self._get_one_hot(labels, num_actions)
                                       
                    if not isinstance(outputs, list):
                        outputs = [None]*len(labels)
                        
                    brier_mses = []                    
                    # Validate outputs and calculate Brier MSEs
                    for o, output in enumerate(outputs):
                        if _validate_text_output(output, num_actions):
                            output = {int(k): float(v) for d in output for k, v in d.items()}
                            probs = [0.0]*num_actions
                            for i in range(len(probs)):
                                if i in output:
                                    probs[i] = output[i]

                            # TODO: placeholder metric
                            brier_mses.append(np.sum((np.array(probs) - one_hot_labels[o])**2))
                        else:
                            # max possible Brier MSE is 2.0
                            brier_mses.append(2.0)
                            total_invalid_preds += 1
                    timestep_mse.extend(brier_mses)

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
        
    def _run_eval_dataset(self, dataset_batch_info_folder):
        result = {}
        
        timestep_mse = []
        total_invalid_preds = 0
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
            labels = batch_info['labels']
            num_inputs = batch_info['num_inputs'].item()
            is_lasts = [bool(is_last) for is_last in batch_info['is_lasts']]
            
            status = self.modality_module.get_batch_job_status(batch_id)
            if status == 'completed':
                outputs = self.modality_module.retrieve_batch_results(batch_id, output_types)
            else:
                raise Exception(f'Batch not completed for batch {ds} batch num {batch_num} '
                                f'with batch id {batch_id}. Status: {status}. Skipping...')
            
            action_space = self._get_action_space(ds, 'default')
            num_actions  = 0
            for action_idx, (_, action_dict) in action_space.items():
                num_actions += len(action_dict)
            labels = np.array([label[0] for label in labels])
            one_hot_labels = self._get_one_hot(labels, num_actions)
            
            if not isinstance(outputs, list):
                outputs = [None]*len(labels)
                
            brier_mses = []                    
            # Validate outputs and calculate Brier MSEs
            for o, output in enumerate(outputs):
                if _validate_text_output(output, num_actions):
                    output = {int(k): float(v) for d in output for k, v in d.items()}
                    probs = [0.0]*num_actions
                    for i in range(len(probs)):
                        if i in output:
                            probs[i] = output[i]

                    # TODO: placeholder metric
                    brier_mses.append(np.sum((np.array(probs) - one_hot_labels[o])**2))
                else:
                    # max possible Brier MSE is 2.0
                    brier_mses.append(2.0)
                    total_invalid_preds += 1
            timestep_mse.extend(brier_mses)
            
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
        result['total_invalid_preds'] = total_invalid_preds
            
        return result