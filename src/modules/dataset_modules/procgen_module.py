from src.modules.dataset_modules.base_dataset_module import DatasetModule, DatasetBatchModule
from definitions.procgen import ProcGenDefinitions
from src.data_utils.procgen_dataloader import get_procgen_dataloader
from definitions.procgen_prompt import format_instruction_prompt
from pathlib import Path
import numpy as np
import time
import warnings

class ProcgenModule(DatasetModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0) -> None:
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = ProcGenDefinitions
        self.get_dataloader_fn = get_procgen_dataloader
        self.dataset_family = 'procgen'
        self.format_instruction_prompt_fn = format_instruction_prompt
    
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
            total_preds = 0
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

        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result


class ProcgenBatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self._definitions_class = ProcGenDefinitions
        self.get_dataloader_fn = get_procgen_dataloader
        self.dataset_family = 'procgen'
        self.format_instruction_prompt_fn = format_instruction_prompt
        
    def _run_eval_dataset(self, dataset_batch_info_folder):
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