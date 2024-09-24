from src.modules.modality_modules.vlm_module import VLMModule
from src.data_utils.openx_dataloader import get_openx_dataloader
from definitions.prompt import format_instruction_prompt
from definitions.openx import DESCRIPTIONS, ACTION_SPACES, ACTION_EXCLUSIVENESS, ADDITIONAL_INSTRUCTIONS
from typing import Any, Union
from glob import glob

import numpy as np
import json
import string
import time


class OpenXModule:
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_size: int, k_shots: int) -> None:
        self.disk_root_dir = disk_root_dir
        self.datasets = list(DESCRIPTIONS.keys())
        self.modality = modality
        self.source = source
        self.model = model
        self.batch_size = batch_size
        self.k_shots = k_shots

    # Main evaluation function.
    def run_eval(self) -> None:
        # Since OpenX consists of multiple datasets, a modality module should be initialized per every evaluation step for each dataset.
        total_results = {}
        for dataset in self.datasets:
            if self.modality == 'vlm':
                modality_module = VLMModule(self.source, self.model)
                result = self._run_eval_dataset(dataset, modality_module)

                total_results[dataset] = result

        with open(f"openx_results.json", 'w') as f:
            json.dump(total_results, f)

    # Evaluation of one dataset.
    def _run_eval_dataset(self, dataset: str, modality_module: Any) -> dict[str, Union[list, float]]:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader = get_openx_dataloader(tfds_shards, batch_size=self.batch_size)

            avg_mse_list = []
            total_dataset_amse = 0.0
            episode_count = 0
            total_success_counts = []
            for batch in dataloader:
                batch_size = len(batch['text_observation'])
                episode_mses = [[] for b in range(batch_size)]
                success_counts = [0 for b in range(batch_size)]

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts in self._process_batch(batch, dataset):
                    outputs = modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)

                    # Any invalid output 'None' should be initialized into a random action.
                    outputs = [np.random.random(size=(labels[o].shape)) if output is None else output for o, output in enumerate(outputs)]

                    # This only works for continuous vector actions. (Okay for OpenX)
                    mses = np.mean((np.array(labels) - np.array(outputs)) ** 2, axis=-1)
                    assert len(mses) == len(idxs), "The calculated MSEs are not matched with the processed inputs."

                    for i, idx in enumerate(idxs):
                        episode_mses[idx].append(mses[i])

                        # If any episodes are the last, recording the success rate.
                        if is_lasts[i] and np.array_equal(np.array(outputs[i]), np.array(labels[i])):
                            success_counts[idx] += 1

                # Calculate average RMSE for the episode
                avg_episode_mses = [np.mean(episode_mse) for episode_mse in episode_mses]
                avg_mse_list += avg_episode_mses
                total_dataset_amse += np.sum(avg_episode_mses)
                episode_count += batch_size
                total_success_counts += success_counts

            end_time = time.time()
            eval_time = end_time - start_time

            result['action_success_rate'] = (sum(total_success_counts) / len(total_success_counts)) * 100
            result['avg_mse_list'] = avg_mse_list
            result['episode_count'] = episode_count
            result['total_dataset_amse'] = total_dataset_amse

            # Calculating average AMSE over all episodes
            avg_dataset_amse = total_dataset_amse / episode_count
            
            # Calculating min-max normalized AMSE
            min_amse = min(avg_mse_list)
            max_amse = max(avg_mse_list)
            result['normalized_amse'] = (avg_dataset_amse - min_amse) / (max_amse - min_amse) if max_amse != min_amse else 0
            result['eval_time'] = eval_time

        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result
    
    # Finding the translated TFDS shards.
    def _find_shards(self, dataset: str) -> list[str]:
        try:
            dataset_dir = glob(f"{self.disk_root_dir}/mount_dir*/openx_*_translated/{dataset}")[0]
            tfds_shards = glob(f"{dataset_dir}/translated_shard_*")
            return tfds_shards
        except IndexError:
            print(f"Cannot identify the directory to the dataset {dataset}. Skipping this dataset.")
            return []

    # Forming the batch.
    def _process_batch(self, batch: dict[str, list[Any]], dataset: str):
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

                    if 'continuous_observation' in batch and batch['continuous_observation'][b][t] is not None:
                        contiunuous_obs = batch['continuous_observation'][b][t]
                        cur_inputs[-1].append(('continuous_observation', contiunuous_obs))

                    if 'discrete_observation' in batch and batch['discrete_observation'][b][t] is not None:
                        discrete_obs = batch['discrete_observation'][b][t]
                        cur_inputs[-1].append(('discrete_observation', discrete_obs))

                    cur_inputs[-1].append(('text_observation', text_obs[b][t]))

            yield cur_inputs, k_shots_examples, instructions, labels, idxs, output_types, is_lasts

    # Generating the instruction text for VLMModule.
    def _get_vlm_instruction(self, dataset: str, env_name: str):
        assert dataset in DESCRIPTIONS, f"The dataset {dataset} is not included in the OpenX group."
        assert env_name in DESCRIPTIONS[dataset], f"The environment {env_name} is not included in the OpenX group."

        env_desc = ' '.join(DESCRIPTIONS[dataset][env_name])
        action_space = ACTION_SPACES[dataset][env_name]
        only_one_action = ACTION_EXCLUSIVENESS[dataset][env_name]
        additional_inst = None
        if dataset in ADDITIONAL_INSTRUCTIONS and env_name in ADDITIONAL_INSTRUCTIONS[dataset]:
            additional_inst = ' '.join(ADDITIONAL_INSTRUCTIONS[dataset][env_name])

        instruction = format_instruction_prompt(env_name, env_desc, action_space, only_one_action, additional_inst)
        return instruction
    
    # Getting the output type for VLMModule.
    def _get_output_type(self, dataset: str, env_name: str):
        if ACTION_EXCLUSIVENESS[dataset][env_name]:
            return tuple
        else:
            return list
