from src.modules.modality_modules.vlm_module import VLMModule
from src.data_utils.openx_dataloader import get_openx_dataloader
from definitions.prompt import format_instruction_prompt
from definitions.openx import DESCRIPTIONS, ACTION_SPACES, ACTION_EXCLUSIVENESS, ADDITIONAL_INSTRUCTIONS
from typing import Any, Union
from glob import glob

import numpy as np


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
        for dataset in self.datasets:
            if self.modality == 'vlm':
                modality_module = VLMModule(self.source, self.model)
                result = self._run_eval_dataset(dataset, modality_module)

    # Evaluation of one dataset.
    def _run_eval_dataset(self, dataset: str, modality_module: Any) -> dict[str, Union[list, float]]:
        result = {}

        try:
            tfds_shards = self._find_shards(dataset)
            if len(tfds_shards) == 0:
                return {}

            # Creating the dataloader.
            dataloader = get_openx_dataloader(tfds_shards, batch_size=self.batch_size, num_workers=4)

            avg_mse_list = []
            total_dataset_amse = 0.0
            episode_count = 0
            for batch in dataloader:
                for k, v in batch.items():
                    print("#" * 100)
                    print(k)
                    print([len(vv) for vv in v])

                    if k == 'image_observation':
                        for vv in v:
                            for vvv in vv:
                                print(vvv.shape)
                exit()

                batch_size = len(batch['text_observation'])
                episode_mses = [[] for b in batch_size]

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs, output_types in self._process_batch(batch, dataset):
                    outputs = modality_module.infer_step(cur_inputs, k_shots_examples, instructions, output_types)  # (B)

                    # TODO: This does not work for only one action value.
                    mses = np.mean((np.array(labels) - np.array(outputs)) ** 2, axis=-1)
                    assert len(mses) == len(idxs), "The calculated MSEs are not matched with the processed inputs."

                    for i, idx in enumerate(idxs):
                        episode_mses[idx].append(mses[i])

                # Calculate average RMSE for the episode
                avg_episode_mses = [np.mean(episode_mse) for episode_mse in episode_mses]
                avg_mse_list += avg_episode_mses
                total_dataset_amse += np.sum(avg_episode_mses)
                episode_count += batch_size
        
        except KeyError:
            print(f"The VLMModule cannot be initialized since there is no dataset called {dataset} in OpenX. Moving on to the next one...")
            return {}

        return result
    
    # Finding the translated TFDS shards.
    def _find_shards(self, dataset: str) -> list[str]:
        try:
            dataset_dir = glob(f"{self.disk_root_dir}/mount_dir*/openx_*_translated/{dataset}")[0]
            tfds_shards = glob(f"{dataset_dir}/translated_shard_*")[:1]  # TODO: DEBUG.
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
        batch.pop('is_last')
        
        for t in range(max_timesteps):
            cur_inputs, k_shots_examples, instructions, labels, idxs, output_types = [], [], [], [], [], []
            
            for b in range(batch_size):
                if t < len(text_obs[b]):
                    # This batch is consumed.
                    idxs.append(b)
                    cur_inputs.append([])

                    # First, setting the instructions and output types.
                    env_name = text_obs[b][t]
                    instruction = self._get_vlm_instruction(dataset, env_name)
                    instructions.append(instruction)

                    output_type = self._get_output_type(dataset, env_name)
                    output_types.append(output_type)

                    labels.append(batch['action'][b][t])

                    if 'image_observation' in batch and batch['image_observation'] is not None:
                        image_obs = batch['image_observation'][b][t]
                        if len(image_obs.shape) == 4:
                            image_obs = [('image_observation', image) for image in image_obs]
                            cur_inputs[-1] += image_obs
                        else:
                            cur_inputs[-1].append(('image_observation', image_obs))

                    if 'continuous_observation' in batch and batch['continuous_observation'] is not None:
                        contiunuous_obs = batch['continuous_observation'][b][t]
                        cur_inputs[-1].append(('continuous_observation', contiunuous_obs))

                    if 'discrete_observation' in batch and batch['discrete_observation'] is not None:
                        discrete_obs = batch['discrete_observation'][b][t]
                        cur_inputs[-1].append(('discrete_observation', discrete_obs))

            yield cur_inputs, k_shots_examples, instructions, labels, idxs, output_types

    # Generating the instruction text for VLMModule.
    def _get_vlm_instruction(self, dataset: str, env_name: str):
        instruction = format_instruction_prompt(
            DESCRIPTIONS,
            ACTION_SPACES,
            ACTION_EXCLUSIVENESS,
            dataset,
            env_name,
            additional_inst=ADDITIONAL_INSTRUCTIONS[dataset] if dataset in ADDITIONAL_INSTRUCTIONS else None
        )
        return instruction
    
    # Getting the output type for VLMModule.
    def _get_output_type(self, dataset: str, env_name: str):
        if ACTION_EXCLUSIVENESS[dataset][env_name]:
            return tuple
        else:
            return list
