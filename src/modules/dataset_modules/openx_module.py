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
                    print(v)

                batch_size = len(batch.values[0])
                episode_mses = [[] for b in batch_size]

                # Consuming the batch until all timesteps in the batch.
                for cur_inputs, k_shots_examples, instructions, labels, idxs in self._process_batch(batch):
                    outputs = modality_module.infer_step(cur_inputs, k_shots_examples, instructions)  # (B)
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
    def _process_batch(self, batch: dict[str, list]):
        cur_inputs, k_shots_examples, instructions, labels, idxs = [], [], [], [], []
        # TODO: After the dataloader is completed.

        yield cur_inputs, k_shots_examples, instructions, labels, idxs
        