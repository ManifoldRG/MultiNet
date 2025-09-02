from src.modules.dataset_modules.base_dataset_module import (
    DatasetBatchModule,
    BatchInfo,
)
from definitions.overcooked import OverCookedDefinitions
from src.data_utils.overcooked_dataloader import get_overcooked_dataloader
from definitions.overcooked_prompt import format_instruction_prompt

from pathlib import Path
from typing import Union
from glob import glob

import numpy as np
import time
import string
from typing import Any


def _validate_output(output: Any) -> bool:
    if output is None:
        return False
    out = int(output.strip())
    if isinstance(out, int) and len(out) > 0:
        return True
    return False


def _validate_outputs_and_calculate_metrics(outputs: list[int], labels: list[int]):
    """Validate outputs and calculate text similarity metrics for RoboVQA."""
    exact_matches = []
    total_invalid_preds = 0

    for i, output in enumerate(outputs):
        if _validate_output(output):
            label = labels[i]

            # Calculate exact match
            exact_match = 1.0 if output == label else 0.0
            exact_matches.append(exact_match)

        else:
            # Invalid output - assign worst possible scores
            exact_matches.append(0.0)
            total_invalid_preds += 1

    return exact_matches, total_invalid_preds


def _calculate_final_metrics(exact_matches):
    """Calculate comprehensive final metrics for RoboVQA evaluation."""
    result = {}

    # Calculate accuracy metrics
    total_samples = len(exact_matches)
    exact_match_accuracy = (
        sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    )

    result["exact_match_accuracy"] = exact_match_accuracy
    result["total_samples"] = total_samples

    return result


# Finding the translated TFDS shards.
def _find_shards(dataset, disk_root_dir: str) -> list[str]:
    try:
        # Construct the dataset directory path
        dataset_dir = Path(disk_root_dir) / "overcooked_ai" / "test"
        # Use glob to find all .pickle files
        shard_files = glob(str(dataset_dir / "*.pickle"))
        return shard_files[0]
    except Exception as e:
        print(
            f"Cannot identify the directory to the dataset. Skipping this dataset. Error: {e}"
        )
        return []


class OvercookedBatchModule(DatasetBatchModule):
    def __init__(
        self,
        disk_root_dir: str,
        modality: str,
        source: str,
        model: str,
        batch_info_dir: str,
        batch_size: int = 1,
        k_shots: int = 0,
    ):
        super().__init__(
            disk_root_dir, modality, source, model, batch_info_dir, batch_size, k_shots
        )
        self._definitions_class = OverCookedDefinitions
        self.get_dataloader_fn = get_overcooked_dataloader
        self.dataset_family = "overcooked_ai"
        self.disk_root_dir = "processed_datasets/"
        self._datasets = []
        self.format_instruction_prompt_fn = format_instruction_prompt

    @property
    def action_meanings(self):
        return self._definitions_class.ACTION_MEANINGS

    def _find_shards(self, dataset: str) -> list[str]:
        return _find_shards(dataset, self.disk_root_dir)

    def _send_batch_job(self, batch, dataset_name, batch_num):
        (
            cur_inputs,
            k_shots_examples,
            instructions,
            labels,
            idxs,
            output_types,
            is_lasts,
        ) = self._process_batch(batch, dataset_name)
        num_inputs = len(cur_inputs)
        batch_id, token_count = self.modality_module.send_batch_job(
            cur_inputs, [], instructions
        )
        labels = [np.array(label) for label in labels]
        is_lasts = [int(is_last) for is_last in is_lasts]
        labels = [
            label.numpy() if not isinstance(label, np.ndarray) else label
            for label in labels
        ]

        batch_job = BatchInfo(
            self.dataset_family,
            dataset_name,
            batch_num,
            batch_id,
            output_types,
            token_count,
            is_lasts,
            labels,
            num_inputs,
            self.batch_info_dir,
            self.model,
        )
        fp = batch_job.save_to_file()
        self.batch_list[dataset_name].append(str(fp))

    def _send_batch_jobs_for_dataset(self, dataset):
        tfds_shards = self._find_shards(dataset)
        if len(tfds_shards) == 0:
            return {}

        # Creating the dataloader, getting both the object and the iterable
        dataloader_obj, dataloader = self.get_dataloader_fn(
            tfds_shards, batch_size=self.batch_size, by_episode=False
        )

        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            print("Batch job sent")
            self._send_batch_job(batch, dataset, i)

        print(f"Finished sending jobs for {dataset}.")
        return self.batch_list[dataset]

    def send_batch_jobs_for_all_datasets(self):
        self._send_batch_jobs_for_dataset(self.dataset_family)
        self.action_stats = None
        print(self.batch_list)
        return self.batch_list

    def _run_eval_dataset(self, dataset_batch_info_paths: Union[str, list[str]]):
        result = {}

        exact_matches = []
        start_time = time.time()
        total_invalid_preds = 0

        # If it's a folder path, iterate over all files in the folder
        if isinstance(dataset_batch_info_paths, str):
            paths = Path(dataset_batch_info_paths).iterdir()
        elif isinstance(dataset_batch_info_paths, list):
            paths = dataset_batch_info_paths
        else:
            raise ValueError(
                "data_batch_info_paths should be a path to a folder or a list of filepaths"
            )

        for fp in paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"Could not find file at path {fp}")

            batch_info = np.load(fp, allow_pickle=True)

            # output_types = list(batch_info['output_types'])
            ds = batch_info["dataset_name"].item()
            batch_num = batch_info["batch_num"].item()
            batch_id = batch_info["batch_id"].item()
            labels = [int(label) for label in batch_info["labels"]]
            num_inputs = batch_info["num_inputs"].item()

            status = self.modality_module.get_batch_job_status(batch_id)
            if status == "completed":
                outputs = self.modality_module.retrieve_batch_results(batch_id)
            else:
                raise Exception(
                    f"Batch not completed for batch {ds} batch num {batch_num} "
                    f"with batch id {batch_id}. Status: {status}. Stopping eval."
                )

            matches, invalid_preds = _validate_outputs_and_calculate_metrics(
                outputs, labels
            )
            total_invalid_preds += invalid_preds

            assert (
                len(matches) == num_inputs
            ), "The length of calculated metrics list do not match with the length number of processed inputs."

            exact_matches.extend(matches)

        result = _calculate_final_metrics(exact_matches)
        result["eval_time"] = time.time() - start_time
        result["total_invalid_preds"] = total_invalid_preds

        return result

    def _get_vlm_instruction(self, dataset: str, env_name: str):
        assert (
            dataset in self.descriptions
        ), f"The layout {dataset} is not included in overcooked."

        if env_name in self.descriptions:
            env_desc = " ".join(self.descriptions[dataset][env_name])
        else:
            env_desc = env_name.capitalize() + "."

        action_space = self._get_action_space(dataset, env_name)

        if dataset not in self.action_exclusiveness:
            dataset = "default"

        additional_inst = None
        if dataset in self.additional_instructions:
            if env_name in self.additional_instructions[dataset]:
                additional_inst = " ".join(
                    self.additional_instructions[dataset][env_name]
                )
            else:
                additional_inst = None

        instruction = self.format_instruction_prompt_fn(
            env_name, env_desc, str(self.action_meanings), action_space, additional_inst
        )
        return instruction

    def _process_batch(self, batch: dict[str, list[Any]], dataset: str):
        # Getting the maxmimum length of episodes.
        text_obs = batch["text_observation"]
        num_timesteps = len(text_obs)
        (
            cur_inputs,
            k_shots_examples,
            instructions,
            labels,
            idxs,
            output_types,
            is_lasts,
        ) = [], [], [], [], [], [], []
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

            labels.append(batch["action"][t])
            is_lasts.append(batch["is_last"][t])

            if (
                "image_observation" in batch
                and batch["image_observation"][t] is not None
            ):
                image_obs = batch["image_observation"][t]
                if len(image_obs.shape) == 4:
                    image_obs = [("image_observation", image) for image in image_obs]
                    cur_inputs[-1] += image_obs
                else:
                    cur_inputs[-1].append(("image_observation", image_obs))

        return (
            cur_inputs,
            k_shots_examples,
            instructions,
            labels,
            idxs,
            output_types,
            is_lasts,
        )
