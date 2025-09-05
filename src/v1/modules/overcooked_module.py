from src.modules.dataset_modules.base_dataset_module import (
    DatasetModule,
    DatasetBatchModule,
    BatchInfo,
)
from definitions.overcooked import OverCookedDefinitions
from src.data_utils.overcooked_dataloader import get_overcooked_dataloader
from definitions.overcooked_prompt import format_instruction_prompt
from src.eval_utils import (
    quantile_filter,
    calculate_brier_mae,
    min_max_normalize,
    calculate_brier_mse,
    calculate_mean,
)
from src.eval_utils import (
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
)
from src.eval_utils import (
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    calculate_tp_fp_fn_counts,
)
from src.modules.modality_modules.vlm_module import VLMModule
from src.modules.source_modules.openai_module import OpenAIModule

from pathlib import Path
from typing import Union
from glob import glob

import numpy as np
import time
import string
from typing import Any


MAX_BRIER_MAE_ERROR = 2.0
MAX_BRIER_MSE_ERROR = 2.0
NOOP_ACTION = 28


def _validate_text_output(output, num_actions) -> bool:
    if not isinstance(output, list) or not all([isinstance(d, dict) for d in output]):
        return False

    keys, vals = set(), []
    for d in output:
        for k, v in d.items():
            try:
                k = float(k)
                v = float(v)
                k = int(np.round(k))
            except ValueError:
                return False

            # Check if the key is a digit and within the action space and if it is not a duplicate
            if not 0 <= k < num_actions or k in keys:
                return False
            keys.add(k)
            vals.append(v)

    # Check if the sum of the probabilities is 1, avoiding floating point errors
    if abs(sum(vals) - 1.0) > 1e-5:
        return False

    return True


def _validate_outputs_and_calculate_metrics(outputs, one_hot_labels, num_actions):
    brier_mses, brier_maes, preds = [], [], []
    total_invalid_preds = 0
    # Validate outputs and calculate Brier MSEs
    for o, output in enumerate(outputs):
        if _validate_text_output(output, num_actions):
            output = {int(k): float(v) for d in output for k, v in d.items()}
            probs = [0.0] * num_actions
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
    average_normalized_quantile_filtered_mae = calculate_mean(
        normalized_quantile_filtered_maes
    )

    max_rel_mae = calculate_max_relative_mae(timestep_maes)
    prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)

    # Calculate f1
    possible_actions = list(range(num_actions))
    tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
        preds, trues, possible_actions
    )
    precision = get_micro_precision_from_counts(tp, fp)
    precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    f1_without_invalid = get_micro_f1(precision_without_invalid, recall)

    # Calculate MSE metrics
    total_dataset_amse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0

    normalized_mses = min_max_normalize(timestep_mses)
    normalized_amse = calculate_mean(normalized_mses)

    result["total_dataset_amse"] = total_dataset_amse
    result["total_dataset_amae"] = sum(timestep_maes)
    result["num_timesteps"] = num_timesteps
    result["avg_dataset_amse"] = avg_dataset_amse
    result["avg_dataset_amae"] = average_dataset_mae
    result["normalized_amse"] = normalized_amse
    result["normalized_amae"] = average_normalized_mae
    result["normalized_quantile_filtered_amae"] = (
        average_normalized_quantile_filtered_mae
    )
    result["max_relative_mae"] = max_rel_mae
    result["proportion_beyond_threshold_mae"] = prop_beyond_threshold_mae
    result["recall"] = recall
    result["precision"] = precision
    result["precision_without_invalid"] = precision_without_invalid
    result["f1"] = f1
    result["f1_without_invalid"] = f1_without_invalid
    result["total_invalids"] = int(invalid_fp)
    result["percentage_invalids"] = (invalid_fp / len(preds)) * 100
    result["preds"] = [int(pred) for pred in preds]
    result["gt_actions"] = [int(true) for true in trues]
    return result


def _get_vlm_instruction(
    dataset: str,
    env_name: str,
    definitions_class,
    descriptions,
    action_exclusiveness,
    additional_instructions,
    format_instruction_prompt_fn,
    _get_action_space_fn,
):
    """Get VLM instruction for a given dataset and environment name"""
    assert (
        dataset in descriptions
    ), f"The layout {dataset} is not included in overcooked."

    if env_name in descriptions:
        env_desc = " ".join(descriptions[dataset][env_name])
    else:
        env_desc = env_name.capitalize() + "."

    action_space = _get_action_space_fn(dataset, env_name)

    if dataset not in action_exclusiveness:
        dataset = "default"

    additional_inst = None
    if dataset in additional_instructions:
        if env_name in additional_instructions[dataset]:
            additional_inst = " ".join(additional_instructions[dataset][env_name])
        else:
            additional_inst = None

    instruction = format_instruction_prompt_fn(
        env_name,
        env_desc,
        str(definitions_class.ACTION_MEANINGS),
        action_space,
        additional_inst,
    )
    return instruction


# Finding the pickle file
def _find_pickle_file(dataset, disk_root_dir: str) -> list[str]:
    try:
        # Construct the dataset directory path
        dataset_dir = f"{disk_root_dir}/{dataset}/test"
        print(dataset_dir)
        # Use glob to find .pickle file
        pickle_file = glob(f"{dataset_dir}/*.pickle")
        return pickle_file[0]
    except Exception as e:
        print(
            f"Cannot identify the directory to the dataset. Skipping this dataset. Error: {e}"
        )
        return []


class OvercookedModule(DatasetModule):
    def __init__(
        self,
        disk_root_dir: str,
        modality: str,
        source: str,
        model: str,
        dataset_name: str,
        batch_size: int = 1,
        k_shots: int = 0,
    ) -> None:
        DatasetModule.__init__(self, disk_root_dir, modality, source, model, batch_size, k_shots)
        
        self._definitions_class = OverCookedDefinitions
        self.get_dataloader_fn = get_overcooked_dataloader
        self.format_instruction_prompt_fn = format_instruction_prompt
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        
    @property
    def modality_module(self):
        self._modality_module = VLMModule(
            self.source,
            self.model,
            max_concurrent_prompts=400,
            max_output_tokens_per_query=512,
        )
        
        return self._modality_module

    # Finding the translated pickle files.
    def _find_shards(self) -> list[str]:
        return _find_pickle_file(self.dataset_name, self.disk_root_dir)
    
    

    def _run_eval_dataset(self, dataset: str) -> dict:
        result = {}

        try:
            oc_pickle = self._find_shards()
            if len(oc_pickle) == 0:
                return {}

            start_time = time.time()

            # Creating the dataloader.
            dataloader_obj, dataloader = self.get_dataloader_fn(
                oc_pickle,
                batch_size=self.batch_size,
            )
            
            result = {}

            timestep_mses, timestep_maes, timestep_preds, timestep_trues = (
                [],
                [],
                [],
                [],
            )
            total_invalid_preds = 0
            start_time = time.time()

            action_space = self._get_action_space(dataset, "default")
            num_actions = 0
            for action_idx, (_, action_dict) in action_space.items():
                num_actions += len(action_dict)

            for batch in dataloader:
                # Action stats need to be retrieved only once for each dataset, after they have been populated.
                if self.action_stats is None:
                    self.action_stats = dataloader_obj.action_stats

                # Consuming the batch until all timesteps in the batch.
                for (
                    cur_inputs,
                    k_shots_examples,
                    instructions,
                    labels,
                    idxs,
                    output_types,
                    is_lasts,
                ) in self._process_batch(batch, dataset):
                    outputs = self.modality_module.infer_step(
                        cur_inputs, k_shots_examples, instructions, output_types
                    )  # (B)

                    # Check if labels are within the action space, otherwise set to NoOp action
                    labels = np.array(
                        [
                            label if label < num_actions else NOOP_ACTION
                            for label in labels
                        ]
                    )
                    one_hot_labels = self._get_one_hot(labels, num_actions)

                    if not isinstance(outputs, list):
                        outputs = [None] * len(labels)

                    brier_mses, brier_maes, invalid_preds, preds = (
                        _validate_outputs_and_calculate_metrics(
                            outputs, one_hot_labels, num_actions
                        )
                    )
                    timestep_mses.extend(brier_mses)
                    timestep_maes.extend(brier_maes)
                    total_invalid_preds += invalid_preds
                    timestep_preds.extend(preds)
                    timestep_trues.extend(labels)

            result = _calculate_final_metrics(
                timestep_mses,
                timestep_maes,
                timestep_preds,
                timestep_trues,
                num_actions,
            )
            result["eval_time"] = time.time() - start_time
            result["total_invalid_preds"] = total_invalid_preds

        except KeyError as e:
            print(f"KeyError occurred: {e}")
            print(
                f"The VLMModule cannot be initialized since there is no dataset called {dataset}. Moving on to the next one..."
            )
            return {}

        return result

    def _process_batch(self, batch: dict[str, list[Any]], dataset: str):
        text_obs = batch["text_observation"]

        max_timesteps = max(0, len(text_obs))

        for t in range(max_timesteps):
            (
                cur_inputs,
                k_shots_examples,
                instructions,
                labels,
                idxs,
                output_types,
                is_lasts,
            ) = [], [], [], [], [], [], []


            if t < len(text_obs):
                # This batch is consumed.
                idxs.append(t)
                cur_inputs.append([])

                # First, setting the instructions and output types.
                env_name = text_obs[t].strip().strip(string.punctuation).lower()
                instruction = self._get_vlm_instruction("overcooked_ai", env_name)
                instructions.append(instruction)

                output_type = self._get_output_type("overcooked_ai", env_name)
                output_types.append(output_type)

                labels.append(batch["action"][t])
                is_lasts.append(batch["is_last"][t])

                if (
                    "image_observation" in batch
                    and batch["image_observation"][t] is not None
                ):
                    image_obs = batch["image_observation"][t]
                    if len(image_obs.shape) == 4:
                        image_obs = [
                            ("image_observation", image) for image in image_obs
                        ]
                        cur_inputs[-1] += image_obs
                    else:
                        cur_inputs[-1].append(("image_observation", image_obs))

                cur_inputs[-1].append(("text_observation", text_obs[t]))
            yield (
                cur_inputs,
                k_shots_examples,
                instructions,
                labels,
                idxs,
                output_types,
                is_lasts,
            )

    def _get_vlm_instruction(self, dataset: str, env_name: str):
        return _get_vlm_instruction(
            dataset,
            env_name,
            self._definitions_class,
            self.descriptions,
            self.action_exclusiveness,
            self.additional_instructions,
            self.format_instruction_prompt_fn,
            self._get_action_space,
        )


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
        self.disk_root_dir = disk_root_dir
        self.format_instruction_prompt_fn = format_instruction_prompt
        self.source = source
        self.batch_size = batch_size
        self.dataset_family = "overcooked_ai"
        self.dataset_name = "overcooked_ai"
        self._datasets = ["overcooked_ai"]

    @property
    def modality_module(self):
        self._modality_module = VLMModule(
            self.source,
            self.model,
            max_concurrent_prompts=400,
            max_output_tokens_per_query=512,
        )
        return self._modality_module

    def _find_shards(self, dataset:str) -> list[str]:
        return _find_pickle_file(dataset, self.disk_root_dir)

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
        oc_pickle = self._find_shards(dataset)
        if len(oc_pickle) == 0:
            return {}

        # Creating the dataloader, getting both the object and the iterable
        dataloader_obj, dataloader = self.get_dataloader_fn(
            oc_pickle, batch_size=self.batch_size, by_episode=False
        )

        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            print("Sending Batch Jobs....")
            self._send_batch_job(batch, dataset, i)
            print("Batch Jobs sent!")

        print(f"Finished sending jobs for {dataset}.")
        return self.batch_list[dataset]


    def _run_eval_dataset(self, dataset_batch_info_paths: Union[str, list[str]]):
        result = {}

        timestep_mses, timestep_maes, timestep_preds, timestep_trues = [], [], [], []
        total_invalid_preds = 0
        start_time = time.time()
        total_invalid_preds = 0

        # If it's a folder path, iterate over all files in the folder
        if isinstance(dataset_batch_info_paths, str):
            paths = Path(dataset_batch_info_paths).iterdir()
        elif isinstance(dataset_batch_info_paths, list):
            paths = dataset_batch_info_paths
        else:
            raise ValueError(
                f"data_batch_info_paths should be a path to a folder or a list of filepaths"
            )

        for fp in paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"Could not find file at path {fp}")

            batch_info = np.load(fp, allow_pickle=True)

            print(batch_info)

            output_types = list(batch_info["output_types"])
            ds = batch_info["dataset_name"].item()
            batch_num = batch_info["batch_num"].item()
            batch_id = batch_info["batch_id"].item()
            labels = batch_info["labels"]
            num_inputs = batch_info["num_inputs"].item()
            is_lasts = [bool(is_last) for is_last in batch_info["is_lasts"]]

            status = self.modality_module.get_batch_job_status(batch_id)
            if status == "completed":
                outputs = self.modality_module.retrieve_batch_results(
                    batch_id, output_types
                )
            else:
                raise Exception(
                    f"Batch not completed for batch {ds} batch num {batch_num} "
                    f"with batch id {batch_id}. Status: {status}. Stopping eval."
                )

            action_space = self._get_action_space(ds, "default")
            num_actions = 0
            for action_idx, (_, action_dict) in action_space.items():
                num_actions += 1
            print(action_space)
            print(f"NUMBER OF ACTIONS POSSIBLE: {num_actions}")
            print(labels)

            # Check if labels are within the action space, otherwise set to NoOp action
            labels = np.array(
                [label if label < num_actions else NOOP_ACTION for label in labels]
            )
            one_hot_labels = self._get_one_hot(labels, num_actions)

            if not isinstance(outputs, list):
                outputs = [None] * len(labels)

            brier_mses, brier_maes, invalid_preds, preds = (
                _validate_outputs_and_calculate_metrics(
                    outputs, one_hot_labels, num_actions
                )
            )
            timestep_mses.extend(brier_mses)
            timestep_maes.extend(brier_maes)
            total_invalid_preds += invalid_preds
            timestep_preds.extend(preds)
            timestep_trues.extend(labels)

        result = _calculate_final_metrics(
            timestep_mses, timestep_maes, timestep_preds, timestep_trues, num_actions
        )
        result["eval_time"] = time.time() - start_time
        result["total_invalid_preds"] = total_invalid_preds

        return result

    def _get_vlm_instruction(self, dataset: str, env_name: str):
        return _get_vlm_instruction(
            dataset,
            env_name,
            self._definitions_class,
            self.descriptions,
            self.action_exclusiveness,
            self.additional_instructions,
            self.format_instruction_prompt_fn,
            self._get_action_space,
        )

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
