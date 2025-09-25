from src.modules.dataset_modules.base_dataset_module import DatasetBatchModule, BatchInfo
from src.data_utils.odinw_dataloader import get_odinw_dataloader
from src.modules.source_modules.openai_module import OpenAIModule
from definitions.odinw import ODinWDefinitions
from src.eval_utils import (
    get_exact_match_rate
)
from src.eval_utils import (
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    calculate_tp_fp_fn_counts,
    get_precision_per_class,
    get_recall_per_class,
    get_f1_per_class,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)
from pathlib import Path
import numpy as np
import time
import os
from typing import Union

def _validate_output(output, possible_outputs) -> bool:
    """Validate that output is a valid integer within the possible outputs"""
    # Handle string outputs that might contain the choice
    if isinstance(output, str):
        try:
            return int(output.strip()) in possible_outputs
        except Exception:
            return False

    return False


def _validate_outputs_and_calculate_metrics(outputs, possible_outputs):
    """Validate outputs and convert to predictions for exact match calculation"""
    preds = []
    total_invalid_preds = 0
    
    for output in outputs:
        if _validate_output(output, possible_outputs):
            preds.append(int(output.strip()))
        else:
            total_invalid_preds += 1
            preds.append(-1)
    
    return total_invalid_preds, preds


def _calculate_final_metrics(preds, labels, possible_outputs):
    """Validate outputs and convert to predictions for exact match calculation"""
    result = {}
    
    valid_preds = []
    valid_trues = []
    invalid_count = 0
    
    for pred, true in zip(preds, labels):
        if pred == -1:  # Invalid prediction
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)
    
    preds = np.array([int(pred) for pred in preds])
    labels = np.array([int(true) for true in labels])
    if len(valid_preds) > 0:
        exact_match_rate_without_invalids = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate_without_invalids = 0.0
    
    exact_match_rate_with_invalids = get_exact_match_rate(preds, labels)
    
     # Calculate metrics counts
    tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
        preds, labels, possible_outputs
    )
    
    precision = get_micro_precision_from_counts(tp, fp)
    precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    f1_without_invalid = get_micro_f1(precision_without_invalid, recall)
    
    # Calculate class-wise metrics
    class_precisions = get_precision_per_class(preds, labels, possible_outputs)
    class_recalls = get_recall_per_class(preds, labels, possible_outputs)
    class_f1s = get_f1_per_class(class_precisions, class_recalls)
    
    # Calculate macro metrics
    macro_precision = get_macro_precision(class_precisions)
    macro_recall = get_macro_recall(class_recalls)
    macro_f1 = get_macro_f1(class_f1s)
    
    
    result["exact_match_rate"] = exact_match_rate_with_invalids
    result["exact_match_rate_without_invalids"] = exact_match_rate_without_invalids
    result["recall"] = recall
    result["precision"] = precision
    result["precision_without_invalids"] = precision_without_invalid
    result["f1"] = f1
    result["f1_without_invalids"] = f1_without_invalid
    result["macro_precision"] = macro_precision
    result["macro_recall"] = macro_recall
    result["macro_f1"] = macro_f1
    result["class_precisions"] = class_precisions
    result["class_recalls"] = class_recalls
    result["class_f1s"] = class_f1s
    result["total_invalids"] = int(invalid_fp)
    result["percentage_invalids"] = (invalid_fp / len(preds)) * 100
    result["preds"] = preds
    result["gt_actions"] = labels
    
    return result


def _find_sub_dir(disk_root_dir: str, dataset: str) -> str:
    dataset_dir = f"{disk_root_dir}/odinw/test/{dataset}"
    if os.path.exists(dataset_dir):
        return dataset_dir
    else:
        print(f"dataset name cannot be {dataset}, please enter one of the dataset : {ODinWDefinitions.SUB_DATASET_CATEGORIES.keys}")
        return ""

class ODinWBatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_info_dir: str,  batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_info_dir, batch_size, k_shots)
        self.get_dataloader_fn = get_odinw_dataloader
        self.dataset_family = "odinw"

    @property
    def datasets(self):
        if len(self._datasets) == 0:
            for dataset in list(ODinWDefinitions.SUB_DATASET_CATEGORIES.keys()):
                datafiles = self._find_shards(dataset)
                if len(datafiles) != 0:
                    self._datasets.append(dataset)
        return self._datasets

    def _find_shards(self, dataset) -> str:
        return _find_sub_dir(self.disk_root_dir, dataset)
     
    def _send_batch_jobs_for_dataset(self, dataset):
        """Send batch jobs for ODinW dataset."""
        sub_dir = self._find_shards(dataset)
        if not sub_dir:
            print(f"Error finding dataset dir, skipping: {dataset}")
            return {}

        dataloader_obj, dataloader = self.get_dataloader_fn(
            sub_dir, batch_size=self.batch_size
        )
        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            self._send_batch_job(batch, dataset, i)

        print(f"Finished sending jobs for {dataset}.")
        return self.batch_list[dataset]

    def _send_batch_job(self, batch, dataset_name, batch_num):
        """Process the batch to get inputs in the right format for OpenAI batch processing."""
        questions = batch["question"]
        labels = batch["correct_option_idx"]
        image_obs = batch["image"]

        
        inputs_batch = []  # List of inputs for batch processing
        system_prompt = []
        batch_labels = []
        is_lasts = []
        
        for i, (question, image) in enumerate(zip(questions, image_obs)):
            formatted_input = []
            formatted_input.append(('image', image))
            formatted_input.append(('text', question))
            inputs_batch.append(formatted_input)
            
            system_prompt.append(ODinWDefinitions.SYSTEM_PROMPT)
            
            batch_labels.append(labels[i])
            # ODinW doesn't have is_last field, so we set all to True
            is_lasts.append(True)

        # Send batch job to OpenAI
        batch_id, token_count = self.modality_module.send_batch_job(
            inputs_batch, None, system_prompt
        )

        is_lasts = [int(is_last) for is_last in is_lasts]
        labels_array = [int(label) for label in batch_labels]
        output_types = [str] * len(labels_array)

        batch_job = BatchInfo(
            self.dataset_family,
            dataset_name,
            batch_num,
            batch_id,
            output_types,
            token_count,
            is_lasts,
            labels_array,
            len(questions),
            self.batch_info_dir,
            self.model,
        )
        fp = batch_job.save_to_file()
        self.batch_list[dataset_name].append(str(fp))
        

    def _run_eval_dataset(self, dataset_batch_info_paths: Union[str, list[str]]):
        result = {}

        all_preds = []
        all_trues = []
        all_outs = []
        total_invalid_preds = 0
        start_time = time.time()

        if isinstance(dataset_batch_info_paths, str):
            paths = Path(dataset_batch_info_paths).iterdir()
        elif isinstance(dataset_batch_info_paths, list):
            paths = dataset_batch_info_paths
        else:
            raise ValueError(
                "dataset_batch_info_paths should be a path to a folder or a list of filepaths"
            )

        for fp in paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f"Could not find file at path {fp}")

            batch_info = np.load(fp, allow_pickle=True)

            ds = batch_info["dataset_name"].item()
            batch_num = batch_info["batch_num"].item()
            batch_id = batch_info["batch_id"].item()
            labels = batch_info["labels"]
            num_inputs = batch_info["num_inputs"].item()
            output_types = list(batch_info["output_types"])

            status = self.modality_module.get_batch_job_status(batch_id)
            if status == "completed":
                outputs = self.modality_module.retrieve_batch_results(batch_id, output_types)
            else:
                raise Exception(
                    f"Batch not completed for batch {ds} batch num {batch_num} "
                    f"with batch id {batch_id}. Status: {status}. Stopping eval."
                )

            if not isinstance(outputs, list):
                outputs = [outputs]
            all_outs.extend(outputs)

            possible_outputs = list(range(ODinWDefinitions.SUB_DATASET_CATEGORIES[ds]))
            invalid_preds, preds = _validate_outputs_and_calculate_metrics(outputs, possible_outputs)
            total_invalid_preds += invalid_preds
            all_preds.extend(preds)
            all_trues.extend(labels)

            assert len(outputs) == num_inputs, "The length of outputs do not match with the number of processed inputs."

        result = _calculate_final_metrics(all_preds, all_trues, possible_outputs)
        result["eval_time"] = time.time() - start_time
        result["total_invalid_preds"] = total_invalid_preds
        result["all_outs"] = all_outs

        return result   