from src.modules.dataset_modules.base_dataset_module import DatasetBatchModule, BatchInfo, DatasetModule
from src.data_utils.piqa_dataloader import get_piqa_dataloader
from src.modules.source_modules.openai_module import OpenAIModule
from pathlib import Path
import numpy as np
import time
from glob import glob
from typing import Union
from src.eval_utils import get_exact_match_rate

def _validate_output(output) -> bool:
    """Validate that output is exactly '0' or '1'"""
    if not isinstance(output, str):
        return False
    return output.strip() in ['0', '1']


def _validate_outputs_and_calculate_metrics(outputs, labels):
    """Validate outputs and convert to predictions for exact match calculation"""
    preds = []
    total_invalid_preds = 0
    
    for output in outputs:
        if _validate_output(output):
            preds.append(int(output.strip()))
        else:
            total_invalid_preds += 1
            preds.append(-1)
    
    return total_invalid_preds, preds


def _calculate_final_metrics(preds, trues):
    """Calculate final metrics for PIQA evaluation"""
    result = {}
    
    valid_preds = []
    valid_trues = []
    invalid_count = 0
    
    for pred, true in zip(preds, trues):
        if pred == -1:  # Invalid prediction
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)
    
    # Calculate exact match rate only on valid predictions
    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0
    
    # Calculate exact match rate including invalids as wrong (more conservative metric)
    total_predictions = len(preds)
    correct_predictions = sum(1 for pred, true in zip(preds, trues) if pred == true and pred != -1)
    exact_match_rate_with_invalids = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    result["exact_match_rate"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["total_predictions"] = total_predictions
    result["valid_predictions"] = len(valid_preds)
    result["invalid_predictions"] = invalid_count
    result["percentage_invalids"] = (invalid_count / total_predictions) * 100 if total_predictions > 0 else 0.0
    result["preds"] = [int(pred) for pred in preds]
    result["gt_labels"] = [int(true) for true in trues]
    
    return result


def _find_jsonl_file(dataset, disk_root_dir: str) -> str:
    try:
        dataset_dir = f"{disk_root_dir}/{dataset}/test"
        jsonl_files = glob(f"{dataset_dir}/*.jsonl")
        if not jsonl_files:
            print(f"No .jsonl files found in {dataset_dir}")
        return jsonl_files[0]
    except Exception as e:
        print(f"Cannot identify the directory to the dataset. Skipping this dataset. Error: {e}")
        return ""


class PIQAModule(DatasetModule):
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
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        
        self.get_dataloader_fn = get_piqa_dataloader
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_family = "piqa"
        
    @property
    def datasets(self):
        if len(self._datasets) == 0:
            jsonl_file = self._find_shards()
            if jsonl_file:
                self._datasets.append(self.dataset_family)
        return self._datasets
        
    @property
    def modality_module(self):
        self._modality_module = OpenAIModule(
            self.model,
            max_concurrent_prompts=400,
            max_output_tokens_per_query=16,  # Only need 1 token for "0" or "1", but minimum is 16
        )
        return self._modality_module

    def _find_shards(self) -> str:
        return _find_jsonl_file(self.dataset_name, self.disk_root_dir)
    
    def _run_eval_dataset(self, dataset: str) -> dict:
        result = {}

        try:
            piqa_jsonl = self._find_shards()
            if not piqa_jsonl:
                return {}

            start_time = time.time()

            dataloader_obj, dataloader = self.get_dataloader_fn(
                piqa_jsonl,
                batch_size=self.batch_size,
            )
            
            all_preds = []
            all_trues = []
            total_invalid_preds = 0

            print(f"Running evaluation for dataset: {dataset}...")
            
            for batch_idx, batch in enumerate(dataloader):
                questions = batch["question"]
                labels = batch["label"]
                
                batch_outputs = []
                
                for i, question in enumerate(questions):
                    formatted_input = [('text', question)]
                    
                    try:
                        self.modality_module.clear_history()
                        
                        output = self.modality_module.infer_step(
                            inputs=formatted_input,
                            system_prompt=None
                        )
                        batch_outputs.append(output)
                    except Exception as e:
                        print(f"Error during inference at batch {batch_idx}, sample {i}: {e}")
                        batch_outputs.append("")
                
                # Validate outputs and calculate metrics for this batch
                invalid_preds, preds = _validate_outputs_and_calculate_metrics(batch_outputs, labels)
                total_invalid_preds += invalid_preds
                all_preds.extend(preds)
                all_trues.extend(labels)
                
                # Progress update
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches...")

            result = _calculate_final_metrics(all_preds, all_trues)
            result["eval_time"] = time.time() - start_time
            result["total_invalid_preds"] = total_invalid_preds

        except Exception as e:
            print(f"Error occurred during evaluation: {e}")
            print(f"Cannot evaluate dataset {dataset}. Moving on...")
            return {}

        print(f"Evaluation completed for {dataset}")
        return result


class PIQABatchModule(DatasetBatchModule):
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
        self.get_dataloader_fn = get_piqa_dataloader
        self.disk_root_dir = disk_root_dir
        self.source = source
        self.batch_size = batch_size
        self.dataset_family = "piqa"
        self.dataset_name = "piqa"

    @property
    def datasets(self):
        if len(self._datasets) == 0:
            jsonl_file = self._find_shards(self.dataset_name)
            if jsonl_file:
                self._datasets.append(self.dataset_family)
        return self._datasets

    @property
    def modality_module(self):
        self._modality_module = OpenAIModule(
            self.model,
            max_concurrent_prompts=400,
            max_output_tokens_per_query=16,
        )
        return self._modality_module

    def _find_shards(self, dataset: str) -> str:
        return _find_jsonl_file(dataset, self.disk_root_dir)

    def _send_batch_jobs_for_dataset(self, dataset):
        """Send batch jobs for PIQA dataset."""
        piqa_jsonl = self._find_shards(dataset)
        if not piqa_jsonl:
            return {}

        dataloader_obj, dataloader = self.get_dataloader_fn(
            piqa_jsonl, batch_size=self.batch_size
        )

        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            self._send_batch_job(batch, dataset, i)

        print(f"Finished sending jobs for {dataset}.")
        return self.batch_list[dataset]

    def _send_batch_job(self, batch, dataset_name, batch_num):
        """Process the batch to get inputs in the right format for OpenAI batch processing."""
        questions = batch["question"]
        labels = batch["label"]
        
        inputs_batch = []  # List of inputs for batch processing
        system_prompt = []
        batch_labels = []
        is_lasts = []
        
        for i, question in enumerate(questions):
            formatted_input = [('text', question)]
            inputs_batch.append(formatted_input)
            
            system_prompt.append("")
            
            batch_labels.append(labels[i])
            # PIQA doesn't have is_last field, so we set all to True
            is_lasts.append(True)

        # Send batch job to OpenAI
        batch_responses, batch_id, token_count = self.modality_module.batch_infer_step(
            inputs_batch, system_prompt, retrieve_and_return_results=False
        )
        
        is_lasts = [int(is_last) for is_last in is_lasts]
        labels_array = [int(label) for label in batch_labels]
        output_types = ['text'] * len(labels_array)

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

            status = self.modality_module.get_batch_job_status(batch_id)
            if status == "completed":
                outputs = self.modality_module.retrieve_batch_results(batch_id)
            else:
                raise Exception(
                    f"Batch not completed for batch {ds} batch num {batch_num} "
                    f"with batch id {batch_id}. Status: {status}. Stopping eval."
                )

            if not isinstance(outputs, list):
                outputs = [outputs]

            invalid_preds, preds = _validate_outputs_and_calculate_metrics(outputs, labels)
            total_invalid_preds += invalid_preds
            all_preds.extend(preds)
            all_trues.extend(labels)

            assert len(outputs) == num_inputs, "The length of outputs do not match with the number of processed inputs."

        result = _calculate_final_metrics(all_preds, all_trues)
        result["eval_time"] = time.time() - start_time
        result["total_invalid_preds"] = total_invalid_preds

        return result