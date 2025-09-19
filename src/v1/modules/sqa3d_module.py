from src.modules.dataset_modules.base_dataset_module import (
    DatasetBatchModule,
    BatchInfo,
)
from src.data_utils.sqa3d_dataloader import get_sqa3d_dataloader, get_sqa3d_test_dataloader
from src.modules.modality_modules.vlm_module import VLMModule
from src.eval_utils import get_exact_match_rate
from pathlib import Path
from typing import Union
import numpy as np
import time
import string
from typing import Any
from glob import glob
import re
from definitions.sqa3d_prompt import SQA3DDefinitions
from sentence_transformers import SentenceTransformer, util

def _validate_output(output) -> bool:
    """Validate that output is a non-empty string"""
    return isinstance(output, str) and len(output.strip()) > 0


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and extra spaces."""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def _find_data_files(disk_root_dir: str) -> tuple:
    """Find SQA3D data files (questions, annotations, images)"""
    test_dir = f"{disk_root_dir}/sqa3d/test"
    test_path = Path(test_dir)
    
    # Look for standard test files
    questions_file = test_path / "v1_balanced_questions_test_scannetv2.json"
    annotations_file = test_path / "v1_balanced_sqa_annotations_test_scannetv2.json"
    
    if not questions_file.exists():
        raise FileNotFoundError(f"Test questions file not found: {questions_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Test annotations file not found: {annotations_file}")
    
    # Images should be in scene subdirectories
    images_dir = test_path
    return {
        "questions_file": str(questions_file),
        "annotations_file": str(annotations_file),
        "images_dir": str(images_dir)
    }


def _validate_outputs_and_calculate_metrics(model: SentenceTransformer, outputs: list[str], labels: list[str]):
    """Validate outputs and calculate text similarity metrics for SQA3D"""
    exact_matches = []
    similarity_scores = []
    total_invalid_preds = 0
    preds = []
    for i, output in enumerate(outputs):
        if _validate_output(output):
            normalized_output = _normalize_text(output)
            normalized_label = _normalize_text(labels[i])
            
            # Calculate exact match
            exact_match = 1.0 if normalized_output == normalized_label else 0.0
            exact_matches.append(exact_match)
            
            # Calculate similarity score
            emb1 = model.encode(output, convert_to_tensor=True)
            emb2 = model.encode(labels[i], convert_to_tensor=True)
            
            similarity = util.cos_sim(emb1, emb2).item()
            similarity_scores.append(similarity)

            preds.append(normalized_output)
        else:
            # Invalid output - assign worst possible scores
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            total_invalid_preds += 1
            preds.append("")
    
    return exact_matches, similarity_scores, total_invalid_preds, preds


def _calculate_final_metrics(exact_matches, similarity_scores, total_invalid_preds):
    """Calculate final metrics for SQA3D VQA evaluation"""
    result = {}
    
    # Calculate accuracy metrics
    total_samples = len(exact_matches)
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    exact_match_accuracy_without_invalids = sum(exact_matches) / (total_samples - total_invalid_preds) if total_samples - total_invalid_preds > 0 else 0.0
    
    # Calculate similarity metrics
    avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
    max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
    min_similarity_score = min(similarity_scores) if similarity_scores else 0.0
    
    # Calculate additional statistics
    similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
    
    # Calculate percentage of high similarity matches (threshold-based)
    high_similarity_threshold = 0.8
    high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
    high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0
    
    # Calculate invalid prediction percentage
    invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
    
    result['num_timesteps'] = total_samples
    result['exact_match_rate'] = exact_match_accuracy
    result['exact_match_rate_without_invalids'] = exact_match_accuracy_without_invalids
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_invalids'] = total_invalid_preds
    result['percentage_invalids'] = invalid_percentage

    return result


class SQA3DBatchModule(DatasetBatchModule):
    def __init__(
        self,
        disk_root_dir: str,
        modality: str,
        source: str,
        model: str,
        batch_info_dir: str,
        batch_size: int = 1,
        k_shots: int = 0,
        similarity_model: str = "all-MiniLM-L6-v2",
    ):
        super().__init__(
            disk_root_dir, modality, source, model, batch_info_dir, batch_size, k_shots
        )
        self.get_dataloader_fn = get_sqa3d_dataloader
        self.disk_root_dir = disk_root_dir
        self.source = source
        self.batch_size = batch_size
        self.dataset_family = "sqa3d"
        self.dataset_name = "sqa3d"

        # Explicitly setting clean_up_tokenization_spaces because the default behaviour will 
        # change to False in transformers v4.45
        self.similarity_model = SentenceTransformer(similarity_model, 
                                tokenizer_kwargs={"clean_up_tokenization_spaces": True})

    def _find_shards(self, dataset: str) -> dict:
        return _find_data_files(self.disk_root_dir)

    def _send_batch_jobs_for_dataset(self, dataset):
        """Send batch jobs for SQA3D dataset."""
        data_locations = self._find_shards(dataset)
        questions_file = data_locations["questions_file"]
        annotations_file = data_locations["annotations_file"]
        images_dir = data_locations["images_dir"]
        if questions_file is None or annotations_file is None or images_dir is None:
            raise FileNotFoundError(f"Could not find data files for dataset {dataset}")

        dataloader_obj, dataloader = self.get_dataloader_fn(
            questions_file, annotations_file, images_dir, batch_size=self.batch_size
        )

        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            self._send_batch_job(batch, dataset, i)

        print(f"Finished sending jobs for {dataset}.")
        return self.batch_list[dataset]

    def _send_batch_job(self, batch, dataset_name, batch_num):
        """Process the batch to get inputs in the right format for VLM batch processing."""
        questions = batch["question"]
        answers = batch["answer"]
        scene_images = batch.get("scene_image", [None] * len(questions))
        
        inputs_batch = []  # List of inputs for batch processing
        system_prompt = []
        batch_labels = []
        is_lasts = []
        
        for question, answer, scene_image in zip(questions, answers, scene_images):
            # Format input for VLM
            formatted_input = [('text', question)]
            
            # Add scene image if available
            # TODO: Should we profile on missing images?
            if scene_image is not None:
                formatted_input.append(('image', scene_image))
            
            inputs_batch.append(formatted_input)
            system_prompt.append(SQA3DDefinitions.SYSTEM_PROMPT)
            batch_labels.append(answer)
            # SQA3D doesn't have is_last field, so we set all to True
            is_lasts.append(True)

        # Send batch job to VLM
        batch_id, token_count = self.modality_module.send_batch_job(
            inputs_batch, None, system_prompt
        )
        
        is_lasts = [int(is_last) for is_last in is_lasts]
        output_types = [str] * len(batch_labels)

        batch_job = BatchInfo(
            self.dataset_family,
            dataset_name,
            batch_num,
            batch_id,
            output_types,
            token_count,
            is_lasts,
            batch_labels,
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
        normalized_preds = []
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

            exact_matches, similarity_scores, invalid_preds, preds = _validate_outputs_and_calculate_metrics(self.similarity_model, outputs, labels)
            total_invalid_preds += invalid_preds
            all_preds.extend(outputs)
            all_trues.extend(labels)
            normalized_preds.extend(preds)

            assert len(outputs) == num_inputs, "The length of outputs do not match with the number of processed inputs."

        result = _calculate_final_metrics(exact_matches, similarity_scores, total_invalid_preds)
        result["eval_time"] = time.time() - start_time
        result['preds'] = normalized_preds
        result['gt_actions'] = all_trues
        result['all_outs'] = all_preds

        return result
