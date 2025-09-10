from src.modules.dataset_modules.base_dataset_module import DatasetBatchModule, BatchInfo, DatasetModule
from src.data_utils.openx_dataloader import get_openx_dataloader
from src.modules.source_modules.openai_module import OpenAIModule
from definitions.robovqa_prompt import ROBOVQA_PROMPT
from pathlib import Path
import numpy as np
import time
import re
from glob import glob
from typing import Any, Union
from sentence_transformers import SentenceTransformer, util


def _validate_text_output(output: Any) -> bool:
    """Validate that output is a valid text string."""
    if output is None:
        return False
    if isinstance(output, str) and len(output.strip()) > 0:
        return True
    return False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def _validate_outputs_and_calculate_metrics(model: SentenceTransformer, outputs: list[str], labels: list[str]):
    """Validate outputs and calculate text similarity metrics for RoboVQA."""
    exact_matches = []
    similarity_scores = []
    total_invalid_preds = 0
    
    for i, output in enumerate(outputs):
        if _validate_text_output(output):
            # Normalize both output and label for fair comparison
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
        else:
            # Invalid output - assign worst possible scores
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            total_invalid_preds += 1
    
    return exact_matches, similarity_scores, total_invalid_preds



def _calculate_final_metrics(exact_matches, similarity_scores, total_invalid_preds):
    """Calculate comprehensive final metrics for RoboVQA evaluation."""
    result = {}
    
    # Calculate accuracy metrics
    total_samples = len(exact_matches)
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    
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
    
    result['exact_match_accuracy'] = exact_match_accuracy
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_samples'] = total_samples
    result['total_invalid_preds'] = total_invalid_preds
    result['invalid_percentage'] = invalid_percentage
    
    return result


def _find_shards(dataset, disk_root_dir: str) -> list[str]:
    try:
        # Look for RoboVQA dataset directory pattern
        dataset_dir = f"{disk_root_dir}/{dataset}/test"
        shard_files = glob(f"{dataset_dir}/translated_shard_*")
        tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        return tfds_shards
    except IndexError:
        print("Cannot identify the directory to the dataset. Skipping this dataset.")
        return []
    
    

class RoboVQAModule(DatasetModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, dataset_name: str,  batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_size, k_shots)
        self.get_dataloader_fn = get_openx_dataloader
        self.dataset = dataset_name
        self.batch_size = batch_size
        self.model = model
        self.similarity_model = None
        self.dataset_family = "robot_vqa"
        
    
    def _find_shards(self, dataset: str) -> list[str]:
        return _find_shards("openx_multi_embodiment", self.disk_root_dir)
    
    
    @property
    def descriptions(self):
        if self._definitions_class is None:
            return {}
        else:
            return self._definitions_class.DESCRIPTIONS
        
    
    @property
    def datasets(self):
        if len(self._datasets) == 0:
            tfds_shards = self._find_shards(self.dataset_family)
            if len(tfds_shards) != 0:
                self._datasets.append(self.dataset_family)
        return self._datasets
        
    
    @property
    def modality_module(self):
        self._modality_module = OpenAIModule(model = self.model, max_concurrent_prompts=400)
        return self._modality_module
    
    
    
    def _run_eval_dataset(self, dataset: str) -> dict:
        """Run evaluation for a single dataset using single inference."""
        tfds_shards = self._find_shards(dataset)
        if len(tfds_shards) == 0:
            return {}
        # Create dataloader
        dataloader_obj, dataloader = self.get_dataloader_fn(
            tfds_shards,
            batch_size=self.batch_size,
            dataset_name=dataset,
            by_episode=False
        )
        
        exact_matches = []
        similarity_scores = []
        start_time = time.time()
        total_invalid_preds = 0
        
        # Initialize similarity model if not already done
        if self.similarity_model is None:
            print("Loading similarity model...")
            self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        
        print(f"Running evaluation for dataset: {dataset}...")
        
        for batch_idx, batch in enumerate(dataloader):
            # Process each timestep in the batch
            text_obs = batch['text_observation']
            num_timesteps = len(text_obs)
            
            batch_outputs = []
            batch_labels = []
            
            for t in range(num_timesteps):
                # Prepare input for single inference
                formatted_input = []
                
                # Get the question text and format it
                question_text = text_obs[t]
                
                # Add image if present
                if 'image_observation' in batch and batch['image_observation'][t] is not None:
                    image_obs = batch['image_observation'][t]
                    formatted_input.append(('image', image_obs))
                
                # Add the question text
                formatted_input.append(('text', question_text))
                
                # Run single inference
                try:
                    # Clear history before each inference to ensure clean context
                    self.modality_module.clear_history()
                    
                    # Perform inference
                    output = self.modality_module.infer_step(
                        inputs=formatted_input,
                        system_prompt=ROBOVQA_PROMPT
                    )
                    
                    batch_outputs.append(output)
                except Exception as e:
                    print(f"Error during inference at batch {batch_idx}, timestep {t}: {e}")
                    batch_outputs.append("")  # Add empty string for failed inference
                
                # Collect label
                batch_labels.append(batch['text_answer'][t])
            
            # Calculate metrics for this batch
            if batch_outputs and batch_labels:
                matches, scores, invalid = _validate_outputs_and_calculate_metrics(
                    self.similarity_model, 
                    batch_outputs, 
                    batch_labels
                )
                
                exact_matches.extend(matches)
                similarity_scores.extend(scores)
                total_invalid_preds += invalid
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")
        
        # Calculate final metrics
        result = _calculate_final_metrics(exact_matches, similarity_scores, total_invalid_preds)
        result['eval_time'] = time.time() - start_time
        
        
        print(f"Evaluation completed for {dataset}")
        
        return result
    
    
    
class RoboVQABatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_info_dir: str, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_info_dir, batch_size, k_shots)
        self.get_dataloader_fn = get_openx_dataloader  # Reusing OpenX dataloader
        self.dataset_family = 'robot_vqa'
        self.similarity_model = None
        self.disk_root_dir = disk_root_dir
        self._datasets = []
        self.batch_size = batch_size
        self.dataset = "openx_multi_embodiment"
        self.model = model

    
    def _find_shards(self) -> list[str]:
        return _find_shards(self.dataset, self.disk_root_dir)
    
    
    @property
    def descriptions(self):
        if self._definitions_class is None:
            return {}
        else:
            return self._definitions_class.DESCRIPTIONS
        
    
    @property
    def datasets(self):
        if len(self._datasets) == 0:
            tfds_shards = self._find_shards()
            if len(tfds_shards) != 0:
                self._datasets.append(self.dataset_family)
        return self._datasets
        
    
    @property
    def modality_module(self):
        self._modality_module = OpenAIModule(model = self.model, max_concurrent_prompts=400)
        return self._modality_module
        
    
    # Add this method inside the RoboVQABatchModule class
    def _send_batch_jobs_for_dataset(self, dataset):
        """
        Custom version for RoboVQA that bypasses the action_stats logic.
        """
        tfds_shards = self._find_shards()
        if len(tfds_shards) == 0:
            return {}

        # Creating the dataloader, getting both the object and the iterable
        dataloader_obj, dataloader = self.get_dataloader_fn(
            tfds_shards,
            batch_size=self.batch_size,
            dataset_name=dataset,
            by_episode=False
        )

        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            # We removed the problematic action_stats line here
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
        similarity_scores = []
        start_time = time.time()
        total_invalid_preds = 0
        
        # If it's a folder path, iterate over all files in the folder
        if isinstance(dataset_batch_info_paths, str):
            paths = Path(dataset_batch_info_paths).iterdir()
        elif isinstance(dataset_batch_info_paths, list):
            paths = dataset_batch_info_paths
        else:
            raise ValueError("data_batch_info_paths should be a path to a folder or a list of filepaths")
            
        for fp in paths:
            if not Path(fp).exists():
                raise FileNotFoundError(f'Could not find file at path {fp}') 
            
            batch_info = np.load(fp, allow_pickle=True)    

            # output_types = list(batch_info['output_types'])
            ds = batch_info['dataset_name'].item()
            batch_num = batch_info['batch_num'].item()
            batch_id = batch_info['batch_id'].item()
            labels = [str(label) for label in batch_info['labels']]  # Text labels for RoboVQA
            num_inputs = batch_info['num_inputs'].item()
            # is_lasts = [bool(is_last) for is_last in batch_info['is_lasts']]
            
            status = self.modality_module.get_batch_job_status(batch_id)
            if status == 'completed':
                outputs = self.modality_module.retrieve_batch_results(batch_id)
            else:
                raise Exception(f'Batch not completed for batch {ds} batch num {batch_num} '
                                f'with batch id {batch_id}. Status: {status}. Stopping eval.')
            if self.similarity_model is None:
                print("Loading similarity model...")
                self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device = "cpu") # causes version mismatch with recent cuda version and torch in the conda env
            matches, similarity_score, invalid_preds = _validate_outputs_and_calculate_metrics(self.similarity_model, outputs, labels)
            total_invalid_preds += invalid_preds
                    
            assert len(matches) == num_inputs, "The length of calculated metrics list do not match with the length number of processed inputs."

            exact_matches.extend(matches)
            similarity_scores.extend(similarity_score)
        
        result = _calculate_final_metrics(exact_matches, similarity_scores, total_invalid_preds)
        result['eval_time'] = time.time() - start_time
        
        return result

    # Override the batch processing method for RoboVQA
    def _send_batch_job(self, batch, dataset_name, batch_num):
        # Process the batch to get inputs in the right format for OpenAI
        text_obs = batch['text_observation']
        num_timesteps = len(text_obs)
        
        inputs_batch = []  # List of inputs for batch processing
        system_prompt = []
        labels = []
        is_lasts = []
        
        for t in range(num_timesteps):
            # Prepare input for each timestep
            formatted_input = []
            question_text = text_obs[t]
            
            if 'image_observation' in batch and batch['image_observation'][t] is not None:
                image_obs = batch['image_observation'][t]
                formatted_input.append(('image', image_obs))

            formatted_input.append(('text', question_text))
            inputs_batch.append(formatted_input)
            
            # Simple system prompt for VQA
            system_prompt.append(
                ROBOVQA_PROMPT
                )
            
            labels.append(batch['text_answer'][t])
            is_lasts.append(batch['is_last'][t])

        # Send batch job to OpenAI
        batch_responses, batch_id, token_count = self.modality_module.batch_infer_step(
            inputs_batch, system_prompt, retrieve_and_return_results=False
        )
        
        # Convert to numpy arrays for saving
        is_lasts = [int(is_last) for is_last in is_lasts]
        labels = [str(label) for label in labels]
        output_types = ['text'] * len(labels)

        
        batch_job = BatchInfo(
            self.dataset_family, dataset_name, batch_num, batch_id, 
            output_types, token_count, is_lasts, labels, num_timesteps, 
            self.batch_info_dir, self.model
        )
        fp = batch_job.save_to_file()
        self.batch_list[dataset_name].append(str(fp))