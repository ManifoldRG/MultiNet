"""
MiniGrid Dataset Module for GenESIS Evaluation

Provides MiniGridModule and MiniGridBatchModule following the DatasetModule pattern.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import json
import glob
import numpy as np
import os
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modules.dataset_modules.base_dataset_module import DatasetModule, DatasetBatchModule, BatchInfo
from definitions.minigrid import MiniGridDefinitions
from definitions.minigrid_prompt import format_instruction_prompt
from src.data_utils.minigrid_dataloader import get_minigrid_dataloader


class MiniGridModule(DatasetModule):
    """
    MiniGrid dataset module for VLM evaluation.

    Follows the same pattern as other DatasetModules in the GenESIS framework.
    """

    def __init__(
        self,
        disk_root_dir: str,
        modality: str = "vlm",
        source: str = "openai",
        model: str = "gpt-4o",
        dataset_name: str = "minigrid",
        batch_size: int = 1,
        k_shots: int = 0,
        tier: Optional[int] = None,
    ):
        """
        Initialize the MiniGrid module.

        Args:
            disk_root_dir: Root directory containing task files
            modality: Modality type (only "vlm" supported)
            source: Model source (e.g., "openai")
            model: Model name
            dataset_name: Dataset name (e.g., "tier1", "tier2", etc.)
            batch_size: Batch size for evaluation
            k_shots: Number of few-shot examples
            tier: Optional tier filter (1-5)
        """
        super().__init__(
            disk_root_dir=disk_root_dir,
            modality=modality,
            source=source,
            model=model,
            dataset_name=dataset_name,
            batch_size=batch_size,
            k_shots=k_shots,
        )

        self._definitions_class = MiniGridDefinitions
        self.dataset_family = "minigrid"
        self.format_instruction_prompt_fn = format_instruction_prompt
        self.get_dataloader_fn = get_minigrid_dataloader
        self.tier = tier

    def _find_shards(self, dataset: str) -> List[str]:
        """
        Find task files for the given dataset.

        Args:
            dataset: Dataset name (e.g., "tier1", "minigrid")

        Returns:
            List of task file paths
        """
        # Look for task files in the expected locations
        search_patterns = [
            f"{self.disk_root_dir}/**/{dataset}*.json",
            f"{self.disk_root_dir}/**/tier*/*.json",
            f"{self.disk_root_dir}/**/*.json",
        ]

        task_files = []
        for pattern in search_patterns:
            found = glob.glob(pattern, recursive=True)
            task_files.extend(found)

        # Remove duplicates and filter by tier if specified
        task_files = list(set(task_files))

        if self.tier is not None:
            task_files = [
                f for f in task_files
                if f"tier{self.tier}" in f or self._task_has_tier(f, self.tier)
            ]

        return sorted(task_files)

    def _task_has_tier(self, path: str, tier: int) -> bool:
        """Check if a task file has the specified tier."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "TaskSpecification" in data:
                data = data["TaskSpecification"]
            return data.get("difficulty_tier", 0) == tier
        except Exception:
            return False

    def _run_eval_dataset(self, dataset: str) -> dict:
        """
        Run evaluation on a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Dictionary of evaluation results
        """
        task_files = self._find_shards(dataset)
        if len(task_files) == 0:
            return {"error": f"No task files found for dataset {dataset}"}

        # Create dataloader
        dataloader_obj, dataloader = self.get_dataloader_fn(
            task_files,
            batch_size=self.batch_size,
            dataset_name=dataset,
            by_episode=True,
        )

        # Initialize metrics
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_labels = []

        for episode_batch in dataloader:
            # Process batch through the module
            for batch_data in self._process_batch(episode_batch, dataset):
                cur_inputs, _, instructions, labels, idxs, output_types, is_lasts = batch_data

                # Get predictions from modality module
                predictions = self.modality_module.get_predictions(
                    cur_inputs, instructions
                )

                # Evaluate predictions
                for pred, label in zip(predictions, labels):
                    total_samples += 1
                    all_predictions.append(pred)
                    all_labels.append(label)

                    # Check correctness (exact match for discrete actions)
                    if self._check_prediction(pred, label):
                        correct_predictions += 1

            if self.action_stats is None:
                self.action_stats = dataloader_obj.action_stats

        # Compute metrics
        accuracy = correct_predictions / max(total_samples, 1)

        return {
            "accuracy": accuracy,
            "exact_match_rate": accuracy,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "predictions": all_predictions,
            "labels": [l.tolist() if hasattr(l, 'tolist') else l for l in all_labels],
        }

    def _check_prediction(self, prediction: Any, label: Any) -> bool:
        """
        Check if prediction matches label.

        Args:
            prediction: Model prediction
            label: Ground truth label

        Returns:
            Whether prediction is correct
        """
        try:
            # Handle various prediction formats
            if isinstance(prediction, list):
                pred_action = prediction[0] if prediction else -1
            elif isinstance(prediction, dict):
                # Handle probability distribution
                pred_action = max(prediction, key=prediction.get)
            else:
                pred_action = prediction

            # Handle label formats
            if isinstance(label, np.ndarray):
                true_action = label[0] if label.size > 0 else -1
            elif isinstance(label, list):
                true_action = label[0] if label else -1
            else:
                true_action = label

            return int(pred_action) == int(true_action)
        except Exception:
            return False


class MiniGridBatchModule(DatasetBatchModule):
    """
    MiniGrid batch module for OpenAI batch API evaluation.

    Supports sending batch jobs and processing results.
    """

    def __init__(
        self,
        disk_root_dir: str,
        modality: str = "vlm",
        source: str = "openai",
        model: str = "gpt-4o",
        batch_info_dir: str = "./batch_info",
        batch_size: int = 1,
        k_shots: int = 0,
        tier: Optional[int] = None,
    ):
        """
        Initialize the MiniGrid batch module.

        Args:
            disk_root_dir: Root directory containing task files
            modality: Modality type
            source: Model source
            model: Model name
            batch_info_dir: Directory for batch info files
            batch_size: Batch size
            k_shots: Number of few-shot examples
            tier: Optional tier filter
        """
        super().__init__(
            disk_root_dir=disk_root_dir,
            modality=modality,
            source=source,
            model=model,
            batch_info_dir=batch_info_dir,
            batch_size=batch_size,
            k_shots=k_shots,
        )

        self._definitions_class = MiniGridDefinitions
        self.dataset_family = "minigrid"
        self.format_instruction_prompt_fn = format_instruction_prompt
        self.get_dataloader_fn = get_minigrid_dataloader
        self.tier = tier

    @property
    def datasets(self):
        """Get list of available datasets."""
        if len(self._datasets) == 0:
            # Default datasets by tier
            self._datasets = [
                "tier1", "tier2", "tier3", "tier4", "tier5"
            ]
            if self.tier is not None:
                self._datasets = [f"tier{self.tier}"]
        return self._datasets

    def _find_shards(self, dataset: str) -> List[str]:
        """Find task files for the given dataset."""
        search_patterns = [
            f"{self.disk_root_dir}/**/{dataset}/*.json",
            f"{self.disk_root_dir}/{dataset}/**/*.json",
            f"{self.disk_root_dir}/**/*.json",
        ]

        task_files = []
        for pattern in search_patterns:
            found = glob.glob(pattern, recursive=True)
            task_files.extend(found)

        task_files = list(set(task_files))

        # Filter by tier in filename or content
        if dataset.startswith("tier"):
            tier_num = int(dataset.replace("tier", ""))
            task_files = [
                f for f in task_files
                if f"tier{tier_num}" in f or self._task_has_tier(f, tier_num)
            ]

        return sorted(task_files)

    def _task_has_tier(self, path: str, tier: int) -> bool:
        """Check if a task file has the specified tier."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if "TaskSpecification" in data:
                data = data["TaskSpecification"]
            return data.get("difficulty_tier", 0) == tier
        except Exception:
            return False

    def _run_eval_dataset(self, batch_info_files: List[str]) -> dict:
        """
        Process batch results for evaluation.

        Args:
            batch_info_files: List of batch info file paths

        Returns:
            Dictionary of evaluation results
        """
        total_samples = 0
        correct_predictions = 0
        all_predictions = []
        all_labels = []

        for batch_file in batch_info_files:
            # Load batch info
            batch_data = np.load(batch_file, allow_pickle=True)

            batch_id = str(batch_data["batch_id"])
            labels = batch_data["labels"]
            output_types = batch_data["output_types"]

            # Get predictions from modality module
            predictions = self.modality_module.get_batch_results(batch_id)

            if predictions is None:
                continue

            # Evaluate predictions
            for pred, label in zip(predictions, labels):
                total_samples += 1
                all_predictions.append(pred)
                all_labels.append(label)

                if self._check_prediction(pred, label):
                    correct_predictions += 1

        accuracy = correct_predictions / max(total_samples, 1)

        return {
            "accuracy": accuracy,
            "exact_match_rate": accuracy,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "predictions": all_predictions,
            "labels": [l.tolist() if hasattr(l, 'tolist') else l for l in all_labels],
        }

    def _check_prediction(self, prediction: Any, label: Any) -> bool:
        """Check if prediction matches label."""
        try:
            if isinstance(prediction, list):
                pred_action = prediction[0] if prediction else -1
            elif isinstance(prediction, dict):
                pred_action = max(prediction, key=prediction.get)
            else:
                pred_action = prediction

            if isinstance(label, np.ndarray):
                true_action = label[0] if label.size > 0 else -1
            elif isinstance(label, list):
                true_action = label[0] if label else -1
            else:
                true_action = label

            return int(pred_action) == int(true_action)
        except Exception:
            return False
