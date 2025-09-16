from src.modules.dataset_modules.base_dataset_module import DatasetBatchModule, BatchInfo, DatasetModule
from src.data_utils.odinw_dataloader import get_odinw_dataloader
from src.modules.source_modules.openai_module import OpenAIModule
from definitions.odinw import ODinWDefinitions
from pathlib import Path
import numpy as np
import time
import os
import re
from glob import glob
from typing import Any, Union


def _validate_output(output) -> bool:
    """Validate that output is exactly '0' or '1'"""
    if not isinstance(output, str):
        return False
    return output.strip() in ['0', '1']


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def _find_sub_dir(dataset, disk_root_dir: str, sub_dataset_dir: str) -> str:
    out = []
    if sub_dataset_dir == "all":
        try:
            dataset_dir = f"{disk_root_dir}/{dataset}/test"
            sub_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            if not sub_dirs:
                print(f"No subdirectories found in {dataset_dir}")
                return ""
            for i in range(len(sub_dirs)):
                out.append(os.path.join(dataset_dir, sub_dirs[i]))
            return out
        except Exception as e:
            print(f"Cannot identify the directory to the dataset. Skipping this dataset. Error: {e}")
            return ""
    else:
        try:
            dataset_dir = f"{disk_root_dir}/{dataset}/test/{sub_dataset_dir}"
            if os.path.exists(dataset_dir):
                out.append(dataset_dir)
            return out
        except Exception as e:
            print(f"Cannot identify the directory to the dataset. Skipping this dataset. Error: {e}")
            print(f"sub_dataset_dir cannot be {sub_dataset_dir}, please enter one of the dataset : {ODinWDefinitions.SUB_DATASET_NAMES}")
            return ""


class ODinWBatchModule(DatasetBatchModule):
    def __init__(self, disk_root_dir: str, modality: str, source: str, model: str, batch_info_dir: str, sub_dataset_dir: int, batch_size: int = 1, k_shots: int = 0):
        super().__init__(disk_root_dir, modality, source, model, batch_info_dir, batch_size, k_shots)
        self.get_dataloader_fn = get_odinw_dataloader
        self.disk_root_dir = disk_root_dir
        self.source = source
        self.batch_size = batch_size
        self.dataset_family = "odinw"
        self.dataset_name = "odinw"
        self.sub_dataset_dir = sub_dataset_dir
        
        
    @property
    def datasets(self):
        if len(self._datasets) == 0:
            sub_dirs = self._find_shards()
            if sub_dirs:
                self._datasets.extend(sub_dirs)
        return self._datasets
    
    @property
    def modality_module(self):
        self._modality_module = OpenAIModule(model = self.model, max_concurrent_prompts=400)
        return self._modality_module
    
    
    def _find_shards(self) -> str:
        return _find_sub_dir(self.dataset_name, self.disk_root_dir)
    

    def _send_batch_jobs_for_dataset(self, dataset):
        """Send batch jobs for PIQA dataset."""
        sub_dir = self._find_shards(dataset)
        if not sub_dir:
            return {}

        dataloader_obj, dataloader = self.get_dataloader_fn(
            sub_dir, batch_size=self.batch_size
        )

        print(f"Sending batch jobs for dataset: {dataset}...")
        for i, batch in enumerate(dataloader):
            self._send_batch_job(batch, dataset, i)

        print(f"Finished sending jobs for {dataset}.")
        return self.batch_list[dataset]
    
    
