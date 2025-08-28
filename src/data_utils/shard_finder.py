"""
This module is used to find the translated TFDS shards for given datasets.
"""

from glob import glob
from typing import Optional

def find_shards(dataset_family: str, disk_root_dir: str, dataset: str = None, split: str = 'public') -> list[str]:
    if dataset_family == 'openx':
        return _find_openx_shards(disk_root_dir, dataset, split)
    else:
        raise ValueError(f"Invalid dataset type: {dataset}")

# Finding the translated TFDS shards.
def _find_openx_shards(disk_root_dir: str = None, dataset: Optional[str] = None, split: str = 'public') -> list[str]:
    """
    Find the translated TFDS shards for the OpenX dataset.
    If dataset is None, find all shards for all datasets.
    """
    if split not in ['private', 'public']:
        raise ValueError(f"Invalid split: {split}. Must be 'private' or 'public'.")
    split_dir = 'test' if split == 'private' else 'public'

    if dataset is None: # find all shards for all datasets
        all_dataset_dirs = glob(f"{disk_root_dir}/openx_*/{split_dir}/")
        # go through all dataset dirs and find the shards
        all_shards = []
        for dataset_dir in all_dataset_dirs:
            shard_files = glob(f"{dataset_dir}/translated_shard_*")
            cur_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
            all_shards.extend(cur_shards)
            
        return all_shards
    else:
        try:
            dataset_dir = glob(f"{disk_root_dir}/{dataset}/{split_dir}/")[0] 
            shard_files = glob(f"{dataset_dir}/translated_shard_*")
            tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
            return tfds_shards
        except IndexError:
            print(f"Cannot identify the directory to the dataset {dataset}. Skipping this dataset.")
            return []