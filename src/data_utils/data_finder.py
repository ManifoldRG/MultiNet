"""
This module is used to find the translated TFDS shards for given datasets.
"""

from glob import glob
from pathlib import Path
from typing import Optional

def find_data_files(dataset_family: str, disk_root_dir: str, dataset: str = None, split: str = 'public') -> list[str]:
    if split not in ['private', 'public']:
        raise ValueError(f"Invalid split: {split}. Must be 'private' or 'public'.")

    split_dir = 'test' if split == 'private' else 'public'

    if dataset_family == 'openx':
        return _find_openx_shards(disk_root_dir, dataset, split_dir)
    elif dataset_family == 'overcooked_ai':
        return _find_overcooked_pickles(disk_root_dir, split_dir)
    elif dataset_family == 'piqa':
        return _find_piqa_jsons(disk_root_dir, split_dir)
    elif dataset_family == 'odinw':
        return _find_odinw_dir(disk_root_dir, dataset, split_dir)
    elif dataset_family == 'sqa3d':
        return _find_sqa3d_data(disk_root_dir, split_dir)
    elif dataset_family == 'bfcl':
        return _find_bfcl_data(disk_root_dir, split_dir)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_family}")

# Finding the translated TFDS shards.
def _find_openx_shards(disk_root_dir: str = None, dataset: Optional[str] = None, split_dir: str = 'test') -> list[str]:
    """
    Find the translated TFDS shards for the OpenX dataset.
    If dataset is None, find all shards for all OpenX datasets.
    """
    if dataset is None: # find all shards for all datasets
        path = f"{disk_root_dir}/openx_*/{split_dir}/"
        all_dataset_dirs = glob(path)
        # go through all dataset dirs and find the shards
        all_shards = []
        for dataset_dir in all_dataset_dirs:
            shard_files = glob(f"{dataset_dir}/translated_shard_*")
            cur_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
            all_shards.extend(cur_shards)

    else:
        path = f"{disk_root_dir}/{dataset}/{split_dir}/translated_shard_*"
        shard_files = glob(path)
        all_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))

    if not all_shards:
        raise ValueError(f"Could not find shards in {path}")

    return all_shards

def _find_overcooked_pickles(disk_root_dir: str = None, split_dir: str = 'test') -> list[str]:
    """
    Find the pickles for the Overcooked AI dataset.
    """
    # Construct the dataset directory path
    dataset_dir = f"{disk_root_dir}/overcooked_ai/{split_dir}/*.pickle"

    # Use glob to find .pickle files
    pickle_files = glob(dataset_dir)
    if not pickle_files:
        raise ValueError(f"Could not find pickles in {dataset_dir}")
    return pickle_files
        
def _find_piqa_jsons(disk_root_dir: str = None, split_dir: str = 'test') -> list[str]:
    """
    Find the JSONL files for the PIQA dataset.
    """
    dataset_dir = f"{disk_root_dir}/piqa/{split_dir}/*.jsonl"
    jsonl_files = glob(dataset_dir)
    if not jsonl_files:
        raise ValueError(f"Could not find JSONL files in {dataset_dir}")
    return jsonl_files

def _find_odinw_dir(disk_root_dir: str = None, dataset: str = None, split_dir: str = 'test') -> list[str]:
    """
    Find the directory for the ODinW dataset.
    """
    dataset_dir = f"{disk_root_dir}/odinw/{split_dir}/{dataset}"
    if not Path(dataset_dir).exists():
        raise ValueError(f"Could not find directory in {dataset_dir}")
    return [dataset_dir]

def _find_sqa3d_data(disk_root_dir: str = None, split_dir: str = 'test') -> list[dict]:
    """
    Find the datasets for the SQA3D dataset.
    """
    dataset_dir = f"{disk_root_dir}/sqa3d/{split_dir}"
    datafiles = glob(dataset_dir + "/*")
    if not datafiles:
        raise ValueError(f"Could not find datafiles in {dataset_dir}")
    question_files = [f for f in datafiles if "question" in f]
    annotation_files = [f for f in datafiles if "annotation" in f]

    if len(question_files) != len(annotation_files):
        raise ValueError(f"Number of question files and annotation files do not match for {dataset_dir}")

    data = []
    for q, a in zip(question_files, annotation_files):
        data_dict = {
            "question_file": q,
            "annotation_file": a,
            "images_dir": dataset_dir,
        }
        data.append(data_dict)
    return data

def _find_bfcl_data(disk_root_dir: str = None, split_dir: str = 'test') -> list[dict]:
    """
    Find the datasets for the BFCL dataset.
    """
    dataset_dir = f"{disk_root_dir}/bfcl_v3/{split_dir}/*"
    datafiles = glob(dataset_dir)
    question_files = [f for f in datafiles if "question" in f]
    answer_files = [f for f in datafiles if "answer" in f]
    if len(question_files) != len(answer_files):
        raise ValueError(f"Number of question files and answer files do not match for {dataset_dir}")
    if len(question_files) == 0:
        raise ValueError(f"Could not find question files in {dataset_dir}")
    data = []
    for q, a in zip(question_files, answer_files):
        data_dict = {
            "question_file": q,
            "answer_file": a,
        }
        data.append(data_dict)
    return data

