from glob import glob
from typing import Optional

def find_shards(dataset_family: str, disk_root_dir: str, dataset: str = None) -> list[str]:
    if dataset_family == 'openx':
        return _find_openx_shards(dataset, disk_root_dir)
    else:
        raise ValueError(f"Invalid dataset type: {dataset}")

# Finding the translated TFDS shards.
def _find_openx_shards(dataset: Optional[str], disk_root_dir: str) -> list[str]:
    if dataset is None: # find all shards for all datasets
        all_dataset_dirs = glob(f"{disk_root_dir}/openx_*/")
        # go through all dataset dirs and find the shards
        tfds_shards = []
        for dataset_dir in all_dataset_dirs:
            shard_files = glob(f"{dataset_dir}/translated_shard_*")
            tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
            tfds_shards.extend(tfds_shards)
        return tfds_shards
    else:
        try:
            dataset_dir = glob(f"{disk_root_dir}/openx_*/{dataset}")[0]
            shard_files = glob(f"{dataset_dir}/translated_shard_*")
            tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
            return tfds_shards
        except IndexError:
            print(f"Cannot identify the directory to the dataset {dataset}. Skipping this dataset.")
            return []