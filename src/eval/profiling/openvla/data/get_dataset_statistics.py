import json
import yaml
import os
from datetime import datetime
import numpy as np
import os
import logging
import sys
from pathlib import Path
import tensorflow as tf

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from definitions.openx import OpenXDefinitions
from definitions.procgen import ProcGenDefinitions


DATASET_CONFIG_NAME = 'dataset_statistics_config.yaml'

PROFILING_DATASETS = OpenXDefinitions.DESCRIPTIONS.keys() | ProcGenDefinitions.DESCRIPTIONS.keys()

ENVIRONMENT = os.getenv('ENVIRONMENT', 'prod')
if ENVIRONMENT == 'dev':
    DATASET_STATISTICS_FILE = 'dataset_statistics_dev.json'
else:
    DATASET_STATISTICS_FILE = 'dataset_statistics_prod.json'

logging.basicConfig(
    format='%(asctime)s [%(levelname)-4s] | %(filename)s:%(lineno)d | %(message)s',
)
logger = logging.getLogger(__name__)
if ENVIRONMENT == 'dev':
    logger.setLevel(logging.DEBUG)


class DatasetActionStatisticsCalculator:
    def __init__(self, tfds_shards: list[str], dataset_name: str):
        self.tfds_shards = tfds_shards
        self.action_tensor_size = None
        self.action_stats = None
        self.action_tensors = []
        self.is_last_count = 0
        self.timestep_count = 0
        self.dataset_name = dataset_name
        
    def process_shards(self):
        for shard_idx, shard in enumerate(self.tfds_shards):
            dataset = tf.data.Dataset.load(shard)

            # Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):
                float_action_tensors = []

                if self.dataset_name == 'nyu_door_opening_surprising_effectiveness' or self.dataset_name == 'columbia_cairlab_pusht_real':
                    float_action_tensors.append(np.array(elem['action']['world_vector']))
                    float_action_tensors.append(np.array(elem['action']['rotation_delta']))
                    float_action_tensors.append(np.array(elem['action']['gripper_closedness_action']))
                
                elif self.dataset_name == 'nyu_rot_dataset_converted_externally_to_rlds' or self.dataset_name == 'conq_hose_manipulation' or self.dataset_name == 'plex_robosuite': 
                    float_action_tensors.append(elem['action']) #xyzrpygripper
                
                # Action space is broken for UCSD dataset corresponding to OpenVLA's action space
                elif self.dataset_name == 'ucsd_pick_and_place_dataset_converted_externally_to_rlds' or self.dataset_name == 'usc_cloth_sim_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3]))  # xyz
                    float_action_tensors.append(np.zeros(3))
                    float_action_tensors.append(np.array(elem['action'][3])) # Gripper torque
                
                
                elif self.dataset_name == 'utokyo_pr2_opening_fridge_converted_externally_to_rlds' or self.dataset_name == 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds':
                    float_action_tensors.append(elem['action'][:7])
                
                elif self.dataset_name == 'utokyo_xarm_pick_and_place_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.array(elem['action'][5])) #yaw
                    float_action_tensors.append(np.array(elem['action'][4])) #pitch
                    float_action_tensors.append(np.array(elem['action'][3])) #roll
                    float_action_tensors.append(np.array(elem['action'][6])) #gripper
                
                elif self.dataset_name == 'stanford_mask_vit_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.zeros(2))
                    float_action_tensors.append(np.array(elem['action'][3:])) #gripper
                
                # Action space is broken for this dataset corresponding to OpenVLA's action space
                elif self.dataset_name == 'eth_agent_affordances':
                    float_action_tensors.append(np.array(elem['action'][:6])) #xyz, roll, pitch, yaw
                    float_action_tensors.append(np.zeros(1))
                
                elif self.dataset_name == 'imperialcollege_sawyer_wrist_cam':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.array(elem['action'][5])) # z rotation
                    float_action_tensors.append(np.array(elem['action'][4])) # y rotation
                    float_action_tensors.append(np.array(elem['action'][3])) # x rotation
                    float_action_tensors.append(np.array(elem['action'][6])) #gripper

                elif self.dataset_name == 'utokyo_xarm_bimanual_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.array(elem['action'][5])) #roll
                    float_action_tensors.append(np.array(elem['action'][4])) #pitch
                    float_action_tensors.append(np.array(elem['action'][3])) #yaw
                    float_action_tensors.append(np.array(elem['action'][6])) #gripper
                    float_action_tensors.append(np.array(elem['action'][7:10])) #xyz
                    float_action_tensors.append(np.array(elem['action'][12])) #roll
                    float_action_tensors.append(np.array(elem['action'][11])) #pitch
                    float_action_tensors.append(np.array(elem['action'][10])) #yaw
                    float_action_tensors.append(np.array(elem['action'][13])) #gripper
                    if len(float_action_tensors) == 14:
                        logger.debug(f"float_action_tensors: {float_action_tensors}")

                elif self.dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
                    float_action_tensor = ProcGenDefinitions.set_procgen_unused_special_action_to_stand_still(
                        np.array(elem['actions'][0]), self.dataset_name)
                    float_action_tensors.append(float_action_tensor)

                if float_action_tensors:
                    float_action_tensors = [np.atleast_1d(tensor) for tensor in float_action_tensors]
                    concatenated_action_float = np.concatenate(float_action_tensors, axis=0)

                else:
                    logger.error(f"Undefined action tensor mapping logic for {self.dataset_name}")
                    break

                self.timestep_count += 1
                if elem.get('is_last', False) or elem.get('dones', False):
                    self.is_last_count += 1
                

                self.action_tensors.append(concatenated_action_float)
                

    def _get_action_stats(self):
        if len(self.action_tensors) == 0:
            raise AttributeError("action_stats is None, it has not been populated yet")
        
        action_dim = self.action_tensors[0].shape[0]
        
        mask, discrete = self._define_unnorm_mask_and_discrete_mask(action_dim)
        

        action_stats = {
            "mean": np.mean(self.action_tensors, axis=0).tolist(),
            "std": np.std(self.action_tensors, axis=0).tolist(),
            "max": np.max(self.action_tensors, axis=0).tolist(),
            "min": np.min(self.action_tensors, axis=0).tolist(),
            "q01": np.percentile(self.action_tensors, 1, axis=0).tolist(),
            "q99": np.percentile(self.action_tensors, 99, axis=0).tolist(),
            "mask": mask,
            "discrete": discrete
        }

        if self.dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
            valid_actions = sorted(ProcGenDefinitions.get_valid_action_space(
                self.dataset_name, 'default'))
            if action_stats['max'][0] - action_stats['min'][0] + 1 != len(valid_actions):
                raise ValueError(f"Dataset '{self.dataset_name}' contains invalid actions: {sorted(valid_actions)}")

        return action_stats
    
    def _get_proprio_stats(self):
        if len(self.action_tensors) == 0:
            raise AttributeError("proprio_stats is None, it has not been populated yet")
        
        dim = len(self.action_tensors[0])

        return {
            "mean": np.zeros(dim).tolist(),
            "std": np.zeros(dim).tolist(),
            "max": np.zeros(dim).tolist(),
            "min": np.zeros(dim).tolist(),
            "q01": np.zeros(dim).tolist(),
            "q99": np.zeros(dim).tolist(),
        }

    def _define_unnorm_mask_and_discrete_mask(self, action_dim: int) -> tuple:
        if self.dataset_name in OpenXDefinitions.DESCRIPTIONS.keys():
            mask = [True] * (action_dim - 1) \
                + [True] if self.dataset_name == 'ucsd_pick_and_place_dataset_converted_externally_to_rlds' \
                else [False] * action_dim
            discrete = [False] * action_dim
        elif self.dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
            mask = [True] * action_dim
            discrete = [True] * action_dim
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        return mask, discrete


def load_dataset_path_config():
    config_path = Path(__file__).resolve().parent / DATASET_CONFIG_NAME
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_statistics(file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_path)
    
    if not os.path.exists(full_path):
        with open(full_path, 'w') as f:
            json.dump({}, f)
    
    try:
        with open(full_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading statistics: {e}")
        return {}


def save_dataset_statistics(dataset_statistics, file_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_path)
    
    try:
        with open(full_path, 'w') as f:
            json.dump(dataset_statistics, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving dataset statistics: {e}")


def construct_dataset_folder_paths(dataset: str, config: dict) -> list[str]:
    """
    Construct the parent folder paths for the given dataset.
    """
    if dataset in OpenXDefinitions.DESCRIPTIONS.keys():
        dataset_type = 'openx'
    elif dataset in ProcGenDefinitions.DESCRIPTIONS.keys():
        dataset_type = 'procgen'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    base = config[ENVIRONMENT][dataset_type]['base']
    splits = config[ENVIRONMENT][dataset_type].get('splits', None)

    if splits is None:
        paths = [base]
    else:
        paths = [os.path.join(base, split) for split in splits]

    return paths


def collect_shards(dataset: str, dataset_folder_paths: list[str]) -> list[str]:
    tfds_shards = []
    for path in dataset_folder_paths:
        shard_folder_path = os.path.join(path, dataset)
        shard_files = os.listdir(shard_folder_path)
        
        if dataset in OpenXDefinitions.DESCRIPTIONS.keys():
            sorted_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        elif dataset in ProcGenDefinitions.DESCRIPTIONS.keys():
            sorted_shards = sorted(shard_files, key=lambda x: datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
        tfds_shards.extend([os.path.join(path, dataset, s) for s in sorted_shards])
    return tfds_shards


def get_dataset_action_decoding_strategy(dataset: str) -> str:
    if dataset in OpenXDefinitions.DESCRIPTIONS.keys():
        return OpenXDefinitions.ACTION_DECODE_STRATEGIES.get(
            dataset, OpenXDefinitions.ACTION_DECODE_STRATEGIES.get('default')
        )
    elif dataset in ProcGenDefinitions.DESCRIPTIONS.keys():
        return ProcGenDefinitions.ACTION_DECODE_STRATEGIES.get(
            dataset, ProcGenDefinitions.ACTION_DECODE_STRATEGIES.get('default')
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def calculate_dataset_statistics(stats_calculator: DatasetActionStatisticsCalculator, dataset: str) -> dict:
    try:
        logger.debug(f"Processing {len(stats_calculator.action_tensors)} action tensors for {dataset}")
        action_stats = stats_calculator._get_action_stats()
        proprio_stats = stats_calculator._get_proprio_stats()
        
        return {
            'action': action_stats,
            'proprio': proprio_stats,
            'num_transitions': stats_calculator.timestep_count,
            'num_trajectories': stats_calculator.is_last_count,
            'action_decoding_strategy': get_dataset_action_decoding_strategy(dataset)
        }
    except Exception as e:
        logger.error(f"Error calculating statistics for {dataset}: {str(e)}")
        return None


if __name__ == "__main__":
    try:
        dataset_statistics = load_dataset_statistics(DATASET_STATISTICS_FILE)
    except Exception as e:
        raise Exception(f"Error loading dataset statistics: {str(e)}")

    dataset_path_config = load_dataset_path_config()
    added_dataset_stats_counter = 0

    for dataset in PROFILING_DATASETS:
        if dataset in dataset_statistics and dataset_statistics[dataset] is not None:
            logger.info(f"Skipping {dataset} as it's already in the dataset statistics.")
            continue

        dataset_folder_paths = construct_dataset_folder_paths(dataset, dataset_path_config)

        if dataset not in os.listdir(dataset_folder_paths[0]):
            logger.warning(f"{dataset} not found in '{dataset_folder_paths[0]}'")
            continue
        
        try:
            tfds_shards = collect_shards(dataset, dataset_folder_paths)

            if not tfds_shards:
                logger.warning(f"{dataset} has no shards.")
                continue

            logger.info(f"Calculating dataset statistics for {dataset} with {len(tfds_shards)} shards.")
            stats_calculator = DatasetActionStatisticsCalculator(tfds_shards, dataset)
            stats_calculator.process_shards()

            # If we have processed shards successfully
            if len(stats_calculator.action_tensors) > 0:
                dataset_statistics[dataset] = calculate_dataset_statistics(stats_calculator, dataset)
                added_dataset_stats_counter += 1
            else:
                logger.warning(f"No action tensors were gathered for dataset {dataset}")
        
            save_dataset_statistics(dataset_statistics, DATASET_STATISTICS_FILE)

        except Exception as e:
            logger.exception(f"Error processing dataset {dataset}: {str(e)}")
            raise

    logger.info(
        f"""
        ==============Dataset statistics updated================
        - Added {added_dataset_stats_counter} datasets.
        - Total datasets: {len(dataset_statistics)}
        ========================================================
        """
    )
