import json

from typing import List, Dict, Any
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import logging


IS_DEV = True

logger = logging.getLogger(__name__)
if IS_DEV:
    logger.setLevel(logging.DEBUG)


class OpenXDataset:
    def __init__(self, tfds_shards: List[str]):
        self.tfds_shards = tfds_shards
        self.action_tensor_size = None
        self.action_stats = None
        self.action_tensors = []
        self.is_last_count = 0
        self.timestep_count = 0
        
    def _process_shards(self, dataset_name: str):

        for shard_idx, shard in enumerate(self.tfds_shards):
            
            print(shard)
            dataset = tf.data.Dataset.load(shard)

            #Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):
                    
                float_action_tensors = []

                if dataset_name == 'nyu_door_opening_surprising_effectiveness' or dataset_name == 'columbia_cairlab_pusht_real':
                    float_action_tensors.append(np.array(elem['action']['world_vector']))
                    float_action_tensors.append(np.array(elem['action']['rotation_delta']))
                    float_action_tensors.append(np.array(elem['action']['gripper_closedness_action']))
                
                elif dataset_name == 'nyu_rot_dataset_converted_externally_to_rlds' or dataset_name == 'conq_hose_manipulation' or dataset_name == 'plex_robosuite': 
                    float_action_tensors.append(elem['action']) #xyzrpygripper
                
                #Action space is broken for UCSD dataset corresponding to OpenVLA's action space
                elif dataset_name == 'ucsd_pick_and_place_dataset_converted_externally_to_rlds' or dataset_name == 'usc_cloth_sim_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3]))  # xyz
                    float_action_tensors.append(np.zeros(3))
                    float_action_tensors.append(np.array(elem['action'][3])) # Gripper torque
                
                
                elif dataset_name == 'utokyo_pr2_opening_fridge_converted_externally_to_rlds' or dataset_name == 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds':
                    float_action_tensors.append(elem['action'][:7])
                
                elif dataset_name == 'utokyo_xarm_pick_and_place_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.array(elem['action'][5])) #yaw
                    float_action_tensors.append(np.array(elem['action'][4])) #pitch
                    float_action_tensors.append(np.array(elem['action'][3])) #roll
                    float_action_tensors.append(np.array(elem['action'][6])) #gripper
                
                elif dataset_name == 'stanford_mask_vit_converted_externally_to_rlds':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.zeros(2))
                    float_action_tensors.append(np.array(elem['action'][3:])) #gripper
                
                #Action space is broken for this dataset corresponding to OpenVLA's action space
                elif dataset_name == 'eth_agent_affordances':
                    float_action_tensors.append(np.array(elem['action'][:6])) #xyz, roll, pitch, yaw
                    float_action_tensors.append(np.zeros(1))
                
                elif dataset_name == 'imperialcollege_sawyer_wrist_cam':
                    float_action_tensors.append(np.array(elem['action'][:3])) #xyz
                    float_action_tensors.append(np.array(elem['action'][5])) # z rotation
                    float_action_tensors.append(np.array(elem['action'][4])) # y rotation
                    float_action_tensors.append(np.array(elem['action'][3])) # x rotation
                    float_action_tensors.append(np.array(elem['action'][6])) #gripper

                elif dataset_name == 'utokyo_xarm_bimanual_converted_externally_to_rlds':
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
                        print(float_action_tensors)

                elif dataset_name == 'bigfish':
                    # print(f"action shape: {elem['actions'].shape}")
                    # print(f"actions: {elem['actions']}")

                    # print(f"dones: {elem['dones']}")
                    # print(f"observations: {elem['observations'].shape}")
                    # print(f"rewards: {elem['rewards']}")
                    float_action_tensors.append(np.array(elem['actions'][0]))

                if float_action_tensors:
                    float_action_tensors = [np.atleast_1d(tensor) for tensor in float_action_tensors]
                    concatenated_action_float = np.concatenate(float_action_tensors, axis=0)
                    # if concatenated_action_float.shape != (7,):
                    #     raise ValueError(f"Action tensor shape is {concatenated_action_float.shape}, expected 7")
                
                else:
                    raise ValueError(f"No float action tensors found for dataset {dataset_name}")

                
                self.timestep_count += 1
                if elem.get('is_last', False) or elem.get('dones', False):
                    self.is_last_count += 1
                

                self.action_tensors.append(concatenated_action_float)
                

    def _get_action_stats(self, mask: list, discrete: list):
        if len(self.action_tensors) == 0:
            raise AttributeError("action_stats is None, it has not been populated yet")
        
        action_dim = self.action_tensors[0].shape[0]
        
        if len(mask) != action_dim:
            raise ValueError(f"Mask length {len(mask)} does not match action dimension {action_dim}")
        
        if len(discrete) != action_dim:
            raise ValueError(f"Discrete length {len(discrete)} does not match action dimension {action_dim}")

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

    
if __name__ == "__main__":
    
    dataset_statistics = {}
    
    if IS_DEV:
        # openx_train_datasets_path = '/home/locke/ManifoldRG/MultiNet/data/openx'
        openx_train_datasets_path = '/home/locke/ManifoldRG/MultiNet/data/translated'
    else:
        openx_train_datasets_path = '/mnt/disks/mount_dir/multinettranslated/openx_translated/'
    
    openx_test_datasets_path = '/mnt/disks/mount_dir/openx_test_translated/'
    openx_val_datasets_path = '/mnt/disks/mount_dir/openx_val_translated/'
    
    # multinet_v0_1_openx_datasets = ['nyu_door_opening_surprising_effectiveness', 'columbia_cairlab_pusht_real', 'conq_hose_manipulation', 'plex_robosuite', 'stanford_mask_vit_converted_externally_to_rlds', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam']
    
    # openx_datasets = [
    #     'nyu_door_opening_surprising_effectiveness', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    #     'nyu_rot_dataset_converted_externally_to_rlds',         'usc_cloth_sim_converted_externally_to_rlds',
    #     'columbia_cairlab_pusht_real',       'plex_robosuite',                                       'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    #     'conq_hose_manipulation',            'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
    #     'eth_agent_affordances',             'stanford_mask_vit_converted_externally_to_rlds',       
    #     'imperialcollege_sawyer_wrist_cam'
    # ]
    openx_datasets = ['utokyo_xarm_bimanual_converted_externally_to_rlds', 'bigfish']

    multinet_v0_2_dataset_action_decoding_strategies = {
        'utokyo_xarm_bimanual_converted_externally_to_rlds': 'manual_mapping',
        'bigfish': 'naive_dim_extension'
    }

    for openx_dataset in openx_datasets:
        
        # Check if the dataset is already in the JSON file
        if os.path.exists('dataset_statistics.json'):
            with open('dataset_statistics.json', 'r') as f:
                existing_statistics = json.load(f)
        else:
            existing_statistics = {}
        
        if openx_dataset in existing_statistics:
            print(f"Skipping {openx_dataset} as it's already in the dataset statistics.")
            continue
        
        if not IS_DEV:
            if openx_dataset in ['conq_hose_manipulation', 'plex_robosuite', 'stanford_mask_vit_converted_externally_to_rlds', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds']:
                train_shard_files = os.listdir(os.path.join(openx_train_datasets_path, openx_dataset))
                sorted_train_shard_files = sorted(train_shard_files, key=lambda x: int(x.split('_')[-1]))
                train_tfds_shards = [os.path.join(openx_train_datasets_path, openx_dataset, f) 
                            for f in sorted_train_shard_files]
                
                val_shard_files = os.listdir(os.path.join(openx_val_datasets_path, openx_dataset))
                sorted_val_shard_files = sorted(val_shard_files, key=lambda x: int(x.split('_')[-1]))
                val_tfds_shards = [os.path.join(openx_val_datasets_path, openx_dataset, f) 
                            for f in sorted_val_shard_files]
                tfds_shards = train_tfds_shards + val_tfds_shards
            
            elif openx_dataset in ['nyu_door_opening_surprising_effectiveness', 'columbia_cairlab_pusht_real']:
                train_shard_files = os.listdir(os.path.join(openx_train_datasets_path, openx_dataset))
                sorted_train_shard_files = sorted(train_shard_files, key=lambda x: int(x.split('_')[-1]))
                train_tfds_shards = [os.path.join(openx_train_datasets_path, openx_dataset, f) 
                            for f in sorted_train_shard_files]
                
                test_shard_files = os.listdir(os.path.join(openx_test_datasets_path, openx_dataset))
                sorted_test_shard_files = sorted(test_shard_files, key=lambda x: int(x.split('_')[-1]))
                test_tfds_shards = [os.path.join(openx_test_datasets_path, openx_dataset, f) 
                            for f in sorted_test_shard_files]
                tfds_shards = train_tfds_shards + test_tfds_shards
        else:
            if openx_dataset in ['bigfish']:
                train_shard_files = os.listdir(os.path.join(openx_train_datasets_path, openx_dataset))
                sorted_train_shard_files = sorted(train_shard_files, key=lambda x: datetime.strptime(x.split('_')[0], "%Y%m%dT%H%M%S"))
                tfds_shards = [os.path.join(openx_train_datasets_path, openx_dataset, f) 
                            for f in sorted_train_shard_files]
                # TODO: add train, test or val shard files depending on the cloud folder structures

            else:
                train_shard_files = os.listdir(os.path.join(openx_train_datasets_path, openx_dataset))
                sorted_train_shard_files = sorted(train_shard_files, key=lambda x: int(x.split('_')[-1]))
                train_tfds_shards = [os.path.join(openx_train_datasets_path, openx_dataset, f) 
                            for f in sorted_train_shard_files]
                tfds_shards = train_tfds_shards

        openxobj = OpenXDataset(tfds_shards)
        openxobj._process_shards(openx_dataset)

        discrete_datasets = ['bigfish']

        action_dim = len(openxobj.action_tensors[0])
        mask = [True] * (action_dim - 1) + [False]
        discrete = [True] * action_dim if openx_dataset in discrete_datasets else [False] * action_dim

        action_stats = openxobj._get_action_stats(discrete=discrete, mask=mask)

        dataset_statistics[openx_dataset] = {
            'action': action_stats,
            'proprio': openxobj._get_proprio_stats(),
            'num_transitions': openxobj.timestep_count,
            'num_trajectories': openxobj.is_last_count,
            'action_decoding_strategy': multinet_v0_2_dataset_action_decoding_strategies.get(openx_dataset, 'manual_mapping')
        }

        existing_statistics.update(dataset_statistics)

        # Dump the updated dataset_statistics to a JSON file
        with open('dataset_statistics.json', 'w') as f:
            json.dump(existing_statistics, f, indent=4)
