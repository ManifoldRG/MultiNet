import skimage.measure
from scipy.ndimage import sobel
import torch
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
import pandas as pd
import imageio
from PIL import Image

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.v0_2results.results_analysis import load_results

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from definitions.openx import OpenXDefinitions
from definitions.procgen import ProcGenDefinitions


DATASET_CONFIG_NAME = 'dataset_statistics_config.yaml'

PROFILING_DATASETS = ProcGenDefinitions.DESCRIPTIONS.keys() # | OpenXDefinitions.DESCRIPTIONS.keys()

ENVIRONMENT = os.getenv('ENVIRONMENT', 'prod')


logging.basicConfig(
    format='%(asctime)s [%(levelname)-4s] | %(filename)s:%(lineno)d | %(message)s',
)
logger = logging.getLogger(__name__)
if ENVIRONMENT == 'dev':
    logger.setLevel(logging.DEBUG)


class DatasetActionStatisticsCalculator:
    def __init__(self, tfds_shards: list[str], dataset_name: str):
        self.tfds_shards = tfds_shards
        self.image_tensors = []
        self.is_last_count = 0
        self.timestep_count = 0
        self.dataset_name = dataset_name
        self.shannon_entropies = []
        self.delentropies = []
        self.current_batch_images = []
        self.batch_count = 0
        
    def process_shards(self):
        for shard_idx, shard in enumerate(self.tfds_shards):
            dataset = tf.data.Dataset.load(shard)
            # Process the input data for each element in the shard
            for elem_idx, elem in enumerate(dataset):
                image_tensors = []

                if self.dataset_name == 'nyu_door_opening_surprising_effectiveness' or self.dataset_name == 'columbia_cairlab_pusht_real':
                    image_tensors.append(np.array(elem['image']))
                    image_tensors.append(np.array(elem['action']['rotation_delta']))
                    image_tensors.append(np.array(elem['action']['gripper_closedness_action']))
                
                elif self.dataset_name == 'nyu_rot_dataset_converted_externally_to_rlds' or self.dataset_name == 'conq_hose_manipulation' or self.dataset_name == 'plex_robosuite': 
                    image_tensors.append(elem['action']) #xyzrpygripper
                
                # Action space is broken for UCSD dataset corresponding to OpenVLA's action space
                elif self.dataset_name == 'ucsd_pick_and_place_dataset_converted_externally_to_rlds' or self.dataset_name == 'usc_cloth_sim_converted_externally_to_rlds':
                    image_tensors.append(np.array(elem['action'][:3]))  # xyz
                    image_tensors.append(np.zeros(3))
                    image_tensors.append(np.array(elem['action'][3])) # Gripper torque
                
                elif self.dataset_name == 'utokyo_pr2_opening_fridge_converted_externally_to_rlds' or self.dataset_name == 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds':
                    image_tensors.append(elem['action'][:7])
                
                elif self.dataset_name == 'utokyo_xarm_pick_and_place_converted_externally_to_rlds':
                    image_tensors.append(np.array(elem['action'][:3])) #xyz
                    image_tensors.append(np.array(elem['action'][5])) #yaw
                    image_tensors.append(np.array(elem['action'][4])) #pitch
                    image_tensors.append(np.array(elem['action'][3])) #roll
                    image_tensors.append(np.array(elem['action'][6])) #gripper
                
                elif self.dataset_name == 'stanford_mask_vit_converted_externally_to_rlds':
                    image_tensors.append(np.array(elem['action'][:3])) #xyz
                    image_tensors.append(np.zeros(2))
                    image_tensors.append(np.array(elem['action'][3:])) #gripper
                
                # Action space is broken for this dataset corresponding to OpenVLA's action space
                elif self.dataset_name == 'eth_agent_affordances':
                    image_tensors.append(np.array(elem['action'][:6])) #xyz, roll, pitch, yaw
                    image_tensors.append(np.zeros(1))
                
                elif self.dataset_name == 'imperialcollege_sawyer_wrist_cam':
                    image_tensors.append(np.array(elem['action'][:3])) #xyz
                    image_tensors.append(np.array(elem['action'][5])) # z rotation
                    image_tensors.append(np.array(elem['action'][4])) # y rotation
                    image_tensors.append(np.array(elem['action'][3])) # x rotation
                    image_tensors.append(np.array(elem['action'][6])) #gripper

                elif self.dataset_name == 'utokyo_xarm_bimanual_converted_externally_to_rlds':
                    image_tensors.append(np.array(elem['action'][:3])) #xyz
                    image_tensors.append(np.array(elem['action'][5])) #roll
                    image_tensors.append(np.array(elem['action'][4])) #pitch
                    image_tensors.append(np.array(elem['action'][3])) #yaw
                    image_tensors.append(np.array(elem['action'][6])) #gripper
                    image_tensors.append(np.array(elem['action'][7:10])) #xyz
                    image_tensors.append(np.array(elem['action'][12])) #roll
                    image_tensors.append(np.array(elem['action'][11])) #pitch
                    image_tensors.append(np.array(elem['action'][10])) #yaw
                    image_tensors.append(np.array(elem['action'][13])) #gripper
                    if len(image_tensors) == 14:
                        logger.debug(f"image_tensors: {image_tensors}")

                elif self.dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
                    self.shannon_entropies.append(self.shannon_entropy(torch.from_numpy(np.array(elem['observations']))))
                    self.delentropies.append(self.delentropy(torch.from_numpy(np.array(elem['observations']))))
                    
                    # Add image to batch for GIF generation
                    # img = np.array(elem['observations'])
                    # if img.shape[0] == 3:  # If channels first
                    #     img = np.moveaxis(img, 0, 2)
                    # self.current_batch_images.append(Image.fromarray(img.astype(np.uint8)))
                    # self.batch_count += 1

                    # Create a 3-second GIF (30 frames at 0.1s duration)
                    # if self.batch_count == 30:
                    #     output_dir = "gif_outputs"
                    #     os.makedirs(output_dir, exist_ok=True)
                    #     output_path = os.path.join(output_dir, f"{self.dataset_name}_sequence.gif")
                    #     imageio.mimsave(output_path, self.current_batch_images, duration=0.1)
                    #     logger.info(f"Saved 3-second GIF for {self.dataset_name} to {output_path}")
                    #     self.current_batch_images = []
                    #     self.batch_count = 0
                else:
                    logger.error(f"Undefined action tensor mapping logic for {self.dataset_name}")
                    break

                self.timestep_count += 1
                if elem.get('is_last', False) or elem.get('dones', False):
                    self.is_last_count += 1
            
            # Save any remaining frames as a GIF before moving to next shard
            # if self.current_batch_images and self.dataset_name in ProcGenDefinitions.DESCRIPTIONS.keys():
            #     output_dir = "gif_outputs"
            #     os.makedirs(output_dir, exist_ok=True)
            #     output_path = os.path.join(output_dir, f"{self.dataset_name}_remaining_frames.gif")
            #     imageio.mimsave(output_path, self.current_batch_images, duration=0.1)
            #     logger.info(f"Saved remaining frames GIF for {self.dataset_name} ({len(self.current_batch_images)} frames) to {output_path}")
            #     self.current_batch_images = []
            #     self.batch_count = 0
                
            if shard_idx < 2: 
                break

    def get_dataset_mean_shannon_entropy(self):
        dataset_mean_entropy = np.mean(self.shannon_entropies)

        return dataset_mean_entropy
    
    def shannon_entropy(self, tensor: torch.Tensor, bins=256):
        # Convert tensor to grayscale if RGB
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = torch.mean(tensor, dim=0)  # RGB to grayscale

        # Flatten and compute histogram
        hist = torch.histc(tensor, bins=bins, min=0, max=1.0 if tensor.max() <=1 else 255)
        prob = hist / hist.sum()
        prob = prob[prob > 0]  # Remove zero probabilities
        shannon_entropy = -torch.sum(prob * torch.log2(prob)).item()
        # print(f"Shannon entropy: {shannon_entropy}")
        return shannon_entropy
    
    def delentropy(self, tensor: torch.Tensor, bins=256):
        img = tensor.cpu().numpy().squeeze()
        # Compute gradients
        fx = sobel(img, axis=0)
        fy = sobel(img, axis=1)
        magnitudes = np.sqrt(fx**2 + fy**2)
        orientations = np.arctan2(fy, fx)  # [-π, π]/
        
        # 2D histogram of magnitude vs orientation
        hist, xedges, yedges = np.histogram2d(
            magnitudes.flatten(), 
            orientations.flatten(), 
            bins=bins,
            range=[[0, magnitudes.max()], [-np.pi, np.pi]]
        )
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        delentropy = -np.sum(prob * np.log2(prob)).item()
        # print(f"Delentropy: {delentropy}")
        return delentropy

    def get_dataset_mean_delentropy(self):
        dataset_mean_delentropy = np.mean(self.delentropies)

        return dataset_mean_delentropy
               
def load_dataset_path_config():
    config_path = "/home/locke/ManifoldRG/MultiNet/src/eval/profiling/openvla/data/dataset_statistics_config.yaml"
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



if __name__ == "__main__":
    # try:
    #     dataset_statistics = load_dataset_statistics(DATASET_STATISTICS_FILE)
    # except Exception as e:
    #     raise Exception(f"Error loading dataset statistics: {str(e)}")

    dataset_path_config = load_dataset_path_config()
    added_dataset_stats_counter = 0
    results_dir = "src/v0_2results"
    df = pd.DataFrame(columns=['dataset', 'mean_shannon_entropy', 'mean_delentropy', 
                               'gpt4o_macro_recall', 'gpt4_1_macro_recall',
                               'openvla_macro_recall', 'pi0_base_macro_recall',
                               'pi0_fast_macro_recall'])
    
    results = load_results(results_dir)


    for dataset in PROFILING_DATASETS:
        # First populate the dataset name for this row
        df.loc[dataset, 'dataset'] = dataset

        # for model in results.keys():
        #     if model in ['pi0_base', 'pi0_fast']:
        #         if dataset == 'bigfish':
        #             df.loc[dataset, f'{model}_macro_recall'] = results[model]['bigfish']['bigfish']['macro_recall']
        #         else:
        #             df.loc[dataset, f'{model}_macro_recall'] = results[model][dataset][dataset]['macro_recall']
        #     elif model in ['gpt4o', 'gpt4_1']:
        #         df.loc[dataset, f'{model}_macro_recall'] = results[model][dataset][dataset]['macro_recall']
        #     elif model == 'openvla':
        #         if dataset == 'bigfish':
        #             df.loc[dataset, f'{model}_macro_recall'] = results[model]['bigfish']['bigfish']['macro_recall']
        #         else:
        #             df.loc[dataset, f'{model}_macro_recall'] = results[model][dataset]['macro_recall']
        #     else:
        #         raise ValueError(f"Unknown model: {model}")
        #     print(df.head())

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

            print(f"Processing {dataset}")
            dataset_mean_shannon_entropy = stats_calculator.get_dataset_mean_shannon_entropy()
            dataset_mean_delentropy = stats_calculator.get_dataset_mean_delentropy()
            df.loc[dataset, f'mean_shannon_entropy'] = dataset_mean_shannon_entropy
            df.loc[dataset, f'mean_delentropy'] = dataset_mean_delentropy

            # If we have processed shards successfully
            if len(stats_calculator.image_tensors) > 0:
                # dataset_statistics[dataset] = calculate_dataset_statistics(stats_calculator, dataset)
                added_dataset_stats_counter += 1
            else:
                logger.warning(f"No action tensors were gathered for dataset {dataset}")
        
            # save_dataset_statistics(dataset_statistics, DATASET_STATISTICS_FILE)

        except Exception as e:
            logger.exception(f"Error processing dataset {dataset}: {str(e)}")
            raise

    df.to_csv(f"{results_dir}/dataset_analysis_results.csv", index=False)