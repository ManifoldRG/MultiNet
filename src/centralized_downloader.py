from argparse import ArgumentParser
import tensorflow_datasets as tfds
import torch
import datasets
from datasets import load_dataset, get_dataset_config_names
import os
import torchrl
from torchrl.data.datasets import VD4RLExperienceReplay
from collections.abc import Sequence
from absl import app
import loco_mujoco
import gymnasium as gym
import functools
import multiprocessing as mp
import shutil
import typing as tp
import tensorflow as tf
import requests
from tqdm import tqdm
import ssl
import certifi
import requests
from urllib.parse import urlparse, unquote
from google.cloud import storage

os.environ['CURL_CA_BUNDLE'] = ''

#List of datasets in v0 MultiNet
multinetv0list = ['obelics', 'coyo_700m', 'ms_coco_captions', 'conceptual_captions', 'a_okvqa', 'vqa_v2', 'datacomp', 'finewebedu', 'dm_lab_rlu', 'dm_control_suite_rlu', 'atari', 'baby_ai', 'mujoco', 'vd4rl', 'metaworld', 'procgen', 'language_table', 'openx', 'locomuojoco']

def build_arg_parser() -> ArgumentParser:

    parser = ArgumentParser(description=f'Download the dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Mention the dataset in MultiNet that needs to be translated. Different datasets are: 'obelics', 'coyo_700m', 'ms_coco_captions', 'conceptual_captions', 'a_okvqa', 'vqa_v2', 'datacomp', 'finewebedu', 'dm_lab_rlu', 'dm_control_suite_rlu', 'atari', 'baby_ai', 'mujoco', 'vd4rl', 'meta_world', 'procgen', 'language_table', 'openx', 'locomuojoco' ")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide the path to store the downloaded dataset")
    return parser


## Control dataset download modules

# RL unplugged
def rlu(dataset_name: str, output_dir: str):

    if dataset_name == 'dm_lab_rlu':
        #GCP storage bucket details
        bucket_name = "rl_unplugged"
        source_folders = ['dmlab/explore_object_rewards_few', 'dmlab/explore_object_rewards_many', 'dmlab/rooms_select_nonmatching_object', 'dmlab/rooms_watermaze', 'dmlab/seekavoid_arena_01']
        
        for source_folder in source_folders:
            destination_folder = os.path.join(output_dir, dataset_name)
            print('Downloading')
            # Initialize the Google Cloud Storage client
            storage_client = storage.Client.create_anonymous_client()
            # Get the bucket
            bucket = storage_client.bucket(bucket_name)
            # List all blobs in the source folder
            blobs = bucket.list_blobs(prefix=source_folder)

            # Download each blob
            for blob in blobs:
                # Skip if the blob is a folder (ends with '/')
                if blob.name.endswith('/'):
                    continue
                # Create the local file path
                local_path = os.path.join(destination_folder, blob.name[len(source_folder):].lstrip('/'))
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                # Download the blob
                blob.download_to_filename(local_path)
                print(f"Downloaded {blob.name} to {local_path}")
            
        print('Successfully downloaded DM Lab from RL unplugged')
        return
    
    elif dataset_name == 'dm_control_suite_rlu':
        #GCP storage bucket details
        bucket_name = "rl_unplugged"
        source_folder = "dm_control_suite"
        destination_folder = os.path.join(output_dir, dataset_name)
        print('Downloading')
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client.create_anonymous_client()
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        # List all blobs in the source folder
        blobs = bucket.list_blobs(prefix=source_folder)
        print(blobs)
        # Download each blob
        for blob in blobs:
            # Skip if the blob is a folder (ends with '/')
            if blob.name.endswith('/'):
                continue
            # Create the local file path
            local_path = os.path.join(destination_folder, blob.name[len(source_folder):].lstrip('/'))
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Download the blob
            blob.download_to_filename(local_path)
            print(f"Downloaded {blob.name} to {local_path}")
        
        print('Successfully downloaded DM Control Suite from RL unplugged')
        return

# JAT    
def jat(dataset_name: str, output_dir: str):

    print('Downloading')
    config_names = get_dataset_config_names('jat-project/jat-dataset')
    
    try:
        specific_configs = [config for config in config_names if config.startswith(dataset_name)]#"metaworld, mujoco", "atari", "babyai"
        print(f"Found {len(specific_configs)} {dataset_name} configurations in {dataset_name}.")
    except:
        print('Choose one of - atari, mujoco, metaworld, or babyai for one of the JAT datasets')
        return


    for config in specific_configs:
        print(f"Downloading configuration: {config}")
        try:
            dataset = load_dataset('jat-project/jat-dataset',config)
            os.makedirs(os.path.join(output_dir,config))
            dataset.save_to_disk(os.path.join(output_dir,config))
            print(f"Successfully downloaded {config}")
        except Exception as e:
            print(f"Error downloading {config}: {str(e)}")

    print(f"Finished downloading all available {dataset_name} configurations from jat-project/jat-dataset.")

#V-D4RL
def vd4rl(dataset_name: str, output_dir: str):

    #V-D4RL tasks
    for dataset_id in VD4RLExperienceReplay.available_datasets:
        if 'expert' in dataset_id:
            try:
                print(f'Downloading {dataset_id}...')
                vd4rldataset = VD4RLExperienceReplay(dataset_id = dataset_id, batch_size=64)
                batch_ctr=0
                os.makedirs(os.path.join(output_dir,dataset_name), exist_ok=True)
                for batch in vd4rldataset:
                    torch.save(batch, output_dir+'/'+dataset_name+'/'+'_'.join(dataset_id.split('/'))+str(batch_ctr)+'.pt')
                    batch_ctr+=1
            except:
                print(f'Error downloading {dataset_id}')
                return

    print('Successfully downloaded all V-D4RL expert datasets')


#LocoMuJoCo
def locomujoco(dataset_name: str, output_dir: str):

    #Download all locomujoco perfect datasets through the library
    os.system("loco-mujoco-download-perfect")
    #Locomujoco task names
    locomujoco_tasks = loco_mujoco.get_all_task_names()
    for task in locomujoco_tasks:
        if task.split('.')[-1] == 'perfect':
            try:
                print(f'Downloading {task}...')
                env = gym.make('LocoMujoco', env_name=task)
                dict_env = env.create_dataset()
                os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
                torch.save(dict_env, os.path.join(os.path.join(output_dir, dataset_name),task+'.pt'))
            except:
                print(f"Error downloading {task}")
                return
    
    print('Successfully downloaded all LocoMuJoCo expert datasets')
    
#Procgen
def procgen(dataset_name: str, output_dir: str):

    #Procgen env names
    BASE_URL = "https://dl.fbaipublicfiles.com/DGRL/1M/expert/"
    ENV_NAMES = [
        "bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun",
        "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze",
        "miner", "ninja", "plunder", "starpilot"
    ]

    for env_name in ENV_NAMES:
        file_name = f"{env_name}.tar.xz"
        url = BASE_URL + file_name
        file_path = os.path.join(output_dir, file_name)
        
        print(f"Downloading {file_name}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, 'wb') as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

        print(f"Extracting {file_name}...")
        shutil.unpack_archive(file_path, output_dir)
        os.remove(file_path)

    print("Successfully downloaded and extracted Procgen expert data")

#Language Table
def language_table(dataset_name: str, output_dir: str):

    #Language table dataset names
    dataset_directories = {

    'language_table': 'gs://gresearch/robotics/language_table',
    'language_table_sim': 'gs://gresearch/robotics/language_table_sim',
    'language_table_blocktoblock_sim': 'gs://gresearch/robotics/language_table_blocktoblock_sim',
    'language_table_blocktoblock_4block_sim': 'gs://gresearch/robotics/language_table_blocktoblock_4block_sim',
    'language_table_blocktoblock_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblock_oracle_sim',
    'language_table_blocktoblockrelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim',
    'language_table_blocktoabsolute_oracle_sim': 'gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim',
    'language_table_blocktorelative_oracle_sim': 'gs://gresearch/robotics/language_table_blocktorelative_oracle_sim',
    'language_table_separate_oracle_sim': 'gs://gresearch/robotics/language_table_separate_oracle_sim',

    }

    #Change the dataset name and version to load another dataset
    dataset_path = os.path.join(dataset_directories['language_table'], '0.0.1')
    try:
        print('Downloading...')
        builder = tfds.builder_from_directory(dataset_path)
        ds = builder.as_dataset(split='train')
        ds = ds.flat_map(lambda x: x['steps'])
        os.makedirs(os.path.join(output_dir,dataset_name))
        tf.data.Dataset.save(ds, os.path.join(output_dir,dataset_name))
    except:
        print(f'Error while downloading {dataset_name}...')
        return

    print('Successfully downloaded Language Table dataset')

#OpenX-Embodiment
def openx(dataset_name: str, output_dir: str):

    #OpenX datasets
    DATASETS = [
    'fractal20220817_data',
    'kuka',
    'bridge',
    'taco_play',
    'jaco_play',
    'berkeley_cable_routing',
    'roboturk',
    'nyu_door_opening_surprising_effectiveness',
    'viola',
    'berkeley_autolab_ur5',
    'toto',
    'columbia_cairlab_pusht_real',
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
    'nyu_rot_dataset_converted_externally_to_rlds',
    'stanford_hydra_dataset_converted_externally_to_rlds',
    'austin_buds_dataset_converted_externally_to_rlds',
    'nyu_franka_play_dataset_converted_externally_to_rlds',
    'maniskill_dataset_converted_externally_to_rlds',
    'cmu_franka_exploration_dataset_converted_externally_to_rlds',
    'ucsd_kitchen_dataset_converted_externally_to_rlds',
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    'austin_sailor_dataset_converted_externally_to_rlds',
    'austin_sirius_dataset_converted_externally_to_rlds',
    'bc_z',
    'usc_cloth_sim_converted_externally_to_rlds',
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
    'utokyo_saytap_converted_externally_to_rlds',
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
    'utokyo_xarm_bimanual_converted_externally_to_rlds',
    'robo_net',
    'berkeley_mvp_converted_externally_to_rlds',
    'berkeley_rpt_converted_externally_to_rlds',
    'kaist_nonprehensile_converted_externally_to_rlds',
    'stanford_mask_vit_converted_externally_to_rlds',
    'tokyo_u_lsmo_converted_externally_to_rlds',
    'dlr_sara_pour_converted_externally_to_rlds',
    'dlr_sara_grid_clamp_converted_externally_to_rlds',
    'dlr_edan_shared_control_converted_externally_to_rlds',
    'asu_table_top_converted_externally_to_rlds',
    'stanford_robocook_converted_externally_to_rlds',
    'eth_agent_affordances',
    'imperialcollege_sawyer_wrist_cam',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
    'uiuc_d3field',
    'utaustin_mutex',
    'berkeley_fanuc_manipulation',
    'cmu_play_fusion',
    'cmu_stretch',
    'berkeley_gnm_recon',
    'berkeley_gnm_cory_hall',
    'berkeley_gnm_sac_son'
    ]

    for ds in DATASETS:
        if ds == 'robo_net':
            version = '1.0.0'
        else:
            version = '0.1.0'
        file_path = f'gs://gresearch/robotics/{ds}/{version}'
    
        try:
            if os.path.isdir(ds) == False:
                print(f'Downloading {ds}...')
                builder = tfds.builder_from_directory(builder_dir=file_path)
                os.makedirs(os.path.join(output_dir, ds), exist_ok=True)
                b = builder.as_dataset(split='train')
                b = b.flat_map(lambda x: x['steps'])
                tf.data.Dataset.save(b,os.path.join(output_dir, ds))
        except:
            print(f'Error while downloading {ds}')
            
    
    print('OpenX downloads complete')


## Vision-Language dataset download modules

def vislang(dataset_name: str, output_dir: str):

    #Download the specified dataset from HuggingFace
    if dataset_name == 'obelics':
        ds = load_dataset("HuggingFaceM4/OBELICS", "default")
    elif dataset_name == 'coyo_700m':
        ds = load_dataset("kakaobrain/coyo-700m")
    elif dataset_name == 'ms_coco_captions':
        ds = load_dataset("HuggingFaceM4/COCO", trust_remote_code=True)
    elif dataset_name == 'conceptual_captions':
        ds = load_dataset("google-research-datasets/conceptual_captions")
    elif dataset_name == 'a_okvqa':
        ds = load_dataset("HuggingFaceM4/A-OKVQA")
    elif dataset_name == 'vqa_v2':
        ds = load_dataset('HuggingFaceM4/VQAv2', trust_remote_code=True)
    elif dataset_name == 'datacomp':
        ds = load_dataset('mlfoundations/datacomp_1b')
    elif dataset_name == 'finewebedu':
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "default")


    print('Successfully downloaded.')
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    ds.save_to_disk(os.path.join(output_dir, dataset_name))

def download_datasets(dataset_name: str, output_dir: str):
    
    if dataset_name in ['obelics','coyo_700m', 'ms_coco_captions', 'conceptual_captions', 'a_okvqa', 'vqa_v2', 'datacomp', 'finewebedu']:
        vislang(dataset_name, output_dir)
    elif dataset_name == 'dm_lab_rlu' or dataset_name == 'dm_control_suite_rlu':
        rlu(dataset_name, output_dir)
    elif dataset_name == 'atari' or dataset_name == 'mujoco' or dataset_name == 'babyai' or dataset_name == 'metaworld':
        jat(dataset_name, output_dir)
    elif dataset_name == 'vd4rl':
        vd4rl(dataset_name, output_dir)
    elif dataset_name == 'procgen':
        procgen(dataset_name, output_dir)
    elif dataset_name == 'locomujoco':
        locomujoco(dataset_name, output_dir)
    elif dataset_name == 'language_table':
        language_table(dataset_name, output_dir)
    elif dataset_name == 'openx':
        openx(dataset_name, output_dir)
    else:
        print('Enter the name of a dataset in Multinet v0')

    return


if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    download_datasets(args.dataset_name, args.output_dir) 
