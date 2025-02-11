from argparse import ArgumentParser
import tensorflow_datasets as tfds
import torch
from datasets import load_dataset, get_dataset_config_names
import os
from torchrl.data.datasets import VD4RLExperienceReplay
import shutil
import tensorflow as tf
import requests
from tqdm import tqdm
import requests
from google.cloud import storage
import gc
import psutil
#List of datasets in v0 MultiNet
multinetv0list = ['obelics', 'coyo_700m', 'ms_coco_captions', 'conceptual_captions', 'a_okvqa', 'vqa_v2', 'datacomp', 'finewebedu', 'dm_lab_rlu', 'dm_control_suite_rlu', 'atari', 'baby_ai', 'mujoco', 'vd4rl', 'metaworld', 'procgen', 'language_table', 'openx', 'locomuojoco']

def build_arg_parser() -> ArgumentParser:

    parser = ArgumentParser(description=f'Download the dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Mention the dataset in MultiNet that needs to be translated. Different datasets are: 'obelics', 'coyo_700m', 'ms_coco_captions', 'conceptual_captions', 'a_okvqa', 'vqa_v2', 'datacomp', 'finewebedu', 'dm_lab_rlu', 'dm_control_suite_rlu', 'atari', 'baby_ai', 'mujoco', 'vd4rl', 'meta_world', 'procgen', 'language_table', 'openx', 'locomuojoco' ")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide the path to store the downloaded dataset")
    return parser


## Control dataset download modules

#RL unplugged through TFDS -- dataset is formatted more intuitively and easier to understand and work with. Documentation is available at https://www.tensorflow.org/datasets/catalog/rlu_control_suite
#Make sure you have sufficient memory to download the dataset. During the TFDS load function, the entire dataset is loaded into memory. Check the sizes at https://www.tensorflow.org/datasets/catalog/rlu_control_suite
def rlu_tfds(dataset_name: str, output_dir: str):

    rlu_dmcs_dataset_list = ['rlu_control_suite/cartpole_swingup', 'rlu_control_suite/cheetah_run', 'rlu_control_suite/finger_turn_hard', 'rlu_control_suite/fish_swim', 'rlu_control_suite/humanoid_run', 'rlu_control_suite/manipulator_insert_ball', 'rlu_control_suite/manipulator_insert_peg', 'rlu_control_suite/walker_stand', 'rlu_control_suite/walker_walk']

    for dataset in rlu_dmcs_dataset_list:

        # Download the dataset and retrieve dataset information
        dataset_name, info = tfds.load(
            dataset,
            split='train',
            data_dir=output_dir,
            #batch_size = 2,
            download=True,
            with_info=True
        )

        print("Dataset downloaded and stored at:", output_dir)
        print("Dataset information:", info)


# RL unplugged
def rlu(dataset_name: str, output_dir: str):

    if dataset_name == 'dm_lab_rlu':
        #GCP storage bucket details
        bucket_name = "rl_unplugged"
        source_folders = ['dmlab/explore_object_rewards_few', 'dmlab/explore_object_rewards_many', 'dmlab/rooms_select_nonmatching_object', 'dmlab/rooms_watermaze', 'dmlab/seekavoid_arena_01']
        
        for source_folder in source_folders:
            destination_folder = os.path.join(output_dir, source_folder)
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
        raise ValueError('Choose one of - atari, mujoco, metaworld, or babyai for one of the JAT datasets')


    for config in specific_configs:
        print(f"Downloading configuration: {config}")
        try:
            dataset = load_dataset('jat-project/jat-dataset',config)
            os.makedirs(os.path.join(output_dir,config))
            dataset.save_to_disk(os.path.join(output_dir,config))
            print(f"Successfully downloaded {config}")
        except Exception as e:
            raise ValueError(f"Error downloading {config}: {str(e)}")

    print(f"Finished downloading all available {dataset_name} configurations from jat-project/jat-dataset.")
    return

#V-D4RL
def vd4rl(dataset_name: str, output_dir: str):

    #V-D4RL tasks
    for dataset_id in VD4RLExperienceReplay.available_datasets:
        if '/expert' in dataset_id:
            try:
                
                print(f'Downloading {dataset_id}...')
                os.makedirs(os.path.join(output_dir,dataset_name), exist_ok=True)
                vd4rldataset = VD4RLExperienceReplay(dataset_id = dataset_id, batch_size = None)
                batch_size = len(vd4rldataset)
                # Iterate through the dataset and save batches
                for i in range(0, len(vd4rldataset), batch_size):
                    end = min(i + batch_size, len(vd4rldataset))
                    batch = vd4rldataset[i:end]
                    file_path = os.path.join(output_dir+'/'+dataset_name+'/'+'_'.join(dataset_id.split('/'))+str(i)+'.pt')
                    torch.save(batch, file_path)
                    print(f"Downloaded and saved batch {i//batch_size} to {file_path}")
            except:
                raise ValueError(f'Error downloading {dataset_id}')

    print('Successfully downloaded all V-D4RL expert datasets')
    return


#LocoMuJoCo
def locomujoco(dataset_name: str, output_dir: str):

    #Download all locomujoco perfect datasets
    # Check if loco-mujoco repo exists, if not clone it
    if not os.path.exists('loco-mujoco') and not os.path.exists('loco-mujoco/.git'):
        print("Cloning loco-mujoco repository...")
        os.system("git clone https://github.com/robfiras/loco-mujoco.git")
        print("Successfully cloned loco-mujoco")
    else:
        print("loco-mujoco repository already exists")
    try:
        os.system(f"loco-mujoco-download-perfect")
    except:
        raise ValueError("Error downloading LocoMuJoCo datasets")
    print('Successfully downloaded all LocoMuJoCo expert datasets to loco-mujoco/loco_mujoco/datasets/')
    return
    
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
            unit_divisor=512,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=512):
                size = file.write(data)
                progress_bar.update(size)

        print(f"Extracting {file_name}...")
        shutil.unpack_archive(file_path, output_dir)
        os.remove(file_path)

    print("Successfully downloaded and extracted Procgen expert data")
    return

# OpenX
    
#Saving datasets as big as these to disk is a very RAM-intensive process. This function optimizes for this purpose by sharding and freeing up memory after saving to disk    
def shard_and_save(ds, dataset_name: str, output_dir: str, start_from_shard: int, shard_size: int):

    for i, shard in enumerate(ds.batch(shard_size), start=start_from_shard):

        if os.path.exists(os.path.join(output_dir, dataset_name,'shard_'+str(i))) == True:
            print(f'Shard {i} of {dataset_name} already downloaded')
            continue
            
        # Check RAM usage
        ram_usage = psutil.virtual_memory().percent
        #If RAM usage is more than 90% free up memory and restart the sharding+saving procedure from the same shard
        if ram_usage > 90:
            print(f"\nRAM usage is {ram_usage}%. Restarting from shard {i}...\n")
            # Clean up resources after pausing the sharding+saving procedure
            del shard
            del ds
            gc.collect()
            return i
    
        #Saving with torch instead of tf as tf has a memory leakage issue that leads to the program crashing before completion
        
        #torch.save(shard, f"{os.path.join(output_dir, dataset_name)}/shard_{i}")
                    
        #del shard
        #gc.collect()

        shard = tf.data.Dataset.from_tensor_slices(shard)
        flattened_dataset = shard.flat_map(lambda x: x['steps'])
        dataset_dict = {i: item for i, item in enumerate(flattened_dataset.as_numpy_iterator())}
        #print(dataset_dict)
        torch.save(dataset_dict, f"{os.path.join(output_dir, dataset_name)}/shard_{i}")

        # Print current RAM usage
        print(f"Processed shard {i}. Current RAM usage: {ram_usage}%")
    
    return None

#OpenX-Embodiment
def openx(dataset_name: str, output_dir: str):
    
    #OpenX datasets
    OPENX_DATASETS = [
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
    'language_table',
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
    'berkeley_gnm_sac_son',
    'robot_vqa',
    'droid',
    'conq_hose_manipulation',
    'dobbe',
    'fmb',
    'io_ai_tech',
    'mimic_play',
    'aloha_mobile',
    'robo_set',
    'tidybot',
    'vima_converted_externally_to_rlds',
    'spoc',
    'plex_robosuite',
    'furniture_bench_dataset_converted_externally_to_rlds',
    'qut_dexterous_manipulation',
    'cmu_playing_with_food'
    ]

    datasets = OPENX_DATASETS
    
    if dataset_name != 'openx' and dataset_name in datasets:
        datasets = [dataset_name]
        
    #Shard size to save the dataset to disk
    shard_size = 1

    for ds in datasets:
        
        # Try all version combinations
        versions = ['0.0.0', '0.0.1', '0.1.0', '0.1.1', '1.0.0', '1.0.1', '1.1.0', '1.1.1']
        for v in versions:
            try:
                version = v
                # If this version works, break out of loop
                temp_file_path = f'gs://gresearch/robotics/{ds}/{version}'
                builder = tfds.builder_from_directory(builder_dir=temp_file_path)
                break
            except:
                continue
        else:
            # If no version worked, raise an error
            raise ValueError(f'No version found for {ds}')
        
        file_path = f'gs://gresearch/robotics/{ds}/{version}'
    
        try:
            print(f'Downloading {ds}...')
            builder = tfds.builder_from_directory(builder_dir=file_path)
            try:
                b = builder.as_dataset(split='test')
                split_name = 'test'
                print('Downloading test split')
            except:
                try:
                    b = builder.as_dataset(split='val')
                    split_name = 'val'
                    print('Downloading val split')
                except:
                    b = builder.as_dataset(split='train')
                    split_name = 'train'
                    print('Downloading train split')
            #b = b.flat_map(lambda x: x['steps'])
            os.makedirs(os.path.join(output_dir, ds+'_'+split_name), exist_ok=True)
            
            shard_func_catch=0
            while(1):
                if shard_func_catch is not None:
                    shard_func_catch = shard_and_save(b,ds+'_'+split_name, output_dir, shard_func_catch, shard_size)
                else:
                    break

        except:
            raise ValueError(f'Error while downloading {ds}')
            
    
    print('OpenX downloads complete')
    return


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


    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    ds.save_to_disk(os.path.join(output_dir, dataset_name))
    print('Successfully downloaded and saved')
    return

def download_datasets(dataset_name: str, output_dir: str):
    
    if dataset_name in ['obelics','coyo_700m', 'ms_coco_captions', 'conceptual_captions', 'a_okvqa', 'vqa_v2', 'datacomp', 'finewebedu']:
        vislang(dataset_name, output_dir)
    elif dataset_name == 'dm_lab_rlu' or dataset_name == 'dm_control_suite_rlu':
        rlu(dataset_name, output_dir)
    elif dataset_name == 'dm_lab_rlu_tfds' or dataset_name == 'dm_control_suite_rlu_tfds':
        rlu_tfds(dataset_name, output_dir)
    elif dataset_name == 'atari' or dataset_name == 'mujoco' or dataset_name == 'babyai' or dataset_name == 'metaworld':
        jat(dataset_name, output_dir)
    elif dataset_name == 'vd4rl':
        vd4rl(dataset_name, output_dir)
    elif dataset_name == 'procgen':
        procgen(dataset_name, output_dir)
    elif dataset_name == 'locomujoco':
        locomujoco(dataset_name, output_dir)
    elif dataset_name == 'openx' or dataset_name in OPENX_DATASETS:
        openx(dataset_name, output_dir)
    else:
        print('Enter the name of a dataset in Multinet v0')

    return


if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    download_datasets(args.dataset_name, args.output_dir) 
