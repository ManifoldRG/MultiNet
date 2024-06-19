import gc
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import dataset_builders
import torch
import numpy as np
import torchrl
from torchrl.data.datasets import VD4RLExperienceReplay
from argparse import ArgumentParser
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import datasets
from datasets import load_dataset, Dataset

#List of control datasets in v0 MultiNet
multinetv0list = ['dm_lab_rlu', 'dm_control_suite_rlu', 'ale_atari', 'baby_ai', 'mujoco', 'vd4rl', 'meta_world', 'procgen', 'language_table', 'openx', 'locomuojoco']


def build_arg_parser() -> ArgumentParser:

    parser = ArgumentParser(description=f'Translate the dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Mention the dataset in MultiNet that needs to be translated. Different datasets are: 'dm_lab_rlu', 'dm_control_suite_rlu', 'ale_atari', 'baby_ai', 'mujoco', 'vd4rl', 'meta_world', 'procgen', 'language_table', 'openx', 'locomuojoco' ")
    parser.add_argument("--dataset_path", type=str, required=True, help="Provide the path to the specified dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide the path to store the translated dataset")
    parser.add_argument("--limit_schema", type=bool, default=False, help="Set to True if schema needs to be trimmed to [observations, actions, rewards]")
    parser.add_argument("--hf_test_data", type=bool, default=False, help="Set to True if test split from Huggingface JAT datasets needs to be returned along with train data")
    return parser

# RL unplugged datasets translation
def rlu(dataset_path: str, limit_schema: bool):

    dm_lab_dict = defaultdict(list)

    #Load the RL unplugged dataset as a TFRecord Dataset
    try:
        raw_dataset = tf.data.TFRecordDataset(dataset_path, compression_type = 'GZIP')
    except:
        print('Enter the correct path to a DM Lab or DM Control Suite file downloaded from RL unplugged')
        return None

    #Access the values in the dataset based on the feature type --only accessing 5 episodes with this code (remove .take() if entire ds needs to be downloaded)
    for raw_record in raw_dataset.take(5):

        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        for key, feature in example.features.feature.items():

            if feature.HasField('int64_list'):
                values = tf.convert_to_tensor(feature.int64_list.value)
                dm_lab_dict[key].append(values)

            elif feature.HasField('float_list'):
                values = tf.convert_to_tensor(feature.float_list.value)
                dm_lab_dict[key].append(values)
            
            elif feature.HasField('bytes_list'):
                values = []
                for step in feature.bytes_list.value:
                    values.append(tf.image.decode_jpeg(step, channels=3))
                
                values = tf.convert_to_tensor(values)
                dm_lab_dict[key].append(values)
        
    #Convert data dict to TFDS
    dm_lab_dict = {k: tf.convert_to_tensor(v) for k, v in dm_lab_dict.items()}

    # Trim the data if limit_schema flag is set during code execution
    if limit_schema:
        dm_lab_dict_trimmed = {}
        if 'actions' in dm_lab_dict.keys():
            dm_lab_dict_trimmed['actions'] = dm_lab_dict['actions']
        else:
            dm_lab_dict_trimmed['actions'] = dm_lab_dict['action']
        dm_lab_dict_trimmed['observations'] = {k:v for k,v in dm_lab_dict.items() if 'observation' in k}
        if 'rewards' in dm_lab_dict.keys():
            dm_lab_dict_trimmed['rewards'] = dm_lab_dict['rewards']
        else:
            dm_lab_dict_trimmed['rewards'] = dm_lab_dict['reward']
        
        print('Translating...')
        dm_lab_dict_trimmed_tfds = tf.data.Dataset.from_tensor_slices(dm_lab_dict_trimmed)
        return dm_lab_dict_trimmed_tfds

    print('Translating...')
    dm_lab_tfds = tf.data.Dataset.from_tensor_slices(dm_lab_dict)
    return dm_lab_tfds
    

# JAT datasets translation
def jat(dataset_path: str, hf_test_data: bool, limit_schema: bool):

    jat_tfds = None
    try: 
        #Load HF dataset from local path
        jat_hf = datasets.load_from_disk(dataset_path)
    
    except:
        print('Enter the correct path to a JAT HF dataset')
        return jat_tfds

    print('Translating...')
    #Translate HF DatasetDict to TFDS
    jat_tfds_train = jat_hf['train'].to_tf_dataset(columns=list(jat_hf['train'][0].keys()))

    if limit_schema:
        print('The JAT datasets only contain observations, actions, and rewards. No further trimming will be done.')

    if hf_test_data:
        
        jat_tfds_test = jat_hf['test'].to_tf_dataset(columns=list(jat_hf['test'][0].keys()))
    
        #Return tuple of translated train and test splits
        jat_tfds = (jat_tfds_train, jat_tfds_test)
        return (jat_tfds_train,jat_tfds)

    else:
        return jat_tfds_train


#TorchRL datasets translation
def torchrlds(dataset_path: str, dataset_name, limit_schema: bool):

    trl_tfds = None
    try:
        #Load PyTorch dataset from local path
        trl_torch = torch.load(dataset_path)
    except:
        print('Enter the correct path to the Torch dataset')
        return trl_tfds  
    
    # Trim the data if limit_schema flag is set during code execution
    if limit_schema:
        if dataset_name == 'locomujoco':
            trl_torch_trimmed = {}
            trl_torch_trimmed['observations'] = trl_torch['states']
            trl_torch_trimmed['actions'] = trl_torch['actions']
            trl_torch_trimmed['rewards'] = trl_torch['rewards']
            print('Translating...')
            trl_torch_trimmed_tfds = tf.data.Dataset.from_tensor_slices(trl_torch_trimmed)
            return trl_torch_trimmed_tfds
        elif dataset_name == 'vd4rl':
            trl_torch_trimmed = {}
            trl_torch_trimmed['observations'] = trl_torch['pixels']
            trl_torch_trimmed['actions'] = trl_torch['action']
            trl_torch_trimmed['rewards'] = trl_torch['next']['reward']
            print('Translating...')
            trl_torch_trimmed_tfds = tf.data.Dataset.from_tensor_slices(trl_torch_trimmed)
            return trl_torch_trimmed_tfds


    #Translate TorchRL dataset to TFDS
    print('Translating...')
    trl_tfds = tf.data.Dataset.from_tensor_slices(trl_torch)  
    return trl_tfds

def procgen(dataset_path: str, limit_schema: bool):

    #Taken from https://github.com/ManifoldRG/MultiNet/blob/main/src/control_translation/procgen/convert2tfds.py

    mega_dataset = None
    save_counter=0

    print('Translating...')
    #Iterate through Procgen dataset folder and translate file by file. Returns a consolidated mega TFDS containing translated versions of all the files.
    for path in tqdm(os.listdir(dataset_path), total=len(os.listdir(dataset_path)), unit='file'):
        
        try:
            procgen_np = np.load(os.path.join(dataset_path, path), allow_pickle=True).item()
        except:
            print('Enter the correct path to a Procgen file in Numpy format')
            return None
        
        #Translate TorchRL dataset to TFDS
        if limit_schema:
            del procgen_np['dones']

        procgen_dict = {key: tf.data.Dataset.from_tensor_slices(procgen_np[key]) for key in procgen_np.keys()}
        procgen_tfds = tf.data.Dataset.zip(procgen_dict)

        if mega_dataset is None:
            mega_dataset = procgen_tfds
        else:
            mega_dataset = mega_dataset.concatenate(procgen_tfds)
        
        save_counter+=1

        #Testing on 500 files
        if save_counter==500:
            break

    return mega_dataset


#Decides the translation module to be called based on the dataset
def categorize_datasets(dataset_name: str, dataset_path: str, hf_test_data: bool, limit_schema: bool):

    try:
        if dataset_name=='dm_lab_rlu' or dataset_name=='dm_control_suite_rlu':
            translated_ds = rlu(dataset_path, limit_schema)
            return translated_ds
        elif dataset_name=='baby_ai' or dataset_name=='ale_atari' or dataset_name=='mujoco' or dataset_name=='meta_world':
            translated_ds = jat(dataset_path, hf_test_data, limit_schema)
            return translated_ds
        elif dataset_name=='vd4rl' or dataset_name=='locomujoco':
            translated_ds = torchrlds(dataset_path, dataset_name, limit_schema)
            return translated_ds
        elif dataset_name=='procgen':
            translated_ds = procgen(dataset_path, limit_schema)
            return translated_ds
        
        else:
            raise ValueError('Enter a dataset in the current version of MultiNet')
    except ValueError as e:
        print(f"Error:{e}")

#Shard function to ensure the translated TFDS data is saved as one shard
def custom_shard_func(element):
    return np.int64(0)

if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    translated_ds = categorize_datasets(args.dataset_name, args.dataset_path, args.hf_test_data, args.limit_schema)

    if translated_ds is not None:

        if isinstance(translated_ds,tuple):
            translated_ds_train, translated_ds_test = translated_ds
            tf.data.Dataset.save(translated_ds_train,os.path.join(args.output_dir,args.dataset_name+'_translated_train'), shard_func = custom_shard_func)
            print('Translated train data stored')
            tf.data.Dataset.save(translated_ds_train,os.path.join(args.output_dir,args.dataset_name+'_translated_test'), shard_func = custom_shard_func)
            print('Translated test data stored')
            
            #Testing
            finalds = tf.data.Dataset.load('/Users/guruprasad/Desktop/MetarchCode/MultiNet/mujoco_translated_train')
            '''for elem in finalds:
                print(elem)
                break

            finalds = tf.data.Dataset.load('/Users/guruprasad/Desktop/MetarchCode/MultiNet/mujoco_translated_test')
            for elem in finalds:
                print(elem)
                break'''
        else:
            #Saving the translated dataset in output_dir as a single shard
            tf.data.Dataset.save(translated_ds,os.path.join(args.output_dir,args.dataset_name+'_translated'), shard_func = custom_shard_func)
            print('Translated and stored')
            
            #Testing
            '''finalds = tf.data.Dataset.load('/Users/guruprasad/Desktop/MetarchCode/MultiNet/vd4rl_translated')
            print(len(finalds))
            for elem in finalds:
               print(elem)
               break'''
               
                
        
    
