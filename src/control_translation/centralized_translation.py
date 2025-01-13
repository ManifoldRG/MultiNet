import os
import tensorflow as tf
import torch
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
import datasets
import tensorflow_datasets as tfds

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

#Process the nested dictionary to create a nested dictionary where the values of each key and subkey are lists of lists containing the timestep values for each episode.
# This is required to convert it to TFDS
def process_dict(d, key_path, result_dict):
 # Parameters:
    # d: input dictionary to process
    # key_path: list tracking the current path in the nested structure
    # result_dict: dictionary where results are accumulated
    for key, value in d.items():
        # Iterate through each key-value pair in the input dictionary 'd'
        str_key = str(key)
        current_path = key_path + [str_key]
        # Create new path by adding current key to existing path
        # Example: if key_path is ['level1'] and key is 'level2'
        # current_path becomes ['level1', 'level2']
        
        if isinstance(value, dict):
            # Create nested dict structure if it doesn't exist
            temp_dict = result_dict
            for k in current_path[:-1]:
                # Iterate through all keys except the last one
                if k not in temp_dict:
                    temp_dict[k] = {}
                temp_dict = temp_dict[k]
            if current_path[-1] not in temp_dict:
                temp_dict[current_path[-1]] = {}
                
            process_dict(value, current_path, result_dict)
        else:
            # Navigate to correct nested level
            temp_dict = result_dict
            for k in current_path[:-1]:
                if k not in temp_dict:
                    temp_dict[k] = {}
                temp_dict = temp_dict[k]
            
            # Initialize list if needed and append value
            if current_path[-1] not in temp_dict:
                temp_dict[current_path[-1]] = []
            temp_dict[current_path[-1]].append(value)




# RL unplugged datasets translation
def rlu_tfds(dataset_path: str, limit_schema: bool, output_dir, dataset_name):

    try:
        tfds_name = str(dataset_path.replace('../', ''))
        if tfds_name.endswith('/'):
            tfds_name = tfds_name[:-1]
    except:
        raise ValueError('Enter the correct path to a DM Lab or DM Control Suite file downloaded from RL unplugged using TFDS. It should be in the format rlu_control_suite/cartpole_swingup')

    #print(tfds_name)

    #Data dir needs to be the parent directory of the dataset path
    data_dir = dataset_path.split('/')[0] if '/' in dataset_path else '.'
    #print(data_dir)
    #Load the TFDS dataset
    loaded_dataset = tfds.load(
    tfds_name,
    split='train',
    data_dir=data_dir,
    download=False
    )

    count=0
    
    # Process dataset episode by episode, and save each episode as a separate file. This is done to avoid memory issues as some of these datasets are large
    for ele in loaded_dataset:
        #Creating a dictionary where the values of each key are a list of lists each containing the timestep values for each episode
        rlu_dict= defaultdict(list)  
        episode_dict = {}
        for key, value in ele.items():
            if key == 'steps':
                
                # Handle steps variant dataset while maintaining hierarchy
                step_dict = {}
                for step_idx, step in enumerate(value):
                    step_dict[step_idx] = {}
                    for step_key, step_value in step.items():
                        # Handle pixel observations if present
                        if step_key == 'observations' and 'pixels' in step_value:
                            # Convert Image object to tensor with correct shape and dtype
                            pixels = tf.convert_to_tensor(step_value['pixels'])
                            pixels = tf.ensure_shape(pixels, [72, 96, 3])
                            pixels = tf.cast(pixels, tf.uint8)
                            step_value['pixels'] = pixels
                        step_dict[step_idx][step_key] = step_value
                episode_dict['steps'] = step_dict
            else:
                episode_dict[key] = value
        
        #Count denotes the episode number
        rlu_dict[count] = episode_dict
        count+=1
    
        rlu_torch_list = []
        for key, value in rlu_dict.items():
            rlu_torch_list.append(value)

        rlu_torch_list_dict = {}
        for ele in rlu_torch_list:
            process_dict(ele, [], rlu_torch_list_dict)

        rlu_tfds = tf.data.Dataset.from_tensor_slices(rlu_torch_list_dict)

        mod_file_path = dataset_path.replace('../', '')
        path_to_translated = os.path.join(dataset_name+'_translated/', mod_file_path)
        tf.data.Dataset.save(rlu_tfds, os.path.join(output_dir, path_to_translated)+'translated_episode_'+str(count),shard_func=custom_shard_func)
    #return rlu_tfds


def rlu(dataset_path: str, limit_schema: bool):

    dm_lab_dict = defaultdict(list)

    #Load the RL unplugged dataset as a TFRecord Dataset
    try:
        raw_dataset = tf.data.TFRecordDataset(dataset_path)#, compression_type = 'GZIP')
    except:
        print('Enter the correct path to a DM Lab or DM Control Suite file downloaded from RL unplugged')
        return None

    #Access the values in the dataset based on the feature type --only accessing 5 episodes with this code (remove .take() if entire ds needs to be downloaded)
    for raw_record in raw_dataset.take(5):

        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        for key, feature in example.features.feature.items():

            if feature.HasField('int64_list'):
                print(key)
                values = tf.convert_to_tensor(feature.int64_list.value)
                dm_lab_dict[key].append(values)

            elif feature.HasField('float_list'):
                print(key)
                values = tf.convert_to_tensor(feature.float_list.value)
                dm_lab_dict[key].append(values)
            
            elif feature.HasField('bytes_list'):
                print(key)
                values = []
                for step in feature.bytes_list.value:
                    try:
                        # Try decoding as image first
                        values.append(tf.image.decode_jpeg(step, channels=3))
                    except:
                        # If not an image, just decode the bytes
                        values.append(tf.io.decode_raw(step, tf.uint8))
                
                print(values)
                values = tf.convert_to_tensor(values)
                dm_lab_dict[key].append(values)
            else:
                print(f"Unsupported feature type: {key}")
        
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
def jat(dataset_name: str, dataset_path: str, hf_test_data: bool, limit_schema: bool):

    jat_tfds = None
    try: 
        #Load HF dataset from local path
        jat_hf = datasets.load_from_disk(dataset_path)
    
    except:
        print('Enter the correct path to a JAT HF dataset')
        return jat_tfds

    print('Translating...')

    #Translate HF DatasetDict to TFDS
    #Baby AI has a slightly different data structure compared to other JAT datasets, which requires pre-processing before conversion to TFDS
    if dataset_name == 'baby_ai':

        text_observations = [example['text_observations'] for example in jat_hf['train']]
        discrete_observations = [example['discrete_observations'] for example in jat_hf['train']]
        discrete_actions = [example['discrete_actions'] for example in jat_hf['train']]
        rewards = [example['rewards'] for example in jat_hf['train']]

        # Create the dataset
        jat_tfds_train = tf.data.Dataset.from_tensor_slices({
            'text_observations': tf.ragged.constant(text_observations, dtype=tf.string),
            'discrete_observations': tf.ragged.constant(discrete_observations, dtype=tf.int64),
            'discrete_actions': tf.ragged.constant(discrete_actions, dtype=tf.int64),
            'rewards': tf.ragged.constant(rewards, dtype=tf.float32)
        })

    else:

        jat_tfds_train = jat_hf['train'].to_tf_dataset(columns=list(jat_hf['train'][0].keys()))

    if limit_schema:
        print('The JAT datasets only contain observations, actions, and rewards. No further trimming will be done. Observations and Actions are stored as Continuous/Discrete Observations and Actions depending on the task')

    if hf_test_data:

        if dataset_name == 'baby_ai':

            text_observations = [example['text_observations'] for example in jat_hf['test']]
            discrete_observations = [example['discrete_observations'] for example in jat_hf['test']]
            discrete_actions = [example['discrete_actions'] for example in jat_hf['test']]
            rewards = [example['rewards'] for example in jat_hf['test']]

            # Create the dataset
            jat_tfds_test = tf.data.Dataset.from_tensor_slices({
                'text_observations': tf.ragged.constant(text_observations, dtype=tf.string),
                'discrete_observations': tf.ragged.constant(discrete_observations, dtype=tf.int64),
                'discrete_actions': tf.ragged.constant(discrete_actions, dtype=tf.int64),
                'rewards': tf.ragged.constant(rewards, dtype=tf.float32)
            })
        
        else:
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
        if dataset_name == 'vd4rl':
            trl_torch_trimmed = {}
            trl_torch_trimmed['observations'] = trl_torch['pixels']
            trl_torch_trimmed['actions'] = trl_torch['action']
            trl_torch_trimmed['rewards'] = trl_torch['next']['reward']
            print('Translating...')
            trl_torch_trimmed_tfds = tf.data.Dataset.from_tensor_slices(trl_torch_trimmed)
            return trl_torch_trimmed_tfds
        elif dataset_name == 'openx':
            trl_torch_trimmed = {}
            trl_torch_trimmed['observations'] = trl_torch['observation']
            trl_torch_trimmed['actions'] = trl_torch['action']
            trl_torch_trimmed['rewards'] = trl_torch['reward']
            print('Translating...')
            trl_torch_trimmed_tfds = tf.data.Dataset.from_tensor_slices(trl_torch_trimmed)
            return trl_torch_trimmed_tfds


    #Translate TorchRL dataset to TFDS
    print('Translating...')
    if type(trl_torch) is not dict:
        trl_tfds = tf.data.Dataset.from_tensor_slices(trl_torch.to_dict())  
    else:
        
        #Convert it to a dict of lists where each key has a list of values for all the timesteps in a given episode
        trl_torch_list = []
        for key, value in trl_torch.items():
            trl_torch_list.append(value)
            #print(trl_torch_list)

        trl_torch_list_dict = {}

        for ele in trl_torch_list:
            process_dict(ele, [], trl_torch_list_dict)
            
        trl_tfds = tf.data.Dataset.from_tensor_slices(trl_torch_list_dict)

    return trl_tfds

def locomujoco(dataset_path: str, limit_schema: bool):

    try:
        locomujoco_np = np.load(dataset_path, allow_pickle=True).items()
    except:
        print('Enter the correct path to a LocoMuJoCo file')
        return None

    locomujoco_np_dict = {}
    for key, value in locomujoco_np:
        locomujoco_np_dict[key] = value
    
    if limit_schema:
        del locomujoco_np_dict['last']
        del locomujoco_np_dict['absorbing']

    locomujoco_tfds_dict = tf.data.Dataset.from_tensor_slices(locomujoco_np_dict)
    return locomujoco_tfds_dict



def procgen(dataset_path: str, limit_schema: bool):

    #Taken from https://github.com/ManifoldRG/MultiNet/blob/main/src/control_translation/procgen/convert2tfds.py


    print('Translating...')
    #Iterate through Procgen dataset folder and translate file by file. Returns a consolidated mega TFDS containing translated versions of all the files.
    #for path in tqdm(os.listdir(dataset_path), total=len(os.listdir(dataset_path)), unit='file'):
        
    try:
        procgen_np = np.load(dataset_path, allow_pickle=True).item()
    except:
        print('Enter the correct path to a Procgen file in Numpy format')
        return None
    
    #Translate TorchRL dataset to TFDS
    if limit_schema:
        del procgen_np['dones']
    
    # Add zero padding at start for non-observation fields as there is one extra observation field in the dataset
    for key, value in procgen_np.items():
        # Get length of first value to determine padding size
        first_val = value[0]
        # Handle both array and scalar values
        '''if isinstance(first_val, np.ndarray):
            pad_size = len(first_val)
        else:
            pad_size = 1'''
        
        
        # Add zero padding at start for non-observation fields
        if key != 'observations':
            #print(key)
            #print(value)
            if isinstance(first_val, np.ndarray):
                # Create zero array with same shape as first value
                zero_pad = np.zeros_like(first_val)
            else:
                # Create zero with same type as first value 
                zero_pad = type(first_val)(0)
            
            if isinstance(first_val, np.ndarray):
                value = np.insert(value, 0, zero_pad, axis=0)
            else:
                value = np.insert(value, 0, zero_pad)
            print(value)
        
        procgen_np[key] = value

        #print(key)
        #print(len(procgen_np[key]))

    procgen_tfds_dict = tf.data.Dataset.from_tensor_slices(procgen_np)
    #procgen_tfds = tf.data.Dataset.zip(procgen_dict)

    '''if mega_dataset is None:
        mega_dataset = procgen_tfds_dict    
    else:
        mega_dataset = mega_dataset.concatenate(procgen_tfds_dict)
    
    save_counter+=1

    #Testing on 500 files
    if save_counter==50:
        break

    return mega_dataset'''
    return procgen_tfds_dict


#Decides the translation module to be called based on the dataset
def categorize_datasets(dataset_name: str, dataset_path: str, hf_test_data: bool, limit_schema: bool):

    try:
        if dataset_name=='dm_lab_rlu' or dataset_name=='dm_control_suite_rlu':
            translated_ds = rlu(dataset_path, limit_schema)
            return translated_ds
        elif dataset_name=='baby_ai' or dataset_name=='ale_atari' or dataset_name=='mujoco' or dataset_name=='meta_world':
            translated_ds = jat(dataset_name, dataset_path, hf_test_data, limit_schema)
            return translated_ds
        elif dataset_name=='vd4rl' or dataset_name=='openx':
            translated_ds = torchrlds(dataset_path, dataset_name, limit_schema)
            return translated_ds
        elif dataset_name=='procgen':
            translated_ds = procgen(dataset_path, limit_schema)
            return translated_ds
        elif dataset_name=='locomujoco':
            translated_ds = locomujoco(dataset_path, limit_schema)
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
            '''finalds = tf.data.Dataset.load('<output_dir>/<name of translated train dataset>')
            for elem in finalds:
                print(elem)
                break

            finalds = tf.data.Dataset.load('<output_dir>/<name of translated test dataset>')
            for elem in finalds:
                print(elem)
                break'''
        else:
            #Saving the translated dataset in output_dir as a single shard
            tf.data.Dataset.save(translated_ds,os.path.join(args.output_dir,args.dataset_name+'_translated'), shard_func = custom_shard_func)
            print('Translated and stored')
            
            #Testing
            '''finalds = tf.data.Dataset.load('<output_dir>/<name of translated train dataset>')
            print(len(finalds))
            for elem in finalds:
               print(elem)
               break'''
            