import os
from centralized_translation import rlu, rlu_tfds, jat, torchrlds, procgen, custom_shard_func, locomujoco
from argparse import ArgumentParser
import tensorflow as tf
import gc
import psutil
import pickle

def build_arg_parser() -> ArgumentParser:

    parser = ArgumentParser(description=f'Translate the dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Mention the dataset in MultiNet that needs to be translated. Different datasets are: 'dm_lab_rlu', 'dm_control_suite_rlu', 'ale_atari', 'baby_ai', 'mujoco', 'vd4rl', 'meta_world', 'procgen', 'language_table', 'openx', 'locomuojoco' ")
    parser.add_argument("--dataset_path", type=str, required=True, help="Provide the path to the specified dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Provide the path to store the translated dataset")
    parser.add_argument("--limit_schema", type=bool, default=False, help="Set to True if schema needs to be trimmed to [observations, actions, rewards]")
    parser.add_argument("--hf_test_data", type=bool, default=False, help="Set to True if test split from Huggingface JAT datasets needs to be returned along with train data")
    return parser

def translate_shards(dataset_name, dataset_path, hf_test_data, limit_schema, output_dir):
    
    if dataset_name=='dm_lab_rlu' or dataset_name=='dm_control_suite_rlu':
        
        dir_path = dataset_path
        translated_ds = rlu_tfds(dir_path, limit_schema, output_dir, dataset_name)
        #mod_file_path = dir_path.replace('../', '')
        #path_to_translated = os.path.join(dataset_name+'_translated/', mod_file_path)
        #print(path_to_translated)
        #tf.data.Dataset.save(translated_ds, os.path.join(output_dir, path_to_translated),shard_func=custom_shard_func)
        print(f'Translated and stored file {output_dir}')

    elif dataset_name=='baby_ai' or dataset_name=='ale_atari' or dataset_name=='mujoco' or dataset_name=='meta_world':

        all_files = []
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            # Skip if no files or if the file is a json file
            # Only collect immediate subdirectories of dataset_path
            if dirpath == dataset_path:
                for dirname in dirnames:
                    all_files.append(os.path.join(dirpath, dirname))
                break
        
        #print(all_files)
        
        # Process each file
        for idx, file_path in enumerate(all_files):
            translated_ds = jat(dataset_name,file_path, hf_test_data, limit_schema)
            mod_file_path = file_path.replace('../', '')
            path_to_translated = os.path.join(dataset_name+'_translated/', mod_file_path)
            #print(path_to_translated)
            tf.data.Dataset.save(translated_ds, os.path.join(output_dir, path_to_translated),shard_func=custom_shard_func)
            print(f'Translated and stored file {file_path}')
    
    elif dataset_name=='vd4rl':
        # Get all .pt files under dataset_path
        all_files = []
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            # Only collect .pt files
            for f in filenames:
                if f.endswith('.pt'):
                    all_files.append(os.path.join(dirpath, f))
        
        #print(all_files)
        
        # Process each .pt file
        for idx, file_path in enumerate(all_files):
            translated_ds = torchrlds(file_path, dataset_name, limit_schema)
            mod_file_path = file_path.replace('../', '')
            # Remove extension if present in final component
            base, ext = os.path.splitext(mod_file_path)
            if ext:
                mod_file_path = base
            path_to_translated = os.path.join(dataset_name+'_translated/', mod_file_path)
            #print(path_to_translated)
            tf.data.Dataset.save(translated_ds, os.path.join(output_dir, path_to_translated), shard_func=custom_shard_func)
            print(f'Translated and stored file {file_path}')

    elif dataset_name == 'locomujoco':
        # Get all .npz files under dataset_path
        all_files = []
        for dirpath, dirnames, filenames in os.walk(dataset_path):
            # Only collect .npz files
            for f in filenames:
                if f.endswith('.npz'):
                    all_files.append(os.path.join(dirpath, f))
        
        #print(all_files)
        
        # Process each .npz file
        for idx, file_path in enumerate(all_files):
            translated_ds = locomujoco(file_path, limit_schema)
            mod_file_path = file_path.replace('../', '')
            # Remove extension if present in final component
            base, ext = os.path.splitext(mod_file_path)
            if ext:
                mod_file_path = base
            path_to_translated = os.path.join(dataset_name+'_translated/', mod_file_path)
            #print(path_to_translated)
            tf.data.Dataset.save(translated_ds, os.path.join(output_dir, path_to_translated), shard_func=custom_shard_func)
            print(f'Translated and stored file {file_path}')
    
    elif dataset_name == 'openx':

        #Element spec contains the original schema of the dataset
        pickle_path = os.path.join(output_dir, 'element_specs.pkl')
        shard_files = []
        for root, dirs, files in os.walk(dataset_path):
            shard_files = [f for f in files]

        sorted_shard_files = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))

        #print(sorted_shard_files)

        if sorted_shard_files:
            print(f"Translating shards in {root}")
            for idx, shard_file in enumerate(sorted_shard_files):

                shard_path = os.path.join(root, shard_file)
                translated_ds = torchrlds(shard_path, dataset_name, limit_schema)
                
                # Extract element_spec and store in pickle
                # Load existing specs if file exists, otherwise create new dict
                if os.path.exists(pickle_path):
                    with open(pickle_path, 'rb') as f:
                        element_specs = pickle.load(f)
                else:
                    element_specs = {}
                
                # If dataset_name not in specs, add it
                if root.split('/')[-1] not in element_specs:
                    element_specs[root.split('/')[-1]] = translated_ds.element_spec
                    
                    # Save updated specs
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(element_specs, f)
                    
                    print(f"Element spec for {root.split('/')[-1]} added to {pickle_path}")
                else:
                    print(f"Element spec for {root.split('/')[-1]} already exists in {pickle_path}")
                modified_shard_path = shard_path.replace('../', '')
                path_to_translated = os.path.join(dataset_name+'_translated/', modified_shard_path) 
                #print(path_to_translated)
                tf.data.Dataset.save(translated_ds, os.path.join(output_dir, path_to_translated), shard_func=custom_shard_func)
                print(f'Translated and stored file {shard_path}')
            
            print('Translated and stored')
    
    elif dataset_name == 'procgen':
        # Get all .npy files in root directory
        npy_files = []
        for dirpath, dirs, filenames in os.walk(dataset_path):
            for f in filenames:
                if f.endswith('.npy'):
                    npy_files.append(os.path.join(dirpath, f))

        #print(npy_files)
        if npy_files:
            print(f"Translating .npy files in {dataset_path}")
            for npy_file in npy_files:
                translated_ds = procgen(npy_file, limit_schema)
                # Get filename without extension for saving
                base, ext = os.path.splitext(npy_file)
                if ext:
                    mod_file_path = base
                mod_file_path = mod_file_path.replace('../', '')
                mod_file_path = os.path.join(output_dir, mod_file_path)

                path_to_translated = os.path.join(dataset_name+'_translated/', mod_file_path)
                #print(path_to_translated)
                # Save translated dataset
                tf.data.Dataset.save(translated_ds, os.path.join(output_dir, path_to_translated), shard_func=custom_shard_func)
                print(f'Translated and stored file {npy_file}')

            print('Successfully translated all .npy files')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    

if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    translated_ds = translate_shards(args.dataset_name, args.dataset_path, args.hf_test_data, args.limit_schema, args.output_dir)
    #Test the translated 
    '''finalds = tf.data.Dataset.load('<path to shard you want to test>')
    print(len(finalds))
    for elem in finalds:
        print(elem)'''