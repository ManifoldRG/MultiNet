import os
from centralized_translation import rlu, jat, torchrlds, procgen, custom_shard_func
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

    # Path to the pickle file to store the element specs
    pickle_path = os.path.join(output_dir, 'element_specs.pkl')

    # Walk through the folder structure
    for root, dirs, files in os.walk(dataset_path):
        
        shard_files = [f for f in files]
        sorted_shard_files = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))

        if sorted_shard_files:
            print(f"Translating shards in {root}")
            for idx, shard_file in enumerate(sorted_shard_files):
                
                shard_path = os.path.join(root, shard_file)
                #print(shard_path)
                if dataset_name=='dm_lab_rlu' or dataset_name=='dm_control_suite_rlu':
                    translated_ds = rlu(shard_path, limit_schema)
                    tf.data.Dataset.save(translated_ds,os.path.join(output_dir, 'translated_shard_'+str(idx)), shard_func = custom_shard_func)
                    print('Translated and stored')
                elif dataset_name=='baby_ai' or dataset_name=='ale_atari' or dataset_name=='mujoco' or dataset_name=='meta_world':
                    if 'test' not in root and 'train' not in root:
                        translated_ds = jat(dataset_name, root, hf_test_data, limit_schema) #Enter parent folder of a given jat dataset
                        tf.data.Dataset.save(translated_ds,os.path.join(output_dir, 'translated_shard_'+str(idx)), shard_func = custom_shard_func)
                        print('Translated and stored')
                elif dataset_name=='vd4rl' or dataset_name=='locomujoco' or dataset_name=='language_table' or dataset_name=='openx':
                    if os.path.exists(os.path.join(os.path.join(output_dir, root.split('/')[-1]), 'translated_shard_'+str(idx))):
                        print(f'Skipping because shard {idx} is already translated and saved')
                        continue
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

                    tf.data.Dataset.save(translated_ds,os.path.join(os.path.join(output_dir, root.split('/')[-1])), shard_func = custom_shard_func)
                    print('Translated and stored')
                elif dataset_name=='procgen':
                    translated_ds = procgen(root, limit_schema) #Enter parent folder of a given procgen dataset
                    tf.data.Dataset.save(translated_ds,os.path.join(output_dir, 'translated_shard_'+str(idx)), shard_func = custom_shard_func)
                    print('Translated and stored')
        
    

if __name__ == "__main__":
    
    parser = build_arg_parser()
    args = parser.parse_args()
    translated_ds = translate_shards(args.dataset_name, args.dataset_path, args.hf_test_data, args.limit_schema, args.output_dir)
    #Test the translated 
    '''finalds = tf.data.Dataset.load('<path to shard you want to test>')
    print(len(finalds))
    for elem in finalds:
        print(elem)'''