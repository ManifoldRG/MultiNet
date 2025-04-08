from collections import defaultdict
import os
import sys
import argparse
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Create test splits for datasets')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Name of the dataset to create test splits for')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Path to the output directory')
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Path to the base directory')
    
    return parser.parse_args()

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


def main(dataset_name: str, output_dir: str, base_dir: str):
    
    if dataset_name in ['meta_world_translated', 'mujoco_translated']:
        # Get base directory for this dataset
        dataset_path = f'{base_dir}'
        
        # Create output directory if it doesn't exist
        output_base = f'{output_dir}/{dataset_name}'
        os.makedirs(output_base, exist_ok=True)
        
        # Iterate through subdirectories
        for dirs in os.listdir(dataset_path):

            if 'test' in dirs:
                test_dir = os.path.join(dataset_path, dirs)
                    
                # Keep the test directory name in the output path
                output_dir = os.path.join(output_base, dirs)
                os.makedirs(output_dir, exist_ok=True)
                
                # Copy test directory contents
                os.system(f'cp -r {test_dir}/* {output_dir}/')
                print(f'Copied test data from {test_dir} to {output_dir}')
    
    elif dataset_name in ['ale_atari_translated', 'baby_ai_translated']:
        # Get base directory for this dataset
        dataset_path = f'{base_dir}'
        
        # Create output directory if it doesn't exist
        output_base = f'{output_dir}/{dataset_name}'
        os.makedirs(output_base, exist_ok=True)
        
        # Iterate through subdirectories
        for dirs in os.listdir(dataset_path):
            for subdir in dirs:
                #print(subdir)
                subdir_path = os.path.join(dataset_path, subdir)
                #print(subdir_path)
                # Skip if not a directory
                if not os.path.isdir(subdir_path):
                    print(f'{subdir_path} is not a directory')
                    continue
                
                # Look for test directories in all subdirectories
                for dir in os.listdir(subdir_path):

                    if dir == 'test':
                        test_dir = os.path.join(subdir_path, dir)
                        
                        # Create relative path structure from base_dir
                        output_dir = os.path.join(output_base, subdir, dir)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Copy all episode files from test directory
                        for episode_file in os.listdir(test_dir):

                            src_file = os.path.join(test_dir, episode_file)
                            
                            os.system(f'cp -r {src_file} {output_dir}')
                            print(f'Copied {src_file} to {output_dir}')
                        print(f'Copied test episodes from {test_dir} to {output_dir}')
    
    elif dataset_name == 'procgen_translated':

        output_base = f'{output_dir}/{dataset_name}'
        os.makedirs(output_base, exist_ok=True)

        for dirs in os.listdir(f'{base_dir}'):
            #Go to the directory that contains the different procgen subdatasets
            dir_path = os.path.join(base_dir, dirs)

            print(dir_path)

            # First sort the files based on naming. Example of a file name in a subdataset of procgen - 20230329T100909_5858_450_91_6.00
            #sorted_files = sorted(os.listdir(dir_path))

            # Each file is an episode. We can split 20% of the episodes as test set.
            #num_test_episodes = int(len(sorted_files) * 0.2)
            #test_episodes = sorted_files[:num_test_episodes]
            #train_episodes = sorted_files[num_test_episodes:]

            # Get all episode files
            all_files = os.listdir(dir_path)

            # Calculate number of test episodes (20%)
            #num_test_episodes = min(int(len(all_files) * 0.2), 1000)
            num_test_episodes = int(len(all_files) * 0.2)

            # Randomly sample test episodes
            test_episodes = np.random.choice(all_files, num_test_episodes, replace=False)

            # Create test directory
            test_dir = os.path.join(output_base, dirs, 'test_final')
            os.makedirs(test_dir, exist_ok=True)

            # Save list of test episodes to file
            test_episodes_file = os.path.join(output_base, dirs, 'test_episodes.txt')
            with open(test_episodes_file, 'w') as f:
                for episode in test_episodes:
                    f.write(f'{episode}\n')
            print(f'Saved list of {len(test_episodes)} test episodes to {test_episodes_file}')

            # Copy the test episodes to the test directory
            for episode in test_episodes:
                src_file = os.path.join(dir_path, episode)
                os.system(f'cp -r {src_file} {test_dir}')
                print(f'Copied {src_file} to {test_dir}')
        
    
    elif dataset_name == 'locomujoco_translated':

        # Get the main categories (humanoids/quadrupeds)
        categories = os.listdir(f'{base_dir}')
        output_base = f'{output_dir}/{dataset_name}'
        os.makedirs(output_base, exist_ok=True)

        for category_dir in categories:
            #print(category_dir)
            category_path = os.path.join(base_dir, category_dir)
                
            # Get the perfect directory
            perfect_path = os.path.join(category_path, 'perfect')
            
            if not os.path.exists(perfect_path):
                print(f'Perfect directory not found in {category_path}')
                continue
            
            # Get the subdatasets within the perfect directory
            intermediate_leaf_datasets = os.listdir(perfect_path)

            for intermediate_leaf_dataset in intermediate_leaf_datasets:
                intermediate_leaf_dataset_path = os.path.join(perfect_path, intermediate_leaf_dataset)

                # Get all leaf datasets under perfect/
                leaf_datasets = os.listdir(intermediate_leaf_dataset_path)
                
                for leaf_dataset in leaf_datasets:
                    leaf_path = os.path.join(intermediate_leaf_dataset_path, leaf_dataset)
                    
                    print(f'Creating test splits for {leaf_dataset}')
                    # Get all episodes in the leaf dataset
                    dataset = tf.data.Dataset.load(leaf_path)
                    episodes_dict = []
                    temp_episode = defaultdict(list)
                    for timestep in dataset:
                        for key, value in timestep.items():
                            temp_episode[key].append(value)
                        if timestep['last'] == 1.0:
                            temp_episode = dict(temp_episode)
                            episodes_dict.append(temp_episode)
                            temp_episode = defaultdict(list)
                    
                    #print(episodes_dict)
                    
                    # Split the episodes into test and train  - the first 20% of the episodes are test episodes
                    num_test_episodes = int(len(episodes_dict) * 0.2)
                    test_episodes = episodes_dict[:num_test_episodes]
                    
                    test_episodes_list_dict = {}    
                    for episode in test_episodes:
                        process_dict(episode, [], test_episodes_list_dict)

                    #Save test episodes as tf file
                    test_dir = os.path.join(output_base, category_dir, 'perfect', leaf_dataset, 'test')
                    test_episodes_tf = tf.data.Dataset.from_tensor_slices(test_episodes_list_dict)
                    tf.data.Dataset.save(test_episodes_tf, test_dir)

                    print(f'Created and saved test splits for {leaf_dataset}')


    elif dataset_name == 'dm_control_suite_rlu_tfds_translated':

        dataset_path = f'{base_dir}'
        for dirs in os.listdir(dataset_path):

            output_base = f'{output_dir}/{dataset_name}'
            os.makedirs(os.path.join(output_base, dirs, 'test'), exist_ok=True)

            # Get all episode files and sort them by episode number
            episode_files = os.listdir(os.path.join(dataset_path, dirs))
            episode_files.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by the number at the end

            # Calculate number of test episodes (20%)
            num_test_episodes = int(len(episode_files) * 0.2)
            
            # Select first 20% as test episodes
            test_episode_files = episode_files[:num_test_episodes]

            # Copy test episodes to test directory
            for episode_file in test_episode_files:
                src_path = os.path.join(dataset_path, dirs, episode_file)
                dst_path = os.path.join(output_base, dirs, 'test', episode_file)
                os.system(f'cp -r {src_path} {dst_path}')

            print(f'Created test split for {dirs} ({num_test_episodes} episodes)')

    elif dataset_name == 'dm_lab_rlu_translated':

        dataset_path = f'{base_dir}'
        for dirs in os.listdir(dataset_path):

            output_base = f'{output_dir}/{dataset_name}'
            os.makedirs(os.path.join(output_base, dirs, 'test'), exist_ok=True)

            # Get all directory names and sort them by tfrecord number
            episode_dirs = os.listdir(os.path.join(dataset_path, dirs))
            episode_dirs.sort(key=lambda x: int(x.split('-')[1]))  # Sort by the number after first hyphen

            #Calculate number of test episodes (20%)
            num_test_episodes = int(len(episode_dirs) * 0.2)

            #Select first 20% as test episodes
            test_episode_dirs = episode_dirs[:num_test_episodes]

            #Copy test episodes to test directory
            for episode_dir in test_episode_dirs:
                src_path = os.path.join(dataset_path, dirs, episode_dir)
                dst_path = os.path.join(output_base, dirs, 'test', episode_dir)
                os.system(f'cp -r {src_path} {dst_path}')
                print(f'Copied {src_path} to {dst_path}')

            print(f'Created test split for {dirs} ({num_test_episodes} episodes)')


    elif dataset_name == 'openx_translated':

        dataset_path = f'{base_dir}'
        for dirs in os.listdir(dataset_path):

            if 'train' in dirs:
                # Get path to this train directory
                train_dir = os.path.join(dataset_path, dirs)
                
                # Get all shard files and sort them by shard number
                shard_files = [f for f in os.listdir(train_dir) if not f.endswith('.pkl')]
                shard_files.sort(key=lambda x: int(x.split('_')[1]))  # Sort by number after shard_
                
                # Calculate number of test shards (20%)
                num_test_shards = int(len(shard_files) * 0.2)
                
                # Select first 20% as test shards
                test_shard_files = shard_files[:num_test_shards]
                
                # Create test directory structure
                output_base = f'{output_dir}/{dataset_name}'
                test_dir = os.path.join(output_base, dirs.replace('train', 'test'))
                os.makedirs(test_dir, exist_ok=True)
                
                # Copy test shards to test directory
                for shard_file in test_shard_files:
                    src_path = os.path.join(train_dir, shard_file)
                    dst_path = os.path.join(test_dir, shard_file)
                    os.system(f'cp -r {src_path} {dst_path}')
                    print(f'Copied {src_path} to {dst_path}')
                
                print(f'Created test split for {dirs} ({num_test_shards} shards)')
            
            elif 'test' or 'val' in dirs:
                # Get path to this test directory
                test_dir = os.path.join(dataset_path, dirs)
                
                # Create test directory structure in output
                output_base = f'{output_dir}/{dataset_name}'
                output_test_dir = os.path.join(output_base, dirs.replace('val', 'test'))
                os.makedirs(output_test_dir, exist_ok=True)
                
                # Copy all shards from test directory
                for shard_file in os.listdir(test_dir):
                    src_path = os.path.join(test_dir, shard_file)
                    dst_path = os.path.join(output_test_dir, shard_file)
                    os.system(f'cp -r {src_path} {dst_path}')
                    print(f'Copied {src_path} to {dst_path}')
                
                print(f'Copied all test shards from {test_dir} to {output_test_dir}')

    
    elif dataset_name == 'vd4rl_translated':

        #Need to create the episodes based on the timesteps over multiple files - there are 512 timesteps per file
        # Get base directory for this dataset
        dataset_path = f'{base_dir}'
        
        # Create output directory if it doesn't exist
        output_base = f'{output_dir}/{dataset_name}'
        os.makedirs(output_base, exist_ok=True)

        # Get all files and sort them by the numeric value after 'px'
        categories = os.listdir(dataset_path)
        for category in categories:
            px_category = os.listdir(os.path.join(dataset_path, category))
            #print(px_category)
            for px in px_category:
                all_files = []
                output_base_px = None
                if os.path.isdir(os.path.join(output_base, category, px)) == False:
                    output_base_px = os.path.join(output_base, category, px)
                    os.makedirs(output_base_px, exist_ok=True)
                for f in os.listdir(os.path.join(dataset_path, category, px)):
                        all_files.append(os.path.join(dataset_path, category, px, f))
            
                #print(all_files)
            
            
                # Sort files based on the numeric value after 'px'
                sorted_files = sorted(all_files, key=lambda x: int(x.split('px')[-1]))

                #print(sorted_files)

                # Calculate number of test files (20%)
                num_test_files = int(len(sorted_files) * 0.2)
                test_files = sorted_files[:num_test_files]
                #print(test_files)

                # Load and process files
                current_episode = []
                episode_complete = True
                file_idx = num_test_files - 1
                

                print(f'Creating splits in {dataset_path}')

                episode_counter = 0

                for i, test_file in enumerate(test_files):
                    # Load the tf dataset
                    ds = tf.data.Dataset.load(test_file)

                    for ts in ds:
                        first_timestep = ts
                        break


                    if 'is_init' in first_timestep.keys():
                        for timestep in ds:
                            if timestep['next']['done'] == True:
                                current_episode.append(timestep)
                                episode_complete = True
                                # Save completed episode
                                episode_counter += 1
                                output_path = os.path.join(output_base_px, f'test_episode_{episode_counter}')
                                test_episodes_list_dict = {}
                                for t in current_episode:
                                    process_dict(t, [], test_episodes_list_dict)
                                episode_ds = tf.data.Dataset.from_tensor_slices(test_episodes_list_dict)
                                tf.data.Dataset.save(episode_ds, output_path)
                                current_episode = []

                            elif timestep['is_init']==True:
                                # Start new episode
                                if episode_complete == False:
                                    raise ValueError('Previous episode wasn\'t complete')
                                else:
                                    current_episode = [timestep]
                                    episode_complete = False
                            
                            else:
                                current_episode.append(timestep)
                    
                    else:
                        for timestep in ds:
                            if timestep['state']['step_type'] == 0:
                                # Start new episode
                                if episode_complete == False:
                                    raise ValueError('Previous episode wasn\'t complete')
                                else:
                                    current_episode = [timestep]
                                    episode_complete = False
                            else:
                                current_episode.append(timestep)
                            
                            if timestep['state']['step_type'] == 2:
                                episode_complete = True
                                # Save completed episode
                                episode_counter += 1
                                output_path = os.path.join(output_base_px, f'test_episode_{episode_counter}')
                                test_episodes_list_dict = {}
                                for t in current_episode:
                                    process_dict(t, [], test_episodes_list_dict)
                                episode_ds = tf.data.Dataset.from_tensor_slices(test_episodes_list_dict)
                                tf.data.Dataset.save(episode_ds, output_path)
                                current_episode = []

                    # If we're at the last file and episode isn't complete
                    if i == len(test_files)-1 and not episode_complete:
                        # Keep loading next files until episode completes
                        while not episode_complete and file_idx < len(sorted_files):
                            file_idx += 1
                            next_file = sorted_files[file_idx]
                            next_ds = tf.data.Dataset.load(next_file)
                            
                            for timestep in next_ds:
                                current_episode.append(timestep)
                                if 'is_init' in timestep.keys():
                                    if timestep['next']['done'] == True:
                                        episode_complete = True
                                        episode_counter += 1
                                        # Save the completed episode
                                        test_episodes_list_dict = {}
                                        for t in current_episode:
                                            process_dict(t, [], test_episodes_list_dict)
                                        episode_ds = tf.data.Dataset.from_tensor_slices(test_episodes_list_dict)
                                        output_path = os.path.join(output_base_px, f'test_episode_{episode_counter}')
                                        tf.data.Dataset.save(episode_ds, output_path)
                                        current_episode = []
                                        break
                                else:
                                    if timestep['state']['step_type'] == 2:
                                        episode_complete = True
                                        episode_counter += 1
                                        # Save completed episode
                                        output_path = os.path.join(output_base_px, f'test_episode_{episode_counter}')
                                        test_episodes_list_dict = {}
                                        for t in current_episode:
                                            process_dict(t, [], test_episodes_list_dict)
                                        episode_ds = tf.data.Dataset.from_tensor_slices(test_episodes_list_dict)
                                        tf.data.Dataset.save(episode_ds, output_path)
                                        current_episode = []
                                        break
                            
                            if episode_complete:
                                break

                print(f'Created test split with {num_test_files} files in {output_base}')

    
if __name__ == "__main__":
    args = parse_args()
    main(args.dataset, args.output_dir, args.base_dir)
    
