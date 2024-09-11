import os
import tensorflow as tf
import tensorflow_datasets as tfds
import shutil

def get_eval_split(root_folder, output_dir, eval_percentage=0.2):

    datasets_with_test_split = os.listdir('<path to openx test translated dataset>')
    datasets_with_val_split = os.listdir('<path to openx val translated dataset>')
    dataset_folders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    
    for dataset_folder in dataset_folders:

        print(f'Splitting {dataset_folder}')

        if dataset_folder not in datasets_with_test_split and dataset_folder not in datasets_with_val_split:

            dataset_path = os.path.join(root_folder, dataset_folder)
            shard_files = sorted([f for f in os.listdir(dataset_path)], key=lambda x: int(x.split('_')[-1]))
            eval_shard_count = max(1, int(len(shard_files) * eval_percentage))
            eval_shards = shard_files[:eval_shard_count]
            for shard in eval_shards:
                shutil.copytree(os.path.join(dataset_path, shard), os.path.join(os.path.join(output_dir, dataset_folder), shard))
            
            print(f'Eval shards: {eval_shards}')
            
            # Check if the last shard in eval_shards ends with a complete episode
            last_shard = eval_shards[-1]
            last_shard_path = os.path.join(dataset_path, last_shard)
            #print(last_shard_path)

            last_dataset_shard = tf.data.Dataset.load(last_shard_path)
            last_timestep_complete = False
            
            final_elem = None
            for elem in last_dataset_shard:
                final_elem = elem

            is_last = final_elem['is_last']
            if is_last:
                last_timestep_complete = True
                print('last timestep is complete in the last shard itself')
                print(elem['is_last'])
                
            
            # If the last shard doesn't end with a complete episode, add the next shard
            if last_timestep_complete == False and eval_shard_count < len(shard_files):
                next_shard = tf.data.Dataset.load(os.path.join(dataset_path, shard_files[eval_shard_count]))
                complete_episode_shard = next_shard.take_while(lambda elem_not_last: not elem['is_last']).concatenate(next_shard.filter(lambda elem_is_last: elem['is_last']).take(1))
                
                for el in complete_episode_shard:
                    if el['is_last']:
                        print('last timestep is complete in the next shard')
                        print(el)
                        break
                    else:
                        print(el['is_last'])

                # Save the dataset to the output directory
                output_shard_path = os.path.join(os.path.join(output_dir, dataset_folder), shard_files[eval_shard_count])
                tf.data.Dataset.save(complete_episode_shard, output_shard_path)
                    
            

root_folder = '<path to openx translated dataset>'
output_dir = '<path to output directory>'
get_eval_split(root_folder, output_dir=output_dir, eval_percentage=0.2)
