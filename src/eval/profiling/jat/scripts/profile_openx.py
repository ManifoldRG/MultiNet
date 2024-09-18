import os
import time
import json
from transformers import AutoModelForCausalLM, AutoProcessor
from jat_openx_eval import evaluate_jat_model
import numpy as np
import wandb

def profile_jat_on_openx():

    wandb.login()
    run = wandb.init(project="jat-openx-eval", id="cc39qjbx", resume="must")
    # Load the model and processor
    model_name_or_path = "jat-project/jat"
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Path to OpenX datasets
    openx_datasets_path = '<path to openx datasets>'

    # Get list of all OpenX datasets
    openx_dataset_paths = os.listdir(openx_datasets_path)

    eval_results = {}

    for openx_dataset in openx_dataset_paths:
        # Read the stored JSON file to check completed datasets
        if os.path.exists('jat_openx_evaluation_results.json'):
            with open('jat_openx_evaluation_results.json', 'r') as f:
                completed_datasets = json.load(f)
            
            if openx_dataset in completed_datasets:
                print(f'\nSkipping dataset: {openx_dataset} (already evaluated)\n')
                continue
        
        print(f'\nEvaluating dataset: {openx_dataset}\n')

        # Get all shards for the current dataset
        shard_files = os.listdir(os.path.join(openx_datasets_path, openx_dataset))
        sorted_shard_files = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        tfds_shards = [os.path.join(openx_datasets_path, openx_dataset, f) 
                       for f in sorted_shard_files]

        # Start timing
        start_time = time.time()

        # Evaluate JAT model on the current dataset
        action_success_rate, avg_mse_list, episode_count, total_dataset_amse, normalized_amse = evaluate_jat_model(model, processor, tfds_shards)

        # End timing
        end_time = time.time()

        # Calculate evaluation time
        eval_time = end_time - start_time

        # Store results
        eval_results[openx_dataset] = {
            'action_success_rate': action_success_rate,
            'avg_mse_list': avg_mse_list,
            'episode_count': episode_count,
            'total_dataset_amse': total_dataset_amse,
            'normalized_amse': normalized_amse,
            'eval_time': eval_time
        }

        wandb.log({"action_success_rate": action_success_rate, "eval_time": eval_time, "episode_count": episode_count, "total_dataset_amse": total_dataset_amse, "normalized_amse": normalized_amse, "hover_info": f"Dataset: {openx_dataset}"})

        # Save intermediate results to a JSON file to ensure progress is not lost
        # Check if the file already exists
        if os.path.exists('jat_openx_evaluation_results.json'):
            # If it exists, load the existing data
            with open('jat_openx_evaluation_results.json', 'r') as f:
                existing_results = json.load(f)
            # Append new data to existing data
            existing_results.update(eval_results)
        else:
            # If it doesn't exist, use the current eval_results
            existing_results = eval_results

        # Write the updated or new results to the file
        with open('jat_openx_evaluation_results.json', 'w') as f:
            json.dump(existing_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        print(f'Evaluation time for {openx_dataset}: {eval_time:.2f} seconds')
    
    wandb.finish()

    # Print overall results
    print('\nOverall Results:')
    for dataset, result in eval_results.items():
        print(f'\nDataset: {dataset}')
        print(f'Episodes: {result["episode_count"]}')
        print(f'Total AMSE: {result["total_dataset_amse"]:.4f}')
        print(f'Normalized AMSE: {result["normalized_amse"]:.4f}')
        print(f'Evaluation Time: {result["eval_time"]:.2f} seconds')
        print(f'Action Success Rate: {result["action_success_rate"]:.4f}')
    print("\nEval results have been saved to 'jat_openx_evaluation_results.json'")

if __name__ == "__main__":
    profile_jat_on_openx()

