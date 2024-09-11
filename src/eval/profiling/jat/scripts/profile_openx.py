import os
import time
import json
from transformers import AutoModelForCausalLM, AutoProcessor
from jat_openx_eval import evaluate_jat_model
import numpy as np
def profile_jat_on_openx():
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
        print(f'\nEvaluating dataset: {openx_dataset}\n')

        # Get all shards for the current dataset
        tfds_shards = [os.path.join(openx_datasets_path, openx_dataset, f) 
                       for f in os.listdir(os.path.join(openx_datasets_path, openx_dataset))]

        # Start timing
        start_time = time.time()

        # Evaluate JAT model on the current dataset
        avg_mse_list, episode_count, total_dataset_amse, normalized_amse = evaluate_jat_model(model, processor, tfds_shards)

        # End timing
        end_time = time.time()

        # Calculate evaluation time
        eval_time = end_time - start_time

        # Store results
        eval_results[openx_dataset] = {
            'avg_mse_list': avg_mse_list,
            'episode_count': episode_count,
            'total_dataset_amse': total_dataset_amse,
            'normalized_amse': normalized_amse,
            'eval_time': eval_time
        }

        print(f'Evaluation time for {openx_dataset}: {eval_time:.2f} seconds')

    # Print overall results
    print('\nOverall Results:')
    for dataset, result in eval_results.items():
        print(f'\nDataset: {dataset}')
        print(f'Episodes: {result["episode_count"]}')
        print(f'Total AMSE: {result["total_dataset_amse"]:.4f}')
        print(f'Normalized AMSE: {result["normalized_amse"]:.4f}')
        print(f'Evaluation Time: {result["eval_time"]:.2f} seconds')

    # Save results to a JSON file
    with open('jat_openx_evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    print("\nEval results have been saved to 'jat_openx_evaluation_results.json'")

if __name__ == "__main__":
    profile_jat_on_openx()

