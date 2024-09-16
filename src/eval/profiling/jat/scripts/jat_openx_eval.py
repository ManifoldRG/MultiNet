import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from openx_dataloader import get_openx_dataloader

def evaluate_jat_model(model, processor, tfds_shards):

    # Initialize the dataloader for the OpenX dataset
    dataloader = get_openx_dataloader(tfds_shards, batch_size=1)

    avg_mse_list = []
    total_dataset_amse = 0.0
    episode_count = 0

    for batch in dataloader:

        episode_mse = []
	model.reset_rl() # clear key-value cache for each episode

        #Because the batch size is 1, 1 batch contains 1 episode, which is why the first element is indexed
        for idx in range(len(batch['continuous_observation'][0])):

            #Model is not given a reward prior to the first action it predicts
            if idx == 0:
                reward = None
            else:
                reward = batch['reward'][0][idx-1]

            mse = 0.0
            # Get the model's predicted action
            predicted_action = model.get_next_action(
                processor,
                text_observation=batch['text_observation'][0][idx],
                image_observation=batch['image_observation'][0][idx],
                continuous_observation=batch['continuous_observation'][0][idx],
                discrete_observation=batch['discrete_observation'][0][idx],
                reward=reward,
                action_space=batch['action'][0][idx]
            )

            # Get the actual (expert) action
            actual_action = batch['action'][0][idx]

            # Calculate RMSE for this timestep
            mse = np.mean((np.array(predicted_action) - np.array(actual_action)) ** 2)
            episode_mse.append(mse)

        # Calculate average RMSE for the episode
        avg_episode_mse = np.mean(episode_mse)
        avg_mse_list.append(avg_episode_mse)
        total_dataset_amse += avg_episode_mse
        episode_count += 1

        print(f"Episode {episode_count} - Average MSE: {avg_episode_mse:.4f}")

    # Calculate overall average RMSE across all episodes
    print(f"\nTotal Average MSE across {episode_count} episodes: {total_dataset_amse:.4f}")

    # Calculate average AMSE over all episodes
    avg_dataset_amse = total_dataset_amse / episode_count
    
    # Calculate min-max normalized AMSE
    min_amse = min(avg_mse_list)
    max_amse = max(avg_mse_list)
    normalized_amse = (avg_dataset_amse - min_amse) / (max_amse - min_amse) if max_amse != min_amse else 0
    

    print(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")

    return avg_mse_list, episode_count, total_dataset_amse, normalized_amse

