import logging
from pathlib import Path
import sys
import os
import numpy as np
from typing import Optional

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.eval.profiling.openvla.experiments.robot.openvla_openx_dataloader import get_openx_dataloader
from src.eval.profiling.openvla.experiments.robot.openvla_eval_base import OpenVLABaseEvaluator
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.eval_utils import (
    calculate_mae,
    calculate_mean,
    calculate_success_rate,
    quantile_filter,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
    min_max_normalize,
    standardize_predicted_action,
    preprocess_expert_actions
)


logger = logging.getLogger(__name__)
if os.environ.get('ENVIRONMENT', 'prod') == 'dev':
    logger.setLevel(logging.DEBUG)


class OpenXEvaluator(OpenVLABaseEvaluator):
    """OpenVLA evaluator for OpenX datasets."""
    
    def get_dataloader(self, tfds_shards: list[str]) -> any:
        """Get OpenX dataloader.
        
        Args:
            tfds_shards: List of TensorFlow dataset shard paths
            
        Returns:
            OpenX dataloader
        """
        return get_openx_dataloader(tfds_shards, batch_size=1)
    
    def process_observation(self, batch: dict[str, any], obs: dict[str, any], 
                           timestep_idx: int) -> Optional[tuple[dict[str, any], any]]:
        """Process OpenX observation for the current timestep.
        
        Args:
            batch: Batch of data from the dataloader
            obs: Observation dictionary to be updated
            timestep_idx: Index of the current timestep
            
        Returns:
            Tuple of (updated obs, text_obs) or None if processing failed
        """
        # batch_idx is 0 for OpenX dataset
        obs['full_image'] = batch['image_observation'][0][timestep_idx]
        if obs['full_image'] is None:
            raise ValueError(f"Observation is None for dataset {self.dataset_name}")

        text_obs = batch['text_observation'][0][timestep_idx]
        if text_obs is None:
            raise ValueError(f"Text observation is None for dataset {self.dataset_name}")
        
        logger.debug(f"Image obs shape: {obs['full_image'].shape}")
        logger.debug(f"Text obs: {text_obs}")
        
        return obs, text_obs
    
    def get_actual_action(self, batch: dict[str, any], episode_idx: int, timestep_idx: int) -> any:
        return preprocess_expert_actions(
            batch, 
            self.dataset_name, 
            0,  # batch_idx is 0 for OpenX dataset
            timestep_idx
        )
    
    def process_batch(self, batch: dict[str, any], episode_idx: int) -> tuple[list[float], list[int]]:
        timestep_maes = []
        action_success = []
        obs = {}
        
        try:
            obs_len = len(batch['image_observation'][0])
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Error accessing image_observation: {e}")
        
        for timestep_idx in range(obs_len):
            logger.debug("================================")
            logger.debug(f"{self.dataset_name}")
            logger.debug(f"episode-{episode_idx} timestep-{timestep_idx}")
            logger.debug("================================")
            
            try:
                # Process observation
                obs, text_obs = self.process_observation(batch, obs, timestep_idx)
                if not obs or not text_obs:
                    continue

                # Get and process action
                predicted_action = get_action(self.cfg, self.model, obs, text_obs, self.processor)
                logger.debug(f"Predicted action: {predicted_action}")
                
                # Standardize predicted action
                standardized_predicted_action = standardize_predicted_action(
                    predicted_action,
                    self.action_decoding_strategy,
                    self.dataset_name
                )
                
                # Get actual action
                actual_action = self.get_actual_action(batch, episode_idx, timestep_idx)
                if actual_action is None:
                    raise ValueError(f"Actual action is None for dataset {self.dataset_name}")
                
                logger.debug(f"Standardized predicted action: {standardized_predicted_action}")
                logger.debug(f"Actual action: {actual_action}")
                
                # Calculate MSE
                mae = calculate_mae(standardized_predicted_action, actual_action)
                timestep_maes.append(mae)
                
                # Check if this is the last timestep
                is_last = self.is_last_timestep(batch, timestep_idx)
                if is_last:
                    logger.info(f"Episode {episode_idx} final predicted action: {np.array(standardized_predicted_action)}")
                    logger.info(f"Episode {episode_idx} final actual action: {np.array(actual_action)}")
                    
                    if np.array_equal(np.array(standardized_predicted_action), np.array(actual_action)):
                        action_success.append(1)
                    else:
                        action_success.append(0)
                        
            except (IndexError, KeyError) as e:
                raise f"Error processing dataset at timestep {timestep_idx}: {e}"

        return timestep_maes, action_success

    def is_last_timestep(self, batch: dict[str, any], timestep_idx: int) -> bool:
        """Check if the current timestep is the last one in the OpenX episode"""
        logger.debug(f"is_last: {batch['is_last'][0][timestep_idx]}")
        return batch['is_last'][0][timestep_idx] == True  # batch_idx is 0 for OpenX dataset

    def evaluate(self, tfds_shards: list[str]) -> tuple[float, float, float, int, float]:
        """Evaluate the model on the dataset.
        
        Args:
            tfds_shards: List of TensorFlow dataset shard paths
            
        Returns:
            Tuple of (action_success_rate, total_dataset_amse, avg_dataset_amse,
                     num_timesteps, normalized_amse)
        """
        dataloader = self.get_dataloader(tfds_shards)
        
        all_timestep_maes = []
        all_action_success = []
        
        episode_idx = 0
        
        for batch in dataloader:
            batch_timestep_maes, batch_action_success = self.process_batch(batch, episode_idx)
            
            all_timestep_maes.extend(batch_timestep_maes)
            all_action_success.extend(batch_action_success)
            
            episode_idx += 1

            # Uncomment to limit evaluation to 5 episodes
            # if episode_idx == 2:
            #     break

        # Calculate quantile filtered MAE metrics
        quantile_filtered_maes = quantile_filter(all_timestep_maes)
        total_quantile_filtered_mae = sum(quantile_filtered_maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
        average_quantile_filtered_normalized_mae = calculate_mean(normalized_quantile_filtered_maes)

        max_rel_mae = calculate_max_relative_mae(all_timestep_maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(all_timestep_maes)

        action_success_rate = calculate_success_rate(all_action_success)
        num_timesteps = len(all_timestep_maes)

        total_dataset_amae = sum(all_timestep_maes)
        avg_dataset_amae = calculate_mean(all_timestep_maes)

        normalized_maes = min_max_normalize(all_timestep_maes)
        average_normalized_mae = calculate_mean(normalized_maes)

        logger.debug(f"num_timesteps: {num_timesteps}")
        logger.debug(f"action_success_rate: {action_success_rate}")
        logger.debug(f"total_dataset_amae: {total_dataset_amae}")
        logger.debug(f"avg_dataset_amae: {avg_dataset_amae}")
        logger.debug(f"average_normalized_mae: {average_normalized_mae}")
        logger.debug(f"total_quantile_filtered_mae: {total_quantile_filtered_mae}")
        logger.debug(f"average_quantile_filtered_normalized_mae: {average_quantile_filtered_normalized_mae}")
        logger.debug(f"max_rel_mae: {max_rel_mae}")
        logger.debug(f"prop_beyond_threshold_mae: {prop_beyond_threshold_mae}")

        return (
            num_timesteps,
            action_success_rate,
            total_dataset_amae,
            avg_dataset_amae,
            average_normalized_mae,
            total_quantile_filtered_mae,
            average_quantile_filtered_normalized_mae,
            max_rel_mae,
            prop_beyond_threshold_mae
        )


def evaluate_openvla_on_openx(cfg: any, model: any, processor: any, 
                             tfds_shards: list[str], dataset_name: str) -> tuple[float, float, float, int, float]:
    """Evaluate OpenVLA model on OpenX dataset.
    
    Args:
        cfg: Configuration object
        model: The OpenVLA model to evaluate
        processor: The processor for model inputs
        tfds_shards: List of TensorFlow dataset shard paths
        dataset_name: Name of the dataset being evaluated
        
    Returns:
        Tuple of (action_success_rate, total_dataset_amse, avg_dataset_amse,
                 num_timesteps, normalized_amse)
    """
    evaluator = OpenXEvaluator(cfg, model, processor, dataset_name)
    return evaluator.evaluate(tfds_shards)
