import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional
from abc import ABC, abstractmethod
import sys
import os

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.eval_utils import (
    get_action_decoding_strategy,
    calculate_mse,
    calculate_success_rate,
    normalize_mse_values,
    standardize_predicted_action,
    load_preprocessed_expert_action
)

logger = logging.getLogger(__name__)
if os.environ.get('ENVIRONMENT', 'prod') == 'dev':
    logger.setLevel(logging.DEBUG)


class OpenVLABaseEvaluator:
    """Base class for OpenVLA evaluation on different datasets.
    
    This class contains common evaluation logic that can be shared between
    different dataset types like OpenX and ProcGen.
    """
    
    def __init__(self, cfg: Any, model: Any, processor: Any, dataset_name: str):
        """Initialize the evaluator.
        
        Args:
            cfg: Configuration object
            model: The OpenVLA model to evaluate
            processor: The processor for model inputs
            dataset_name: Name of the dataset being evaluated
        """
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.dataset_name = dataset_name
        self.action_decoding_strategy = get_action_decoding_strategy(model, dataset_name)
        
        if self.action_decoding_strategy == cfg.default_action_decoding_strategy:
            logger.info(f"Action decoding strategy not found for dataset {dataset_name}. "
                       f"Defaulting to {cfg.default_action_decoding_strategy}")
    
    @abstractmethod
    def process_observation(self, batch: Dict[str, Any], obs: Dict[str, Any], 
                           timestep_idx: int) -> Optional[Tuple[Dict[str, Any], Any]]:
        """Process observation for the current timestep.
        
        Args:
            batch: Batch of data from the dataloader
            obs: Observation dictionary to be updated
            timestep_idx: Index of the current timestep
            
        Returns:
            Tuple of (updated obs, text_obs) or None if processing failed
        """
        raise NotImplementedError("Subclasses must implement process_observation")
    
    @abstractmethod
    def get_actual_action(self, batch: Dict[str, Any], episode_idx: int, 
                         timestep_idx: int) -> Any:
        """Get the actual action for the current timestep.
        
        Args:
            batch: Batch of data from the dataloader
            episode_idx: Index of the current episode
            timestep_idx: Index of the current timestep
            
        Returns:
            Actual action for the current timestep
        """
        raise NotImplementedError("Subclasses must implement get_actual_action")
    
    @abstractmethod
    def is_last_timestep(self, batch: Dict[str, Any], timestep_idx: int) -> bool:
        """Check if the current timestep is the last one in the episode.
        
        Args:
            batch: Batch of data from the dataloader
            timestep_idx: Index of the current timestep
            
        Returns:
            True if this is the last timestep, False otherwise
        """
        raise NotImplementedError("Subclasses must implement is_last_timestep")
    
    @abstractmethod
    def get_dataloader(self, tfds_shards: List[str]) -> Any:
        """Get dataloader for the dataset.
        
        Args:
            tfds_shards: List of TensorFlow dataset shard paths
            
        Returns:
            Dataloader for the dataset
        """
        raise NotImplementedError("Subclasses must implement get_dataloader")
    
    def process_batch(self, batch: Dict[str, Any], episode_idx: int) -> Tuple[List[float], List[int]]:
        """Process a single batch (episode) of data.
        
        Args:
            batch: Batch of data from the dataloader
            episode_idx: Index of the current episode
            
        Returns:
            Tuple of (timestep_mses, action_success) for this batch
        """
        timestep_mses = []
        action_success = []
        obs = {}
        
        try:
            obs_len = len(batch['image_observation'][0])
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error accessing image_observation: {e}")
            logger.info(f"Available keys: {batch.keys()}")

            return timestep_mses, action_success
        
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
                mse = calculate_mse(standardized_predicted_action, actual_action)
                timestep_mses.append(mse)
                
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

                
        return timestep_mses, action_success

    def evaluate(self, tfds_shards: List[str]) -> Tuple[float, float, float, int, float]:
        """Evaluate the model on the dataset.
        
        Args:
            tfds_shards: List of TensorFlow dataset shard paths
            
        Returns:
            Tuple of (action_success_rate, total_dataset_amse, avg_dataset_amse,
                     num_timesteps, normalized_amse)
        """
        dataloader = self.get_dataloader(tfds_shards)
        
        all_timestep_mses = []
        all_action_success = []
        
        episode_idx = 0
        
        for batch in dataloader:
            batch_timestep_mses, batch_action_success = self.process_batch(batch, episode_idx)
            
            all_timestep_mses.extend(batch_timestep_mses)
            all_action_success.extend(batch_action_success)
            
            episode_idx += 1

            # Uncomment to limit evaluation to 5 episodes
            # if episode_idx == 2:
            #     break
        
        # Calculate metrics
        action_success_rate = calculate_success_rate(all_action_success)
        logger.debug(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")
        
        total_dataset_amse = sum(all_timestep_mses)
        logger.info(f"\nTotal MSE across {len(all_timestep_mses)} timesteps: {total_dataset_amse:.4f}")
        
        num_timesteps = len(all_timestep_mses)
        avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0
        
        normalized_mses = normalize_mse_values(all_timestep_mses)
        normalized_amse = sum(normalized_mses) / len(normalized_mses) if len(normalized_mses) > 0 else 0.0
        logger.debug(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
        
        return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse
