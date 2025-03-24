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
    calculate_mae,
    calculate_success_rate,
    min_max_normalize,
    standardize_predicted_action
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
    
    @abstractmethod
    def process_batch(self, batch: Dict[str, Any], episode_idx: int) -> Tuple[List[float], List[int]]:
        """Process a single batch (episode) of data.
        
        Args:
            batch: Batch of data from the dataloader
            episode_idx: Index of the current episode
            
        Returns:
            Tuple of (timestep_mses, action_success) for this batch
        """
        raise NotImplementedError("Subclasses must implement process_batch")

    def evaluate(self, tfds_shards: List[str]) -> Tuple[float, float, float, int, float]:
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
        
        # Calculate metrics
        action_success_rate = calculate_success_rate(all_action_success)
        logger.debug(f"Action Success Rate Percentage for the dataset: {action_success_rate:.4f}")
        
        total_dataset_amse = sum(all_timestep_maes)
        logger.info(f"\nTotal MSE across {len(all_timestep_maes)} timesteps: {total_dataset_amse:.4f}")
        
        num_timesteps = len(all_timestep_maes)
        avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0
        
        normalized_maes = min_max_normalize(all_timestep_maes)
        normalized_amse = sum(normalized_maes) / len(normalized_maes) if len(normalized_maes) > 0 else 0.0
        logger.debug(f"Normalized Average AMSE for dataset: {normalized_amse:.4f}")
        
        return action_success_rate, total_dataset_amse, avg_dataset_amse, num_timesteps, normalized_amse
