import logging
from pathlib import Path
import sys
import os
from typing import Dict, Any, Optional

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.data_utils.procgen_dataloader import get_procgen_dataloader
from src.eval.profiling.openvla.experiments.robot.openvla_eval_base import OpenVLABaseEvaluator
from src.eval.profiling.openvla.experiments.robot.eval_utils import load_preprocessed_expert_action

logger = logging.getLogger(__name__)
if os.environ.get('ENVIRONMENT', 'prod') == 'dev':
    logger.setLevel(logging.DEBUG)

class ProcGenEvaluator(OpenVLABaseEvaluator):
    """OpenVLA evaluator for ProcGen datasets."""
    
    def get_dataloader(self, tfds_shards: list[str]) -> any:
        """Get ProcGen dataloader.
        
        Args:
            tfds_shards: List of TensorFlow dataset shard paths
            
        Returns:
            ProcGen dataloader
        """
        _, dataloader = get_procgen_dataloader(tfds_shards, batch_size=1, by_episode=True)
        return dataloader
    
    def process_observation(self, batch: dict[str, any], obs: dict[str, any], 
                           timestep_idx: int) -> Optional[tuple[dict[str, any], any]]:
        """Process ProcGen observation for the current timestep.
        
        Args:
            batch: Batch of data from the dataloader
            obs: Observation dictionary to be updated
            timestep_idx: Index of the current timestep
            
        Returns:
            Tuple of (updated obs, text_obs) or None if processing failed
        """
        obs['full_image'] = batch['image_observation'][0][timestep_idx]
        if obs['full_image'] is None:
            raise ValueError(f"Observation is None for dataset {self.dataset_name}")

        text_obs = batch['text_observation'][0][0]  # ProcGen has the same text for all timesteps
        if text_obs is None:
            raise ValueError(f"Text observation is None for dataset {self.dataset_name}")
        
        logger.debug(f"Image obs shape: {obs['full_image'].shape}")
        logger.debug(f"Text obs: {text_obs}")
        
        return obs, text_obs
    
    def get_actual_action(self, batch: Dict[str, Any], episode_idx: int, timestep_idx: int) -> any:
        return load_preprocessed_expert_action(
            batch, 
            self.dataset_name, 
            episode_idx,
            timestep_idx, 
            self.action_decoding_strategy
        )
    
    def is_last_timestep(self, batch: dict[str, any], episode_idx: int, timestep_idx: int) -> bool:
        """Check if the current timestep is the last one in the ProcGen episode"""
        logger.debug(f"is_last: {batch['is_last'][episode_idx][timestep_idx]}")
        return batch['is_last'][episode_idx][timestep_idx] == True

def evaluate_openvla_on_procgen(cfg: any, model: any, processor: any, 
                               tfds_shards: list[str], dataset_name: str) -> tuple[float, float, float, int, float]:
    """Evaluate OpenVLA model on ProcGen dataset.
    
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
    evaluator = ProcGenEvaluator(cfg, model, processor, dataset_name)
    return evaluator.evaluate(tfds_shards)
