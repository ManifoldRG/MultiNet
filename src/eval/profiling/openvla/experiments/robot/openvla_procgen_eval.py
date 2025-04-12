import logging
from pathlib import Path
import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional

current_file = Path(__file__).resolve()
project_root = next(
    (p for p in current_file.parents if p.name == "MultiNet"),
    current_file.parent
)
sys.path.append(str(project_root))

from src.data_utils.procgen_dataloader import get_procgen_dataloader
from src.eval.profiling.openvla.experiments.robot.openvla_eval_base import OpenVLABaseEvaluator
from src.eval.profiling.openvla.experiments.robot.robot_utils import get_action
from src.eval.profiling.openvla.experiments.robot.eval_utils import (
    standardize_predicted_action,
    preprocess_expert_actions,
)
from definitions.procgen import ProcGenDefinitions
from src.eval_utils import (
    calculate_tp_fp_fn_counts,
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    calculate_success_rate,
    calculate_brier_mae,
    min_max_normalize,
    quantile_filter,
    calculate_mean,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
    get_exact_match_rate
)


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
        _, dataloader = get_procgen_dataloader(tfds_shards, batch_size=1, dataset_name=self.dataset_name, by_episode=True)
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
        return preprocess_expert_actions(
            batch, 
            self.dataset_name, 
            0,  # batch_idx is 0 for ProcGen dataset
            timestep_idx
        )

    def process_batch(self, batch: dict[str, any], episode_idx: int, action_space: list[int]) -> tuple[list[float], list[int]]:
        """Process a single batch (episode) of data.
        
        Args:
            batch: Batch of data from the dataloader
            episode_idx: Index of the current episode
            
        Returns:
            Tuple of (timestep_brier_maes, action_success, batch_preds, batch_actuals)
        """
        timestep_brier_maes = []
        action_success = []
        obs = {}
        
        try:
            obs_len = len(batch['image_observation'][0])
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Error accessing image_observation: {e}")
        
        # Get action space information
        num_actions = len(action_space)
        batch_preds = []
        batch_actuals = []
        batch_argmax_preds = []
        batch_action_probs = []
        batch_argmax_mismatches = 0
        
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
                predictions = get_action(
                    self.cfg,
                    self.model,
                    obs,
                    text_obs,
                    self.processor,
                    return_logits=True
                )
                predicted_action = predictions['actions']
                logger.debug(f"Predicted action: {predicted_action}")

                debug_actions = predictions['debug_actions']
                logger.debug(f"Debug actions: {debug_actions}")

                # Standardize predicted action
                standardized_predicted_action = standardize_predicted_action(
                    predicted_action,
                    self.action_decoding_strategy,
                    self.dataset_name
                )
                
                debug_standardized_actions = standardize_predicted_action(
                    debug_actions,
                    self.action_decoding_strategy,
                    self.dataset_name
                )
                logger.debug(f"Debug standardized actions: {debug_standardized_actions}")

                # Get actual action
                actual_action = self.get_actual_action(batch, episode_idx, timestep_idx)
                if actual_action is None:
                    raise ValueError(f"Actual action is None for dataset {self.dataset_name}")


                batch_preds.append(standardized_predicted_action)
                batch_actuals.append(actual_action)

                logger.debug(f"Actual action: {actual_action}")
                logger.warning(f"Standardized predicted action: {standardized_predicted_action}")

                action_probs = predictions['action_probs_by_dimension'][0]  # Procgen only has 1 action dim
                logger.warning(f"Argmax action probs: {int(np.argmax(action_probs))}")
                logger.debug(f"Action probs: {action_probs}")
                batch_argmax_preds.append(int(np.argmax(action_probs)))

                if standardized_predicted_action[0] != int(np.argmax(action_probs)):
                    logger.warning("MISMATCH")

                if int(np.argmax(action_probs)) != standardized_predicted_action:
                    batch_action_probs.append({'pred': standardized_predicted_action, 'probs': action_probs.tolist(), 'mismatch': True})
                    batch_argmax_mismatches += 1
                else:
                    batch_action_probs.append({'pred': standardized_predicted_action, 'probs': action_probs.tolist(), 'mismatch': False})

                one_hot_actual = [0.0] * num_actions
                one_hot_actual[int(actual_action[0])] = 1.0
                
                brier_mae = calculate_brier_mae(action_probs, one_hot_actual)
                timestep_brier_maes.append(brier_mae)

                logger.debug(f"Predicted probs: {action_probs}")
                logger.debug(f"Actual one-hot: {one_hot_actual}")
                logger.debug(f"Brier MAE: {brier_mae}")
                    
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

        return (
            timestep_brier_maes, 
            action_success,
            batch_preds,
            batch_actuals,
            batch_argmax_preds,
            batch_action_probs,
            batch_argmax_mismatches
        )
    
    def is_last_timestep(self, batch: dict[str, any], timestep_idx: int) -> bool:
        """Check if the current timestep is the last one in the ProcGen episode"""
        logger.debug(f"is_last: {batch['is_last'][0][timestep_idx]}")
        return batch['is_last'][0][timestep_idx] == True

    def evaluate(self, tfds_shards: list[str]) -> tuple[float, float, float, int, float]:
        """Evaluate the model on the dataset.
        
        Args:
            tfds_shards: List of TensorFlow dataset shard paths
            
        Returns:
            Tuple of metrics (
                all_preds,
                all_actuals,
                invalids,
                invalid_percentage,
                num_timesteps,
                action_success_rate,
                total_dataset_brier_mae,
                avg_dataset_brier_mae,
                average_normalized_brier_mae,
                total_quantile_filtered_brier_mae,
                average_quantile_filtered_normalized_brier_mae,
                max_rel_brier_mae,
                prop_beyond_threshold_brier_mae,
                total_micro_precision,
                total_micro_recall,
                total_micro_f1
            )
        """
        dataloader = self.get_dataloader(tfds_shards)
        action_space = sorted(ProcGenDefinitions.get_valid_action_space(self.dataset_name, 'default'))
        all_brier_maes, all_action_successes = [], []
        all_preds, all_actuals, all_argmax_preds, all_action_probs = [], [], [], []
        episode_idx = 0
        all_argmax_mismatches = 0
        
        for batch in dataloader:
            (batch_brier_maes, batch_action_successes, batch_preds, batch_actuals, batch_argmax_preds, batch_action_probs, batch_argmax_mismatches) = self.process_batch(batch, episode_idx, action_space)
            
            all_brier_maes.extend(batch_brier_maes)
            all_action_successes.extend(batch_action_successes)
            all_preds.extend(batch_preds)
            all_actuals.extend(batch_actuals)
            all_argmax_preds.extend(batch_argmax_preds)
            all_action_probs.extend(batch_action_probs)
            all_argmax_mismatches += batch_argmax_mismatches
            episode_idx += 1

            # Uncomment to limit evaluation to 2 episodes
            if episode_idx == 1:
                break

        # Calculate quantile filtered MAE metrics
        quantile_filtered_brier_maes = quantile_filter(all_brier_maes)
        total_quantile_filtered_brier_mae = sum(quantile_filtered_brier_maes)
        normalized_quantile_filtered_brier_maes = min_max_normalize(quantile_filtered_brier_maes)
        average_quantile_filtered_normalized_brier_mae = calculate_mean(normalized_quantile_filtered_brier_maes)

        max_rel_brier_mae = calculate_max_relative_mae(all_brier_maes)
        prop_beyond_threshold_brier_mae = calculate_proportion_beyond_mae_threshold(all_brier_maes)

        action_success_rate = calculate_success_rate(all_action_successes)
        num_timesteps = len(all_brier_maes)

        total_dataset_brier_mae = sum(all_brier_maes)
        avg_dataset_brier_mae = calculate_mean(all_brier_maes)

        normalized_brier_maes = min_max_normalize(all_brier_maes)
        average_normalized_brier_mae = calculate_mean(normalized_brier_maes)

        exact_match_rate = get_exact_match_rate(np.array(all_preds), np.array(all_actuals))

        tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
            np.array(all_preds), np.array(all_actuals), action_space
        )

        invalid_percentage = int(invalid_fp) / len(all_preds) * 100
        total_micro_precision = get_micro_precision_from_counts(tp, fp)
        total_micro_recall = get_micro_recall_from_counts(tp, fn)
        total_micro_f1 = get_micro_f1(total_micro_precision, total_micro_recall)
        average_micro_precision = total_micro_precision / num_timesteps
        average_micro_recall = total_micro_recall / num_timesteps
        average_micro_f1 = total_micro_f1 / num_timesteps

        return (
            all_preds,
            all_actuals,
            all_argmax_preds,
            all_action_probs,
            all_argmax_mismatches,
            int(invalid_fp),
            invalid_percentage,
            num_timesteps,
            action_success_rate,
            total_dataset_brier_mae,
            avg_dataset_brier_mae,
            average_normalized_brier_mae,
            total_quantile_filtered_brier_mae,
            average_quantile_filtered_normalized_brier_mae,
            max_rel_brier_mae,
            prop_beyond_threshold_brier_mae,
            total_micro_precision,
            total_micro_recall,
            total_micro_f1,
            average_micro_precision,
            average_micro_recall,
            average_micro_f1,
            exact_match_rate
        )


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
