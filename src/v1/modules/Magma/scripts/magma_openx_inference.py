import os
import sys
import argparse
import json
import logging
import re
import time
import gc
import tensorflow as tf

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers import logging as transformers_logging

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from src.data_utils.openx_dataloader import get_openx_dataloader
from src.v1.modules.Magma.data.openx.action_tokenizer import ActionTokenizer

from src.eval_utils import (
    quantile_filter,
    calculate_mean,
    min_max_normalize,
    calculate_mse,
    calculate_mae,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def _get_sorted_shard_paths(dataset_dir: str) -> list[str]:
    shard_pattern = re.compile(r'translated_shard_(\d+)$')
    shard_dirs = []
    try:
        search_dir = os.path.join(dataset_dir, 'test')
        if not os.path.exists(search_dir):
            search_dir = dataset_dir

        for dirname in os.listdir(search_dir):
            match = shard_pattern.match(dirname)
            if match:
                shard_num = int(match.group(1))
                full_path = os.path.join(search_dir, dirname)
                shard_dirs.append((shard_num, full_path))
    except OSError as e:
        logger.error(f"Error reading directory {dataset_dir}: {e}")
        raise
    shard_dirs.sort(key=lambda x: x[0])
    return [path for _, path in shard_dirs]

def _calculate_batch_metrics(pred_actions, gt_actions, action_stats=None):
    """Calculate MSE and MAE metrics for a batch of continuous actions."""
    if action_stats is None:
        raise ValueError("action_stats is required for proper invalid prediction handling in OpenX evaluation")
    
    mses, maes = [], []
    total_invalid_preds = 0
    
    for i in range(len(pred_actions)):
        pred = np.array(pred_actions[i])
        gt = np.array(gt_actions[i])
        
        # Check for invalid predictions (NaN, inf, or non-numeric values)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)) or pred.size == 0:
            total_invalid_preds += 1
            # Use worst-case MSE/MAE for invalid predictions using dataset stats directly
            max_vals = np.array(action_stats['max'])
            min_vals = np.array(action_stats['min'])
            mse = calculate_mse(max_vals[:len(gt)], min_vals[:len(gt)])
            mae = calculate_mae(max_vals[:len(gt)], min_vals[:len(gt)])
        else:
            mse = calculate_mse(pred, gt)
            mae = calculate_mae(pred, gt)
        
        mses.append(mse)
        maes.append(mae)
    
    return mses, maes, total_invalid_preds

def _calculate_final_metrics(mses, maes, successes):
    """Calculate comprehensive final metrics for OpenX evaluation."""
    result = {}
    
    # Calculate MSE metrics
    total_dataset_mse = sum(mses)
    num_timesteps = len(mses)
    avg_dataset_mse = total_dataset_mse / num_timesteps if num_timesteps > 0 else 0.0
    
    # Calculate normalized MSE
    if num_timesteps > 1:
        normalized_mses = min_max_normalize(mses)
        normalized_amse = calculate_mean(normalized_mses)
    else:
        normalized_amse = 0.0
    
    # Calculate MAE metrics
    total_dataset_mae = sum(maes)
    avg_dataset_mae = calculate_mean(maes)
    
    if num_timesteps > 1:
        normalized_maes = min_max_normalize(maes)
        normalized_amae = calculate_mean(normalized_maes)
        
        # Calculate quantile filtered MAE metrics
        quantile_filtered_maes = quantile_filter(maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
        normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
        
        # Calculate additional MAE metrics
        max_rel_mae = calculate_max_relative_mae(maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(maes)
    else:
        normalized_amae = 0.0
        normalized_quantile_filtered_amae = 0.0
        max_rel_mae = 0.0
        prop_beyond_threshold_mae = 0.0
    
    # Calculate action success rate
    action_success_rate = None
    if len(successes) > 0:
        action_success_rate = (sum(successes) / len(successes)) * 100
    
    result['action_success_rate'] = action_success_rate
    result['total_dataset_amse'] = total_dataset_mse
    result['total_dataset_amae'] = total_dataset_mae
    result['num_timesteps'] = num_timesteps
    result['avg_dataset_amse'] = avg_dataset_mse
    result['avg_dataset_amae'] = avg_dataset_mae
    result['normalized_amse'] = normalized_amse
    result['normalized_amae'] = normalized_amae
    result['normalized_quantile_filtered_amae'] = normalized_quantile_filtered_amae
    result['max_relative_mae'] = max_rel_mae
    result['proportion_beyond_threshold_mae'] = prop_beyond_threshold_mae
    
    return result

# Action processing from src/v1/modules/Magma/data/openx/datasets/rlds/oxe/transforms.py
def _rel2abs_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).
    Simplified version for numpy arrays.
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1
    thresholded_actions = np.where(opening_mask, 1, np.where(closing_mask, -1, 0))
    
    # Convert to 0 = closed, 1 = open
    new_actions = thresholded_actions / 2 + 0.5
    
    return new_actions

def _create_action_tensor_from_dict(action_dict: dict, dataset_name: str) -> np.ndarray:
    """
    Create action tensor from action dictionary based on dataset type.
    Follows specific transforms from transforms.py for rt1 and bridge_oxe datasets.
    """
    if action_dict is None:
        return None
    
    action_components = []
    logger.debug(f"Processing action_dict for {dataset_name} with keys: {list(action_dict.keys())}")
    
    # RT1 dataset transform (openx_mobile_manipulation)
    if dataset_name == 'openx_mobile_manipulation':
        if 'world_vector' in action_dict and 'rotation_delta' in action_dict and 'gripper_closedness_action' in action_dict:
            # Add world_vector (3D)
            world_vector = np.array(action_dict['world_vector'])
            if world_vector.ndim == 0:
                world_vector = world_vector.reshape(1)
            elif world_vector.ndim > 1:
                world_vector = world_vector.flatten()
            action_components.append(world_vector)
            
            # Add rotation_delta (3D)
            rotation_delta = np.array(action_dict['rotation_delta'])
            if rotation_delta.ndim == 0:
                rotation_delta = rotation_delta.reshape(1)
            elif rotation_delta.ndim > 1:
                rotation_delta = rotation_delta.flatten()
            action_components.append(rotation_delta)
            
            # Add processed gripper_closedness_action (1D) with rel2abs conversion
            gripper_raw = np.array(action_dict['gripper_closedness_action'])
            if gripper_raw.ndim == 0:
                gripper_raw = gripper_raw.reshape(1)
            elif gripper_raw.ndim > 1:
                gripper_raw = gripper_raw.flatten()
            
            # Apply rel2abs_gripper_actions transform (RT1 specific)
            gripper_action = _rel2abs_gripper_actions(gripper_raw)
            if gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1)
            action_components.append(gripper_action)
        else:
            raise KeyError(f"RT1 dataset missing required keys: world_vector, rotation_delta, gripper_closedness_action")
    
    # Bridge OXE dataset transform (openx_single_arm)  
    elif dataset_name == 'openx_single_arm':
        if 'world_vector' in action_dict and 'rotation_delta' in action_dict and 'open_gripper' in action_dict:
            # Add world_vector (3D)
            world_vector = np.array(action_dict['world_vector'])
            if world_vector.ndim == 0:
                world_vector = world_vector.reshape(1)
            elif world_vector.ndim > 1:
                world_vector = world_vector.flatten()
            action_components.append(world_vector)
            
            # Add rotation_delta (3D)
            rotation_delta = np.array(action_dict['rotation_delta'])
            if rotation_delta.ndim == 0:
                rotation_delta = rotation_delta.reshape(1)
            elif rotation_delta.ndim > 1:
                rotation_delta = rotation_delta.flatten()
            action_components.append(rotation_delta)
            
            # Add open_gripper (1D) cast to float32
            gripper_raw = np.array(action_dict['open_gripper'])
            if gripper_raw.ndim == 0:
                gripper_raw = gripper_raw.reshape(1)
            elif gripper_raw.ndim > 1:
                gripper_raw = gripper_raw.flatten()
            gripper_action = gripper_raw.astype(np.float32)
            action_components.append(gripper_action)
        else:
            raise KeyError(f"Bridge OXE dataset missing required keys: world_vector, rotation_delta, open_gripper")
    
    # Concatenate all action components if we have any
    if action_components:
        action_tensor = np.concatenate(action_components)
        logger.debug(f"Created action tensor with shape {action_tensor.shape} for dataset {dataset_name}")
        return action_tensor
    else:
        raise ValueError(f"No valid action components found for dataset {dataset_name}")

#Action stats need to be recalculated when action_dict is processed because of action dict processing 
def _recalculate_action_stats_from_tensors(action_tensors: list, dataset_name: str) -> dict:
    """
    Recalculate action statistics from a list of processed action tensors.
    This is used when action dictionaries are processed to get accurate stats.
    """
    if not action_tensors:
        raise ValueError(f"No action tensors provided for stats calculation for dataset {dataset_name}")
    
    # Convert all tensors to numpy arrays and stack them
    action_arrays = []
    for tensor in action_tensors:
        if tensor is not None:
            action_arrays.append(np.array(tensor))
    
    if not action_arrays:
        raise ValueError(f"No valid action tensors found for dataset {dataset_name}")
    
    # Stack all actions into a single array (num_samples, action_dim)
    try:
        stacked_actions = np.stack(action_arrays)
        action_dim = stacked_actions.shape[1]
        num_samples = stacked_actions.shape[0]
        
        logger.info(f"Recalculating stats from {num_samples} action tensors of dimension {action_dim} for dataset {dataset_name}")
        
        # Calculate statistics
        action_stats = {
            'min': np.min(stacked_actions, axis=0).tolist(),
            'max': np.max(stacked_actions, axis=0).tolist(),
            'sum': np.sum(stacked_actions, axis=0).tolist(),
            'mean': np.mean(stacked_actions, axis=0).tolist(),
            'std': np.std(stacked_actions, axis=0).tolist(),
            'q01': np.quantile(stacked_actions, 0.01, axis=0).tolist(),
            'q99': np.quantile(stacked_actions, 0.99, axis=0).tolist(),
            'count': num_samples,
            'size': [action_dim]
        }
        
        logger.info(f"Recalculated action stats: size={action_stats['size']}, count={action_stats['count']}")
        return action_stats
        
    except Exception as e:
        raise RuntimeError(f"Error recalculating action stats for dataset {dataset_name}: {e}")

#Unnormalize function same as src/v1/modules/Magma/agents/libero/libero_magma_utils.py and src/v1/modules/Magma/tools/simplerenv-magma/simpler_env/policies/magma/magma_model.py
def unnormalize_action(normalized_action, action_stats):
    action_low, action_high = np.array(action_stats["q01"]), np.array(action_stats["q99"])
    return 0.5 * (normalized_action + 1) * (action_high - action_low) + action_low



def run_evaluation(args):
    transformers_logging.set_verbosity_error()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Magma-8B",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    print("Model and processor loaded successfully.")

    logger.info(f"Locating data shards in: {args.dataset_dir}")
    shard_paths = _get_sorted_shard_paths(args.dataset_dir)
    if args.num_shards:
        shard_paths = shard_paths[:args.num_shards]

    if not shard_paths:
        logger.error(f"No data shards found in {args.dataset_dir}. Exiting.")
        return

    logger.info(f"Found {len(shard_paths)} data files. Creating dataloader...")
    dataset, dataloader = get_openx_dataloader(shard_paths, args.batch_size, args.dataset_name)
    action_stats = dataset.action_stats

    # Convert TensorFlow tensors to numpy arrays
    for key in ['min', 'max', 'mean', 'std']:
        if key in action_stats:
            if tf.is_tensor(action_stats[key]):
                action_stats[key] = action_stats[key].numpy()
            else:
                action_stats[key] = np.array(action_stats[key])
    
    # Check if we need to recalculate stats due to action_dict processing
    needs_stats_recalculation = False
    processed_action_tensors = []
    
    # Check first batch to see if action_dict processing is needed
    sample_batch = None
    try:
        for batch in dataloader:
            sample_batch = batch
            break
        
        if (sample_batch is not None and 
            'action_dict' in sample_batch and 
            sample_batch['action_dict'] is not None and 
            len(sample_batch['action_dict']) > 0 and 
            args.dataset_name in ['openx_mobile_manipulation', 'openx_single_arm']):
            needs_stats_recalculation = True
            logger.info(f"Action dictionary processing detected for {args.dataset_name}. Will recalculate stats from processed tensors.")
    except Exception as e:
        raise RuntimeError(f"Error checking for action_dict processing: {e}")
    
    # If we need to recalculate stats, process all batches to collect action tensors
    if needs_stats_recalculation:
        logger.info("Collecting all processed action tensors to recalculate statistics...")
        
        # Recreate dataloader and process all batches
        dataset, dataloader = get_openx_dataloader(shard_paths, args.batch_size, args.dataset_name)
        
        for batch_idx, batch in enumerate(dataloader):
            if ('action_dict' in batch and batch['action_dict'] is not None and 
                len(batch['action_dict']) > 0):
                action_dicts = batch['action_dict']
                
                for action_dict in action_dicts:
                    if action_dict is not None and len(action_dict) > 0:
                        action_tensor = _create_action_tensor_from_dict(action_dict, args.dataset_name)
                        if action_tensor is not None:
                            processed_action_tensors.append(action_tensor)
            
            if batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx} batches for stats calculation, collected {len(processed_action_tensors)} action tensors")
        
        # Recalculate stats from processed tensors
        if processed_action_tensors:
            recalculated_stats = _recalculate_action_stats_from_tensors(processed_action_tensors, args.dataset_name)
            if recalculated_stats is not None:
                action_stats = recalculated_stats
                # Convert to numpy arrays
                for key in ['min', 'max', 'mean', 'std']:
                    if key in action_stats:
                        action_stats[key] = np.array(action_stats[key])
                logger.info(f"Successfully recalculated action stats from {len(processed_action_tensors)} processed action tensors")
            else:
                raise RuntimeError("Failed to recalculate stats, cannot continue without proper statistics")
        else:
            logger.warning("No processed action tensors collected, using original stats")
        
        # Recreate dataloader for evaluation
        dataset, dataloader = get_openx_dataloader(shard_paths, args.batch_size, args.dataset_name)
    
    # Save action stats to file
    stats_output_path = os.path.join(args.output_dir, f'{args.dataset_name}_stats.json')
    with open(stats_output_path, 'w') as f:
        # Convert numpy arrays and TensorFlow tensors to lists for JSON serialization
        stats_to_save = {}
        for k, v in action_stats.items():
            if isinstance(v, np.ndarray):
                stats_to_save[k] = v.tolist()
            elif tf.is_tensor(v):
                stats_to_save[k] = v.numpy().tolist()
            else:
                stats_to_save[k] = v
        json.dump(stats_to_save, f, indent=4)
    
    if needs_stats_recalculation:
        logger.info(f'Recalculated dataset stats saved to {stats_output_path}')
    else:
        logger.info(f'Original dataset stats saved to {stats_output_path}')
    
    logger.info("Action stats calculated and dataloader is ready.")
    
    action_dim = action_stats['min'].shape[0]
    logger.info(f"Action dimension from stats: {action_dim}")
    #Generation args same as src/v1/modules/Magma/tools/simplerenv-magma/simpler_env/policies/magma/magma_model.py and src/v1/modules/Magma/agents/libero/libero_magma_utils.py
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.7, 
        "do_sample": True,
        "num_beams": 1,
        "use_cache": False,  # Disabled to avoid DynamicCache compatibility issues
    }
    
    all_mses, all_maes, all_successes = [], [], []
    total_invalid_predictions = 0
    total_timesteps = 0
    total_batches = len(dataloader)
    logger.info(f"Starting evaluation on {total_batches} batches...")
    start_time = time.perf_counter()
    
    for batch_counter, batch in enumerate(dataloader, 1):
        #try:
        images = [Image.fromarray(img) for img in batch["image_observation"]]
        instructions = [txt for txt in batch["text_observation"]]
        
        # Process ground truth actions - check for action_dict first
        gt_actions = batch['action']
    
        # Check if action_dict is available and dataset requires special processing
        if ('action_dict' in batch and batch['action_dict'] is not None and 
            len(batch['action_dict']) > 0):
            action_dicts = batch['action_dict']
            
            # Check if this is a dataset that requires action dict processing
            if args.dataset_name in ['openx_mobile_manipulation', 'openx_single_arm']:
                logger.debug(f"Processing action_dict for dataset {args.dataset_name}")
                processed_gt_actions = []
                
                for i, action_dict in enumerate(action_dicts):
                    if action_dict is not None and len(action_dict) > 0:
                        # Create action tensor from dictionary
                        action_tensor = _create_action_tensor_from_dict(action_dict, args.dataset_name)
                        if action_tensor is not None:
                            processed_gt_actions.append(action_tensor)
                            logger.debug(f"Sample {i}: Created action tensor with shape {action_tensor.shape}")
                        else:
                            # Fallback to original action if dict processing fails
                            processed_gt_actions.append(np.array(gt_actions[i]))
                            logger.warning(f"Sample {i}: Falling back to original action")
                    else:
                        # Use original action if dict is None or empty
                        processed_gt_actions.append(np.array(gt_actions[i]))
                
                if processed_gt_actions:
                    gt_actions = processed_gt_actions
                    logger.debug(f"Successfully processed {len(processed_gt_actions)} actions from action_dict")
                else:
                    raise ValueError("No valid actions processed from action_dict, cannot continue")
        
        # Convert to numpy array format
        gt_actions = np.array([np.array(action) for action in gt_actions])
        
        # Process each image-prompt pair individually (batch processing throwing errors as of now)
        batch_normalized_actions = []
        
        #Same prompt as src/v1/modules/Magma/tools/simplerenv-magma/simpler_env/policies/magma/magma_model.py and src/v1/modules/Magma/agents/libero/libero_magma_utils.py
        for idx, (image, inst) in enumerate(zip(images, instructions)):
            convs = [
                {"role": "user", "content": f"<image>\nWhat action should the robot take to {inst}?"},
            ]
            convs = [
                {
                    "role": "system",
                    "content": "You are agent that can see, talk and act.", 
                },            
            ] + convs      
            prompt = processor.tokenizer.apply_chat_template(
                convs,
                tokenize=False,
                add_generation_prompt=True
            )
            # Handle image tokens like libero
            if hasattr(model.config, 'mm_use_image_start_end') and model.config.mm_use_image_start_end:
                prompt = prompt.replace("<image>", "<image_start><image><image_end>")
            
            # Process single image with proper dimension handling
            inputs = processor(images=image, texts=prompt, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
            inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            #Generate and process action same as src/v1/modules/Magma/tools/simplerenv-magma/simpler_env/policies/magma/magma_model.py and src/v1/modules/Magma/agents/libero/libero_magma_utils.py
            with torch.inference_mode():
                generate_ids = model.generate(**inputs, **generation_args)
            
            action_ids = generate_ids[0, -8:-1].cpu().tolist() 
            action_ids = np.array(action_ids).astype(np.int64)

            action_tokenizer = ActionTokenizer(processor.tokenizer)
            normalized_action = action_tokenizer.decode_token_ids_to_actions(action_ids)
            batch_normalized_actions.append(normalized_action) 
            
            # Clean up intermediate tensors
            del inputs, generate_ids, action_ids
            torch.cuda.empty_cache()
        
        # Combine all normalized actions
        normalized_actions = np.array(batch_normalized_actions)
        
        logger.info(f"Normalized actions shape: {normalized_actions.shape}")
        logger.info(f"Ground truth actions shape: {gt_actions.shape}")
        
        # Check if we have the right number of action dimensions
        if normalized_actions.shape[1] != action_stats['min'].shape[0]:
            logger.warning(f"Dimension mismatch: predicted {normalized_actions.shape[1]} dims, expected {action_stats['min'].shape[0]} dims")
            # Pad or truncate to match expected dimensions
            expected_dim = action_stats['min'].shape[0]
            if normalized_actions.shape[1] < expected_dim:
                # Pad with zeros if we have fewer dimensions
                padding = np.zeros((normalized_actions.shape[0], expected_dim - normalized_actions.shape[1]))
                normalized_actions = np.concatenate([normalized_actions, padding], axis=1)
                logger.info(f"Padded actions to shape: {normalized_actions.shape}")
            else:
                # Truncate if we have more dimensions
                normalized_actions = normalized_actions[:, :expected_dim]
                logger.info(f"Truncated actions to shape: {normalized_actions.shape}")

        # Unnormalize actions
        pred_actions = np.array([unnormalize_action(act, action_stats) for act in normalized_actions])

        # Dynamic action comparison - clip to minimum dimension for fair comparison
        effective_dim = min(pred_actions.shape[1], gt_actions.shape[1])
        pred_clipped = pred_actions[:, :effective_dim]
        gt_clipped = gt_actions[:, :effective_dim]

        logger.info(f"Comparing actions - Pred shape: {pred_clipped.shape}, GT shape: {gt_clipped.shape}")

        mses, maes, invalid_preds = _calculate_batch_metrics(pred_clipped, gt_clipped, action_stats)
        total_invalid_predictions += invalid_preds
        total_timesteps += len(pred_clipped)
        
        # Calculate action success (exact match for continuous actions)
        successes = []
        for i in range(pred_clipped.shape[0]):
            if np.array_equal(pred_clipped[i], gt_clipped[i]):
                successes.append(1)
            else:
                successes.append(0)
        
        all_mses.extend(mses)
        all_maes.extend(maes)
        all_successes.extend(successes)

        logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
            
        '''except Exception as e:
            logger.error(f"Error in batch {batch_counter}: {e}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()'''

    logger.info("Evaluation loop finished. Calculating final metrics...")
    final_metrics = _calculate_final_metrics(all_mses, all_maes, all_successes)
    
    # Calculate invalid predictions percentage
    if total_timesteps > 0:
        invalid_percentage = (total_invalid_predictions / total_timesteps) * 100
    else:
        invalid_percentage = 0.0
    
    # Add evaluation metadata
    end_time = time.perf_counter()
    eval_time = end_time - start_time
    final_metrics['eval_time'] = eval_time
    final_metrics['total_invalid_preds'] = total_invalid_predictions
    final_metrics['invalid_predictions_percentage'] = invalid_percentage
    final_metrics['total_batches'] = batch_counter  # Use actual count instead of total_batches
    final_metrics['total_timesteps'] = total_timesteps
    
    results_file = os.path.join(args.output_dir, args.results_filename)
    with open(results_file, 'w') as f:
        json.dump({args.dataset_name: final_metrics}, f, indent=4)

    logger.info(f"Success! Detailed results saved to {results_file}")
    logger.info("\n--- Detailed Metrics ---\n" + json.dumps(final_metrics, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Run Magma model evaluation on OpenX datasets.")
    
    parser.add_argument('--dataset_dir', type=str, required=True, help='Root directory of the OpenX dataset shards.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the output results JSON file.')
    parser.add_argument('--dataset_name', type=str, default='openx', help='Name of the dataset being evaluated.')
    parser.add_argument('--results_filename', type=str, default='magma_openx_results.json', help='Name for the output results file.')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference.')
    parser.add_argument('--num_shards', type=int, default=None, help='Number of data shards to process. Processes all if not specified.')
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()