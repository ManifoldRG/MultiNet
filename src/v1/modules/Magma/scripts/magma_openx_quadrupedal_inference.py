"""
Magma inference script for OpenX quadrupedal robot dataset (utokyo_saytap).

This script evaluates the Magma vision-language model on the OpenX quadrupedal locomotion task.
It combines:
- OpenX data loading and metrics (similar to magma_openx_inference.py)
- Vision-language inference approach (similar to overcooked_single_inference.py and openx_module.py)
- Standard OpenX prompt formatting using definitions (same as openx_module.py)

The quadrupedal dataset contains 12 motor joint positions for controlling a quadruped robot
across various gaits.
"""

import os
import sys
import argparse
import ast
import json
import logging
import time
import gc
import re
from glob import glob

import numpy as np
import torch
import tensorflow as tf
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers import logging as transformers_logging

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from src.data_utils.openx_dataloader import get_openx_dataloader
from definitions.openx import OpenXDefinitions
from definitions.openx_prompt import format_instruction_prompt


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

def _validate_text_output(output: any, shape: tuple) -> np.array:
    if output is None or not isinstance(output, list) or len(output) != shape[0] or any(isinstance(x, (str, np.string_, set)) for x in output):
        return False
    return True


def _get_sorted_shard_paths(dataset_dir: str) -> list:
    """Find and sort translated shard directories."""
    try:
        dataset_dir = glob(f"{dataset_dir}/test/")[0]
        shard_files = glob(f"{dataset_dir}/translated_shard_*")
        tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        return tfds_shards
    except IndexError:
        print(f"Cannot identify the directory to the dataset {dataset_dir}. Skipping this dataset.")
        return []



def _calculate_batch_metrics(pred_actions, gt_actions, action_stats):
    """
    Calculate MSE and MAE metrics for a batch of continuous actions.
    Similar to openx_module.py validation and metrics.
    """
    if action_stats is None:
        raise ValueError("action_stats is required for proper invalid prediction handling")
    
    mses, maes = [], []
    total_invalid_preds = 0
    
    for i in range(len(pred_actions)):
        pred = pred_actions[i]
        gt = np.array(gt_actions[i])
        
        # Validate output
        if _validate_text_output(pred, shape=gt.shape):
            pred_array = np.array([float(item) for item in pred])
            mse = calculate_mse(pred_array, gt)
            mae = calculate_mae(pred_array, gt)
        else:
            # Use worst-case MSE/MAE for invalid predictions
            max_vals = np.array(action_stats['max'])
            min_vals = np.array(action_stats['min'])
            mse = calculate_mse(max_vals[:len(gt)], min_vals[:len(gt)])
            mae = calculate_mae(max_vals[:len(gt)], min_vals[:len(gt)])
            total_invalid_preds += 1
        
        mses.append(mse)
        maes.append(mae)
    
    return mses, maes, total_invalid_preds


def _calculate_final_metrics(timestep_mses, timestep_maes, action_success):
    """
    Calculate comprehensive final metrics for OpenX evaluation.
    Same structure as openx_module.py.
    """
    result = {}
    
    # Calculate MSE metrics
    total_dataset_mse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_mse = total_dataset_mse / num_timesteps if num_timesteps > 0 else 0.0
    
    # Calculate normalized MSE
    if num_timesteps > 1:
        normalized_mses = min_max_normalize(timestep_mses)
        normalized_amse = calculate_mean(normalized_mses)
    else:
        normalized_amse = 0.0
    
    # Calculate MAE metrics
    total_dataset_mae = sum(timestep_maes)
    avg_dataset_mae = calculate_mean(timestep_maes)
    
    if num_timesteps > 1:
        normalized_maes = min_max_normalize(timestep_maes)
        normalized_amae = calculate_mean(normalized_maes)
        
        # Calculate quantile filtered MAE metrics
        quantile_filtered_maes = quantile_filter(timestep_maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
        normalized_quantile_filtered_amae = calculate_mean(normalized_quantile_filtered_maes)
        
        # Calculate additional MAE metrics
        max_rel_mae = calculate_max_relative_mae(timestep_maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
    else:
        normalized_amae = 0.0
        normalized_quantile_filtered_amae = 0.0
        max_rel_mae = 0.0
        prop_beyond_threshold_mae = 0.0
    
    # Calculate action success rate
    action_success_rate = None
    if len(action_success) > 0:
        action_success_rate = (sum(action_success) / len(action_success)) * 100
    
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


def _get_action_space(dataset_name: str, env_name: str, action_stats: dict) -> dict:
    """
    Get action space for the dataset, similar to base_dataset_module.py.
    
    Args:
        dataset_name: Name of the OpenX dataset
        env_name: Environment/task name
        action_stats: Action statistics from the dataloader
    
    Returns:
        Dictionary mapping action indices to tuples describing the action
    """
    # Get definitions from OpenXDefinitions
    descriptions = OpenXDefinitions.DESCRIPTIONS
    action_spaces = OpenXDefinitions.ACTION_SPACES
    
    # Get the action space for this dataset
    if dataset_name in action_spaces:
        if env_name in action_spaces[dataset_name]:
            action_space = action_spaces[dataset_name][env_name]
        else:
            action_space = action_spaces[dataset_name]["default"]
    else:
        # Fallback to using action stats if no verbal description
        action_space = {0: (None,)}
    
    # Handle cases where action space doesn't have verbal description
    if len(action_space) == 1 and action_space.get(0) == (None,):
        # Use action stats to create action space description
        action_space = {}
        for i in range(action_stats['size'][0]):
            action_space[i] = (
                "The action space statistics of this dimension of the action space over the entire dataset",
                action_stats['min'][i],
                action_stats['max'][i],
                action_stats['mean'][i]
            )
    else:
        # Augment verbal descriptions with stats
        for i in range(action_stats['size'][0]):
            if i in action_space and not isinstance(action_space[i], tuple):
                # Convert to tuple if needed
                action_space[i] = (action_space[i],)
            elif i in action_space:
                # Augment existing tuple with stats if needed
                if len(action_space[i]) == 1:
                    # Only verbal description, add stats
                    action_space[i] = (
                        action_space[i][0] + ". In addition to this verbal description, here are the action space statistics of this dimension over the entire dataset",
                        action_stats['min'][i],
                        action_stats['max'][i],
                        action_stats['mean'][i]
                    )
    
    return action_space


def _create_prompt(dataset_name: str, env_name: str, action_stats: dict) -> str:
    """
    Create a prompt using the standard OpenX format from definitions.
    Mimics the _get_vlm_instruction method from base_dataset_module.py.
    
    Args:
        dataset_name: Name of the OpenX dataset
        env_name: Environment/task name from text_observation
        action_stats: Action statistics from the dataloader
    
    Returns:
        Formatted prompt string
    """
    # Get definitions from OpenXDefinitions
    descriptions = OpenXDefinitions.DESCRIPTIONS
    action_exclusiveness = OpenXDefinitions.ACTION_EXCLUSIVENESS
    additional_instructions = OpenXDefinitions.ADDITIONAL_INSTRUCTIONS
    
    # Get environment description
    if dataset_name in descriptions:
        if env_name in descriptions[dataset_name]:
            env_desc = ' '.join(descriptions[dataset_name][env_name])
        else:
            # If env_name not defined, use it as the description
            env_desc = env_name.capitalize() + "."
    else:
        env_desc = env_name.capitalize() + "."
    
    # Get action space with stats augmentation
    action_space = _get_action_space(dataset_name, env_name, action_stats)
    
    # Get action exclusiveness
    if dataset_name in action_exclusiveness:
        if env_name in action_exclusiveness[dataset_name]:
            only_one_action = action_exclusiveness[dataset_name][env_name]
        else:
            only_one_action = action_exclusiveness[dataset_name]["default"]
    else:
        only_one_action = False
    
    # Get additional instructions
    additional_inst = None
    if dataset_name in additional_instructions:
        if env_name in additional_instructions[dataset_name]:
            additional_inst = ' '.join(additional_instructions[dataset_name][env_name])
        else:
            if "default" in additional_instructions[dataset_name]:
                additional_inst = ' '.join(additional_instructions[dataset_name]["default"])
    
    # Use the standard format_instruction_prompt
    prompt = format_instruction_prompt(
        env_name=env_name,
        env_desc=env_desc,
        action_space=action_space,
        only_one_action=only_one_action,
        additional_inst=additional_inst
    )
    
    return prompt


def run_evaluation(args):
    """Run Magma evaluation on OpenX quadrupedal dataset."""
    transformers_logging.set_verbosity_error()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading Magma model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Magma-8B",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    logger.info("Model and processor loaded successfully.")
    
    # Load OpenX quadrupedal data
    logger.info(f"Locating data shards in: {args.dataset_dir}")
    shard_paths = _get_sorted_shard_paths(args.dataset_dir)
    
    if args.num_shards:
        shard_paths = shard_paths[:args.num_shards]
    
    if not shard_paths:
        logger.error(f"No data shards found in {args.dataset_dir}. Exiting.")
        return
    
    logger.info(f"Found {len(shard_paths)} data shards. Creating dataloader...")
    dataset_obj, dataloader = get_openx_dataloader(
        shard_paths,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        by_episode=False
    )
    
    # Get action statistics
    action_stats = dataset_obj.action_stats
    
    # Convert TensorFlow tensors to numpy arrays
    for key in ['min', 'max', 'mean', 'std', 'q01', 'q99']:
        if key in action_stats:
            if tf.is_tensor(action_stats[key]):
                action_stats[key] = action_stats[key].numpy()
            else:
                action_stats[key] = np.array(action_stats[key])
    
    # Ensure 'size' is in action_stats for prompt creation
    if 'size' not in action_stats:
        action_dim = action_stats['min'].shape[0] if hasattr(action_stats['min'], 'shape') else len(action_stats['min'])
        action_stats['size'] = [action_dim]
    else:
        action_dim = action_stats['size'][0]
    
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Action stats - min: {action_stats['min']}, max: {action_stats['max']}")
    
    # Map the simplified dataset name to the full definition name if needed
    # The definitions use "utokyo_saytap_converted_externally_to_rlds"
    definition_dataset_name = args.dataset_name
    if args.dataset_name == "openx_quadrupedal":
        definition_dataset_name = "utokyo_saytap_converted_externally_to_rlds"
    
    total_batches = len(dataloader)
    logger.info(f"Loaded dataloader with {total_batches} batches")
    
    # Generation arguments (similar to Overcooked inference)
    generation_args = {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    
    # Storage for all metrics
    all_mses, all_maes, all_successes = [], [], []
    total_invalid_predictions = 0
    total_timesteps = 0
    
    logger.info(f"Starting evaluation on {total_batches} batches...")
    start_time = time.perf_counter()
    
    for batch_counter, batch in enumerate(dataloader, 1):
        try:
            images = [Image.fromarray(img) for img in batch["image_observation"]]
            instructions = [txt.strip() for txt in batch["text_observation"]]
            gt_actions = batch["action"]
            
            # Process each sample individually
            batch_outputs = []
            
            for idx, (image, instruction) in enumerate(zip(images, instructions)):
                # Create prompt using OpenX format with definitions
                # instruction is the env_name from text_observation
                prompt_text = _create_prompt(definition_dataset_name, instruction, action_stats)
                
                # Format using chat template with system message and image
                convs = [
                    {
                        "role": "system",
                        "content": "You are an agent that can see, talk and act.",
                    },
                    {
                        "role": "user",
                        "content": f"<image>\n{prompt_text}",
                    },
                ]
                
                prompt = processor.tokenizer.apply_chat_template(
                    convs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Handle image tokens
                if hasattr(model.config, 'mm_use_image_start_end') and model.config.mm_use_image_start_end:
                    prompt = prompt.replace("<image>", "<image_start><image><image_end>")
                
                # Process inputs
                inputs = processor(images=image, texts=prompt, return_tensors="pt")
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
                inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                
                # Generate output
                with torch.inference_mode():
                    generate_ids = model.generate(**inputs, **generation_args)
                
                # Extract only the generated tokens (skip input prompt)
                generate_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                output_text = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
                batch_outputs.append(output_text)
                
                # Clean up
                del inputs, generate_ids
                torch.cuda.empty_cache()
            
            # Parse text outputs to lists (action vectors)
            outputs = []
            for output_text in batch_outputs:
                try:
                    # Try to parse as list using ast.literal_eval
                    parsed = ast.literal_eval(output_text.strip())
                    if isinstance(parsed, list):
                        outputs.append(parsed)
                    else:
                        # If not a list, wrap in list
                        outputs.append([parsed])
                except:
                    try:
                        # Try JSON parsing
                        parsed = json.loads(output_text.strip())
                        if isinstance(parsed, list):
                            outputs.append(parsed)
                        else:
                            outputs.append([parsed])
                    except:
                        # Invalid output - will be caught by validation
                        outputs.append(None)
            
            # Calculate metrics using OpenX-style validation
            mses, maes, invalid_preds = _calculate_batch_metrics(outputs, gt_actions, action_stats)
            
            # Accumulate metrics
            all_mses.extend(mses)
            all_maes.extend(maes)
            total_invalid_predictions += invalid_preds
            total_timesteps += len(gt_actions)
            
            # Calculate action success (exact match for continuous actions)
            for i, output in enumerate(outputs):
                if _validate_text_output(output, shape=gt_actions[i].shape):
                    output_array = np.array([float(item) for item in output])
                    if np.array_equal(output_array, np.array(gt_actions[i])):
                        all_successes.append(1)
                    else:
                        all_successes.append(0)
                else:
                    all_successes.append(0)
            
            logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
            logger.info(f"Batch invalid predictions: {invalid_preds}/{len(gt_actions)}")
            
            # Limit samples for testing if specified
            if args.max_samples and total_timesteps >= args.max_samples:
                logger.info(f"Reached max_samples limit: {args.max_samples}")
                break
                
        except Exception as e:
            logger.error(f"Error in batch {batch_counter}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    
    end_time = time.perf_counter()
    eval_time = end_time - start_time
    
    logger.info("Evaluation loop finished. Calculating final metrics...")
    
    # Calculate final metrics using OpenX structure
    final_metrics = _calculate_final_metrics(all_mses, all_maes, all_successes)
    
    # Add evaluation metadata
    final_metrics['eval_time'] = eval_time
    final_metrics['total_invalid_preds'] = total_invalid_predictions
    final_metrics['invalid_predictions_percentage'] = (total_invalid_predictions / total_timesteps * 100) if total_timesteps > 0 else 0.0
    final_metrics['total_batches'] = batch_counter
    final_metrics['total_timesteps'] = total_timesteps
    
    # Save results
    results_file = os.path.join(args.output_dir, args.results_filename)
    with open(results_file, 'w') as f:
        json.dump({args.dataset_name: final_metrics}, f, indent=4)
    
    logger.info(f"Success! Results saved to {results_file}")
    logger.info("\n--- Final Metrics ---")
    logger.info(f"Action Success Rate: {final_metrics['action_success_rate']:.4f}%")
    logger.info(f"Average Dataset AMSE: {final_metrics['avg_dataset_amse']:.6f}")
    logger.info(f"Average Dataset AMAE: {final_metrics['avg_dataset_amae']:.6f}")
    logger.info(f"Normalized AMSE: {final_metrics['normalized_amse']:.6f}")
    logger.info(f"Normalized AMAE: {final_metrics['normalized_amae']:.6f}")
    logger.info(f"Normalized Quantile Filtered AMAE: {final_metrics['normalized_quantile_filtered_amae']:.6f}")
    logger.info(f"Invalid Predictions: {final_metrics['invalid_predictions_percentage']:.2f}%")
    logger.info(f"Total Timesteps: {final_metrics['total_timesteps']}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Magma model evaluation on OpenX quadrupedal dataset (utokyo_saytap)."
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Root directory of the OpenX quadrupedal dataset shards.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save the output results JSON file.'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='openx_quadrupedal',
        help='Name of the dataset being evaluated.'
    )
    parser.add_argument(
        '--results_filename',
        type=str,
        default='magma_openx_quadrupedal_results.json',
        help='Name for the output results file.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for inference.'
    )
    parser.add_argument(
        '--num_shards',
        type=int,
        default=None,
        help='Number of data shards to process. Processes all if not specified.'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing).'
    )
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

