import os
import sys
import argparse
import ast
import json
import logging
import time
import gc
import re

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

from src.data_utils.overcooked_dataloader import get_overcooked_dataloader
from definitions.overcooked import OverCookedDefinitions
from definitions.overcooked_prompt import format_instruction_prompt

from src.eval_utils import (
    quantile_filter,
    calculate_brier_mae,
    calculate_brier_mse,
    min_max_normalize,
    calculate_mean,
    get_exact_match_rate,
    calculate_tp_fp_fn_counts,
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    get_precision_per_class,
    get_recall_per_class,
    get_f1_per_class,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Constants
MAX_BRIER_MAE_ERROR = 2.0
MAX_BRIER_MSE_ERROR = 2.0
NUM_JOINT_ACTIONS = 36  # 6 actions per player, 6*6 = 36 joint actions
NOOP_ACTION = 28


def _validate_list_output(output, num_actions) -> bool:
    """Validate that the output is a valid probability distribution."""
    if not isinstance(output, list):
        return False
    if not len(output) == num_actions:
        return False
    
    try:
        vals = [float(v) for v in output]
    except (ValueError, TypeError):
        return False

    # Check if the sum of the probabilities is 1, avoiding floating point errors
    if abs(sum(vals) - 1.0) > 1e-05:
        return False

    return True


def _get_individual_player_labels(joint_one_hot_label):
    """Get individual player action labels from joint one-hot label."""
    individual_action_space = OverCookedDefinitions.INDIVIDUAL_ACTION_SPACE
    discrete_to_joint = OverCookedDefinitions.PLAYER_ACTION_SPACE_TUPLES

    joint_action_truth = np.argmax(joint_one_hot_label)
    player0_truth_action, player1_truth_action = discrete_to_joint[joint_action_truth]
    player0_label, player1_label = \
        individual_action_space[player0_truth_action], individual_action_space[player1_truth_action]
    return player0_label, player1_label


def _calculate_individual_player_metrics(joint_probs: list, player0_label: int, player1_label: int):
    """Calculate individual player metrics from joint action probabilities."""
    individual_action_space = OverCookedDefinitions.INDIVIDUAL_ACTION_SPACE
    discrete_to_joint = OverCookedDefinitions.PLAYER_ACTION_SPACE_TUPLES
    
    # Convert joint discrete action probs to player0 and player1 discrete action probs
    player0_probs = [0.0] * 6
    player1_probs = [0.0] * 6
    
    for action_idx, prob in enumerate(joint_probs):
        player0_action, player1_action = discrete_to_joint[action_idx]
        player0_probs[individual_action_space[player0_action]] += prob
        player1_probs[individual_action_space[player1_action]] += prob
    
    player0_pred = np.argmax(player0_probs)
    player1_pred = np.argmax(player1_probs)
    
    player0_one_hot_label = [0.0] * 6
    player1_one_hot_label = [0.0] * 6
    player0_one_hot_label[player0_label] = 1.0
    player1_one_hot_label[player1_label] = 1.0
    
    player0_mae = calculate_brier_mae(player0_probs, player0_one_hot_label)
    player1_mae = calculate_brier_mae(player1_probs, player1_one_hot_label)
    player0_mse = calculate_brier_mse(player0_probs, player0_one_hot_label)
    player1_mse = calculate_brier_mse(player1_probs, player1_one_hot_label)
    
    return player0_mae, player0_mse, player0_pred, player1_mae, player1_mse, player1_pred


def _validate_outputs_and_calculate_metrics(outputs, one_hot_labels, num_actions):
    """
    Validate outputs and calculate metrics (same as overcooked_module.py).
    
    Args:
        outputs: List of model outputs (expected to be lists of probabilities)
        one_hot_labels: List of one-hot encoded labels
        num_actions: Number of possible joint actions
    
    Returns:
        Tuple of metrics lists
    """
    brier_mses, brier_maes, preds = [], [], []
    player0_mses, player0_maes, player1_mses, player1_maes = [], [], [], []
    player0_preds, player1_preds = [], []
    player0_trues, player1_trues = [], []
    total_invalid_preds = 0
    
    # Validate outputs and calculate Brier MSEs
    for o, output in enumerate(outputs):
        player0_label, player1_label = _get_individual_player_labels(one_hot_labels[o])

        if _validate_list_output(output, num_actions):
            probs = [float(v) for v in output]

            mae = calculate_brier_mae(probs, one_hot_labels[o])
            brier_maes.append(mae)

            mse = calculate_brier_mse(probs, one_hot_labels[o])
            brier_mses.append(mse)

            preds.append(np.argmax(probs))

            player0_mae, player0_mse, player0_pred, \
                player1_mae, player1_mse, player1_pred = \
                    _calculate_individual_player_metrics(probs, player0_label, player1_label)
            player0_maes.append(player0_mae)
            player1_maes.append(player1_mae)
            player0_mses.append(player0_mse)
            player1_mses.append(player1_mse)
            player0_preds.append(player0_pred)
            player1_preds.append(player1_pred)
        else:
            # max possible Brier MSE is 2.0
            brier_maes.append(MAX_BRIER_MAE_ERROR)
            brier_mses.append(MAX_BRIER_MSE_ERROR)
            player0_maes.append(MAX_BRIER_MAE_ERROR)
            player1_maes.append(MAX_BRIER_MAE_ERROR)
            player0_mses.append(MAX_BRIER_MSE_ERROR)
            player1_mses.append(MAX_BRIER_MSE_ERROR)
            
            total_invalid_preds += 1

            preds.append(-1)
            player0_preds.append(-1)
            player1_preds.append(-1)

        player0_trues.append(player0_label)
        player1_trues.append(player1_label)

    return brier_mses, brier_maes, total_invalid_preds, preds, \
        player0_mses, player0_maes, player0_preds, player0_trues, \
        player1_mses, player1_maes, player1_preds, player1_trues


def _calculate_final_metrics(timestep_mses, timestep_maes, preds, trues, num_actions):
    """Calculate comprehensive final metrics."""
    result = {}
    
    # Calculate MAE metrics
    average_dataset_mae = calculate_mean(timestep_maes)
    normalized_maes = min_max_normalize(timestep_maes)
    average_normalized_mae = calculate_mean(normalized_maes)
    
    # Calculate quantile filtered MAE metrics
    quantile_filtered_maes = quantile_filter(timestep_maes)
    normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
    average_normalized_quantile_filtered_mae = calculate_mean(normalized_quantile_filtered_maes)
    
    max_rel_mae = calculate_max_relative_mae(timestep_maes)
    prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
    
    # Calculate micro metrics
    possible_actions = list(range(num_actions))
    tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(preds, trues, possible_actions)
    
    precision = get_micro_precision_from_counts(tp, fp)
    precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    f1_without_invalid = get_micro_f1(precision_without_invalid, recall)
    
    # Calculate class-wise metrics
    class_precisions = get_precision_per_class(preds, trues, possible_actions)
    class_recalls = get_recall_per_class(preds, trues, possible_actions)
    class_f1s = get_f1_per_class(class_precisions, class_recalls)
    
    # Calculate macro metrics
    macro_precision = get_macro_precision(class_precisions)
    macro_recall = get_macro_recall(class_recalls)
    macro_f1 = get_macro_f1(class_f1s)
    
    # Calculate MSE metrics
    total_dataset_amse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0
    
    normalized_mses = min_max_normalize(timestep_mses)
    normalized_amse = calculate_mean(normalized_mses)
    
    # Exact match rate
    exact_match_rate = get_exact_match_rate(preds, trues)
    
    # Store results
    result["exact_match_rate"] = exact_match_rate
    result["total_dataset_amse"] = total_dataset_amse
    result["total_dataset_amae"] = sum(timestep_maes)
    result["num_timesteps"] = num_timesteps
    result["avg_dataset_amse"] = avg_dataset_amse
    result["avg_dataset_amae"] = average_dataset_mae
    result["normalized_amse"] = normalized_amse
    result["normalized_amae"] = average_normalized_mae
    result["normalized_quantile_filtered_amae"] = average_normalized_quantile_filtered_mae
    result["max_relative_mae"] = max_rel_mae
    result["proportion_beyond_threshold_mae"] = prop_beyond_threshold_mae
    result["micro_recall"] = recall
    result["micro_precision"] = precision
    result["micro_precision_without_invalid"] = precision_without_invalid
    result["micro_f1"] = f1
    result["micro_f1_without_invalid"] = f1_without_invalid
    result["macro_precision"] = macro_precision
    result["macro_recall"] = macro_recall
    result["macro_f1"] = macro_f1
    result["class_precisions"] = class_precisions
    result["class_recalls"] = class_recalls
    result["class_f1s"] = class_f1s
    result["total_invalids"] = int(invalid_fp)
    result["percentage_invalids"] = (invalid_fp / len(preds)) * 100 if len(preds) > 0 else 0.0
    result["preds"] = [int(pred) for pred in preds]
    result["gt_actions"] = [int(true) for true in trues]
    
    return result


def _get_action_space(layout_name: str = "default") -> dict:
    """Get the action space for Overcooked."""
    return OverCookedDefinitions.ACTION_SPACES["overcooked_ai"]["default"]


def _create_prompt(layout_name: str, time_left: float, time_elapsed: float) -> str:
    """
    Create a prompt for the Magma model using the standard Overcooked prompt format.
    
    Args:
        layout_name: Name of the Overcooked layout
        time_left: Time remaining in the game
        time_elapsed: Time elapsed in the game
    
    Returns:
        Formatted prompt string
    """
    dataset = "overcooked_ai"
    
    # Get definitions from OverCookedDefinitions
    descriptions = OverCookedDefinitions.DESCRIPTIONS
    action_exclusiveness = OverCookedDefinitions.ACTION_EXCLUSIVENESS
    additional_instructions = OverCookedDefinitions.ADDITIONAL_INSTRUCTIONS
    
    # Validate that the dataset exists in descriptions (matching overcooked_module.py line 253-255)
    assert (
        dataset in descriptions
    ), f"The layout {dataset} is not included in overcooked."
    
    action_meanings = str(OverCookedDefinitions.ACTION_MEANINGS)
    action_space = _get_action_space()
    
    # Check for additional instructions (matching overcooked_module.py line 262-267)
    additional_inst = None
    if dataset in additional_instructions:
        if layout_name in additional_instructions[dataset]:
            additional_inst = ' '.join(additional_instructions[dataset][layout_name])
        else:
            additional_inst = None
    
    # Use the standard format_instruction_prompt
    prompt = format_instruction_prompt(
        env_name=layout_name,
        action_meaning=action_meanings,
        action_space=str(action_space),
        time_left=time_left,
        time_elapsed=time_elapsed,
        additional_inst=additional_inst,
    )
    
    return prompt


def run_evaluation(args):
    """Run Magma evaluation on Overcooked dataset."""
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
    
    # Load Overcooked data
    logger.info(f"Loading Overcooked data from: {args.data_file}")
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    dataset_obj, dataloader = get_overcooked_dataloader(
        args.data_file,
        batch_size=args.batch_size,
        by_episode=False,
        group_by_layout=True
    )
    
    total_batches = len(dataloader)
    logger.info(f"Loaded dataloader with {total_batches} batches")
    
    # Generation arguments (similar to other Magma scripts)
    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": False,
    }
    
    # Storage for all metrics
    all_mses, all_maes, all_preds, all_trues = [], [], [], []
    all_player0_mses, all_player0_maes, all_player0_preds, all_player0_trues = [], [], [], []
    all_player1_mses, all_player1_maes, all_player1_preds, all_player1_trues = [], [], [], []
    total_invalid_predictions = 0
    total_timesteps = 0
    
    logger.info(f"Starting evaluation on {total_batches} batches...")
    start_time = time.perf_counter()
    
    for batch_counter, batch in enumerate(dataloader, 1):
        try:
            images = [Image.fromarray(img) for img in batch["image_observation"]]
            layout_names = [txt.strip() for txt in batch["text_observation"]]
            labels = batch["action"]
            time_lefts = batch.get("time_left", [0.0] * len(images))
            time_elapseds = batch.get("time_elapsed", [0.0] * len(images))
            
            # Process each sample individually
            batch_outputs = []
            
            for idx, (image, layout_name, time_left, time_elapsed) in enumerate(
                zip(images, layout_names, time_lefts, time_elapseds)
            ):
                # Create prompt for this layout using standard Overcooked prompt
                prompt_text = _create_prompt(layout_name, time_left, time_elapsed)
                
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
                generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
                output_text = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
                batch_outputs.append(output_text)
                
                # Clean up
                del inputs, generate_ids
                torch.cuda.empty_cache()
            
            # Parse text outputs to lists (Magma outputs text, need to parse to list)
            outputs = []
            for output_text in batch_outputs:
                try:
                    # Try to parse as list using ast.literal_eval
                    parsed = ast.literal_eval(output_text.strip())
                    outputs.append(parsed)
                except:
                    try:
                        # Try JSON parsing
                        parsed = json.loads(output_text.strip())
                        outputs.append(parsed)
                    except:
                        # Invalid output
                        outputs.append([None])
            
            # Create one-hot labels
            one_hot_labels = []
            for label in labels:
                one_hot = [0.0] * NUM_JOINT_ACTIONS
                one_hot[label] = 1.0
                one_hot_labels.append(one_hot)
            
            # Same validation and metrics as overcooked_module.py
            brier_mses, brier_maes, invalid_preds, preds, \
                player0_mses, player0_maes, player0_preds, player0_trues, \
                player1_mses, player1_maes, player1_preds, player1_trues = \
                    _validate_outputs_and_calculate_metrics(outputs, one_hot_labels, NUM_JOINT_ACTIONS)
            
            # Accumulate metrics
            all_mses.extend(brier_mses)
            all_maes.extend(brier_maes)
            all_preds.extend(preds)
            all_trues.extend(labels)
            
            all_player0_mses.extend(player0_mses)
            all_player0_maes.extend(player0_maes)
            all_player0_preds.extend(player0_preds)
            all_player0_trues.extend(player0_trues)
            
            all_player1_mses.extend(player1_mses)
            all_player1_maes.extend(player1_maes)
            all_player1_preds.extend(player1_preds)
            all_player1_trues.extend(player1_trues)
            
            total_invalid_predictions += invalid_preds
            total_timesteps += len(labels)
            
            logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
            logger.info(f"Batch invalid predictions: {invalid_preds}/{len(labels)}")
            
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
    
    # Calculate final metrics for joint actions
    final_metrics = _calculate_final_metrics(all_mses, all_maes, all_preds, all_trues, NUM_JOINT_ACTIONS)
    
    # Calculate final metrics for player 0
    player0_metrics = _calculate_final_metrics(
        all_player0_mses, all_player0_maes, all_player0_preds, all_player0_trues, 6
    )
    
    # Calculate final metrics for player 1
    player1_metrics = _calculate_final_metrics(
        all_player1_mses, all_player1_maes, all_player1_preds, all_player1_trues, 6
    )
    
    # Add evaluation metadata
    final_metrics['eval_time'] = eval_time
    final_metrics['total_batches'] = batch_counter
    final_metrics['total_timesteps'] = total_timesteps
    final_metrics['player0_results'] = player0_metrics
    final_metrics['player1_results'] = player1_metrics
    
    # Save results
    results_file = os.path.join(args.output_dir, args.results_filename)
    with open(results_file, 'w') as f:
        json.dump({"overcooked": final_metrics}, f, indent=4)
    
    logger.info(f"Success! Results saved to {results_file}")
    logger.info("\n--- Joint Action Metrics ---")
    logger.info(f"Exact Match Rate: {final_metrics['exact_match_rate']:.4f}")
    logger.info(f"Micro Precision: {final_metrics['micro_precision']:.4f}")
    logger.info(f"Micro Recall: {final_metrics['micro_recall']:.4f}")
    logger.info(f"Micro F1: {final_metrics['micro_f1']:.4f}")
    logger.info(f"Macro F1: {final_metrics['macro_f1']:.4f}")
    logger.info(f"Invalid Predictions: {final_metrics['percentage_invalids']:.2f}%")
    logger.info(f"\n--- Player 0 Metrics ---")
    logger.info(f"Exact Match Rate: {player0_metrics['exact_match_rate']:.4f}")
    logger.info(f"Micro F1: {player0_metrics['micro_f1']:.4f}")
    logger.info(f"\n--- Player 1 Metrics ---")
    logger.info(f"Exact Match Rate: {player1_metrics['exact_match_rate']:.4f}")
    logger.info(f"Micro F1: {player1_metrics['micro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run Magma model evaluation on Overcooked dataset.")
    
    parser.add_argument('--data_file', type=str, required=True, 
                       help='Path to Overcooked pickle data file.')
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Directory to save the output results JSON file.')
    parser.add_argument('--results_filename', type=str, default='magma_overcooked_results.json', 
                       help='Name for the output results file.')
    
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for inference.')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to process (for testing).')
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

