import os
import sys
import argparse
import json
import logging
import re
import time
import types
import gc
from typing import Optional, Tuple
import tensorflow as tf

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
from transformers import logging as transformers_logging
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from src.data_utils.openx_dataloader import get_openx_dataloader
from src.v1.modules.Magma.data.openx.action_tokenizer import ActionTokenizer
from src.v1.modules.Magma.data.openx.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
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

@dataclass
class PatchedMagmaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

def fixed_magma_forward(
    self,
    input_ids: torch.LongTensor,
    pixel_values: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    vision_feature_layer: Optional[int] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
):
    """Simplified forward method for Magma model inference."""
    # Set defaults
    vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else -2
    use_cache = use_cache if use_cache is not None else False
    output_attentions = output_attentions if output_attentions is not None else False
    output_hidden_states = output_hidden_states if output_hidden_states is not None else False
    return_dict = return_dict if return_dict is not None else True
    
    # Extract input embeddings
    image_token_index = self.config.image_token_index
    for_inputs_embeds_ids = input_ids.clone()
    for_inputs_embeds_ids[input_ids == image_token_index] = 0
    inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

    # Process images if present
    if pixel_values is not None and (input_ids == image_token_index).any():
        # Get vision features - the vision tower returns a dict with 'clip_vis_dense' key
        vision_outputs = self.vision_tower(pixel_values)
        image_features = vision_outputs['clip_vis_dense']
        
        # Convert from (B, C, H, W) to (B, H, W, C) format expected by projector
        # This matches the original Magma code pattern: permute(0, 2, 3, 1)
        image_features = image_features.permute(0, 2, 3, 1)  # (B, H, W, C)
        image_features = self.multi_modal_projector(image_features)
        
        # Flatten spatial dimensions to get sequence format (B, H*W, hidden_dim)
        b, h, w, hidden_dim = image_features.shape
        image_features = image_features.view(b, h * w, hidden_dim)

        # Merge image features with text embeddings
        new_inputs_embeds = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == image_token_index)[0]
            
            if len(image_token_indices) == 0:
                new_inputs_embeds.append(inputs_embeds[batch_idx])
                continue
                
            # Handle single image token per sequence
            image_token_idx = image_token_indices[0]
            pre_image_embeds = inputs_embeds[batch_idx, :image_token_idx]
            post_image_embeds = inputs_embeds[batch_idx, image_token_idx + 1:]
            current_image_features = image_features[batch_idx]
            
            # Concatenate embeddings
            full_embeds = torch.cat([
                pre_image_embeds, 
                current_image_features, 
                post_image_embeds
            ], dim=0)
            new_inputs_embeds.append(full_embeds)
            
        # Stack all embeddings
        inputs_embeds = torch.stack(new_inputs_embeds, dim=0)
        
        # Update attention mask and position ids for new sequence length
        new_sequence_length = inputs_embeds.shape[1]
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], new_sequence_length),
            dtype=torch.long,
            device=inputs_embeds.device
        )
        position_ids = torch.arange(
            0, new_sequence_length,
            dtype=torch.long,
            device=inputs_embeds.device
        ).unsqueeze(0).expand(inputs_embeds.shape[0], -1)

    # Forward through language model
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    # Extract logits
    logits = outputs[0] if not return_dict else outputs.logits

    return PatchedMagmaCausalLMOutput(
        loss=None,
        logits=logits,
        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
        attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
    )

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

def _calculate_batch_metrics(pred_actions, gt_actions, action_stats):
    mses, maes = [], []
    for i in range(len(pred_actions)):
        pred, gt = np.array(pred_actions[i]), np.array(gt_actions[i])
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)) or pred.size == 0:
            max_vals, min_vals = np.array(action_stats['max']), np.array(action_stats['min'])
            mse, mae = calculate_mse(max_vals[:len(gt)], min_vals[:len(gt)]), calculate_mae(max_vals[:len(gt)], min_vals[:len(gt)])
        else:
            mse, mae = calculate_mse(pred, gt), calculate_mae(pred, gt)
        mses.append(mse)
        maes.append(mae)
    return mses, maes

def _calculate_final_metrics(mses, maes, successes):
    result = {}
    num_timesteps = len(mses)
    result['num_timesteps'] = num_timesteps
    result['action_success_rate'] = calculate_mean(successes) * 100 if successes else 0.0
    result['avg_dataset_amse'] = calculate_mean(mses)
    result['avg_dataset_amae'] = calculate_mean(maes)
    if num_timesteps > 1:
        result['normalized_amse'] = calculate_mean(min_max_normalize(mses))
        result['normalized_amae'] = calculate_mean(min_max_normalize(maes))
        quantile_filtered_maes = quantile_filter(maes)
        if quantile_filtered_maes:
            result['normalized_quantile_filtered_amae'] = calculate_mean(min_max_normalize(quantile_filtered_maes))
        result['max_relative_mae'] = calculate_max_relative_mae(maes)
        result['proportion_beyond_threshold_mae'] = calculate_proportion_beyond_mae_threshold(maes)
    return result

def unnormalize_action(normalized_action, action_stats):
    action_low, action_high = np.array(action_stats["min"]), np.array(action_stats["max"])
    return 0.5 * (normalized_action + 1) * (action_high - action_low) + action_low

def apply_dataset_transform(actions, dataset_name):
    """
    Apply dataset-specific transforms to predicted actions.
    
    Args:
        actions: numpy array of predicted actions
        dataset_name: name of the dataset to determine which transform to apply
    
    Returns:
        transformed actions as numpy array
    """
    if dataset_name not in OXE_STANDARDIZATION_TRANSFORMS:
        logger.warning(f"No transform found for dataset '{dataset_name}', returning actions unchanged")
        return actions
    
    transform_fn = OXE_STANDARDIZATION_TRANSFORMS[dataset_name]
    
    
    
    transformed_actions = []
    for action in actions:
        try:
            # Most transforms expect actions as a simple tensor, but some might expect dict format
            # We'll try the simple format first, which should work for most cases
            trajectory = {
                "action": tf.constant(action[None, :], dtype=tf.float32),  # Add batch dimension
                "observation": {},  # Empty observation dict
                "language_instruction": ""  # Empty language instruction
            }
            
            # Apply the transform
            transformed_trajectory = transform_fn(trajectory)
            # Extract the transformed action and remove batch dimension
            transformed_action = transformed_trajectory["action"][0].numpy()
            transformed_actions.append(transformed_action)
            
        except Exception as e:
            logger.warning(f"Failed to apply transform for {dataset_name} on action {action}: {e}. Using original action.")
            transformed_actions.append(action)
    
    return np.array(transformed_actions)

def run_evaluation(args):
    transformers_logging.set_verbosity_error()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading model and processor...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Magma-8B",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
    
    model.forward = types.MethodType(fixed_magma_forward, model)
    logger.info("Final patched forward function has been applied to the model.")

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

    for key in ['min', 'max', 'mean', 'std']:
        if key in action_stats:
            action_stats[key] = np.array(action_stats[key])
    
    logger.info("Action stats calculated and dataloader is ready.")
    
    action_dim = action_stats['min'].shape[0]
    generation_args = {"max_new_tokens": action_dim, "temperature": 0.0, "do_sample": False}
    
    all_mses, all_maes, all_successes = [], [], []
    total_batches = len(dataloader)
    logger.info(f"Starting evaluation on {total_batches} batches...")
    start_time = time.perf_counter()
    
    for batch_counter, batch in enumerate(dataloader, 1):
        try:
            images = [Image.fromarray(img) for img in batch["image_observation"]]
            instructions = [txt for txt in batch["text_observation"]]
            gt_actions = np.array([action.numpy() for action in batch['action']])
            
            prompts = [processor.tokenizer.apply_chat_template([{"role": "user", "content": f"<image>\n{inst}"}], tokenize=False, add_generation_prompt=True) for inst in instructions]
            inputs = processor(texts=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

            with torch.inference_mode():
                generate_ids = model.generate(**inputs, **generation_args)

            output_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            action_tokenizer = ActionTokenizer(processor.tokenizer)
            normalized_actions = action_tokenizer.decode_token_ids_to_actions(output_ids.cpu().numpy())

            pred_actions = np.array([unnormalize_action(act, action_stats) for act in normalized_actions])
            pred_actions = apply_dataset_transform(pred_actions, args.dataset_name)

            mses, maes = _calculate_batch_metrics(pred_actions, gt_actions, action_stats)
            successes = [1 if mae < 0.05 else 0 for mae in maes]
            all_mses.extend(mses)
            all_maes.extend(maes)
            all_successes.extend(successes)

            logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
            
        except Exception as e:
            logger.error(f"Error in batch {batch_counter}: {e}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    logger.info("Evaluation loop finished. Calculating final metrics...")
    final_metrics = _calculate_final_metrics(all_mses, all_maes, all_successes)
    
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