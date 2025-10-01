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

from src.data_utils.openx_dataloader import get_openx_dataloader
from src.v1.magma.action_tokenizer import ActionTokenizer
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
    **kwargs,
):
    if pixel_values.dim() == 4 and pixel_values.shape[3] == 3:
         pixel_values = pixel_values.permute(0, 3, 1, 2)

    image_token_index = self.config.image_token_index
    for_inputs_embeds_ids = input_ids.clone()
    for_inputs_embeds_ids[input_ids == image_token_index] = 0
    inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

    if pixel_values is not None and (input_ids == image_token_index).any():
        vision_output = self.vision_tower(pixel_values)
        image_features = vision_output['clip_vis_dense']
        b, c, h, w = image_features.shape
        image_features = image_features.flatten(2).transpose(1, 2)
        image_features = self.multi_modal_projector(image_features)

        new_inputs_embeds = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_idx_in_sequence = torch.where(cur_input_ids == image_token_index)[0]
            pre_image_embeds = inputs_embeds[batch_idx, :image_token_idx_in_sequence]
            post_image_embeds = inputs_embeds[batch_idx, image_token_idx_in_sequence + 1:]
            current_image_features = image_features[batch_idx]
            if current_image_features.dim() == 3:
                current_image_features = current_image_features.squeeze(0)
            full_embeds = torch.cat(
                [pre_image_embeds, current_image_features, post_image_embeds],
                dim=0
            )
            new_inputs_embeds.append(full_embeds)
        inputs_embeds = torch.stack(new_inputs_embeds, dim=0)

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
        ).unsqueeze(0)

    outputs = self.language_model.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
    )
    logits = self.language_model.lm_head(outputs[0]).float()

    return PatchedMagmaCausalLMOutput(
        loss=None,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
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

def invert_gripper_action(action):
    action[..., -1] *= -1.0
    return action

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
            pred_actions = np.array([invert_gripper_action(act) for act in pred_actions])

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