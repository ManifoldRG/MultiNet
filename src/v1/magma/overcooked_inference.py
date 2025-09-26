# overcooked_inference.py: Script for running Magma model inference on Overcooked dataset and calculating evaluation metrics.

import os
import sys
import argparse
import json
import logging
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
from src.data_utils.overcooked_dataloader import get_overcooked_dataloader
from src.v1.magma.action_tokenizer import ActionTokenizer
from definitions.overcooked import OverCookedDefinitions
from definitions.overcooked_prompt import format_instruction_prompt
from src.eval_utils import (
    quantile_filter,
    calculate_brier_mae,
    min_max_normalize,
    calculate_brier_mse,
    calculate_mean,
    calculate_max_relative_mae,
    calculate_proportion_beyond_mae_threshold,
    get_exact_match_rate,
    get_micro_precision_from_counts,
    get_micro_recall_from_counts,
    get_micro_f1,
    calculate_tp_fp_fn_counts,
    get_precision_per_class,
    get_recall_per_class,
    get_f1_per_class,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
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
            pre_image_embeds = inputs_embeds[batch_idx, :image_token_idx_in_sequence[0]]
            post_image_embeds = inputs_embeds[batch_idx, image_token_idx_in_sequence[0] + 1:]
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


def calculate_batch_metrics(pred_actions, gt_actions, probs, num_actions):
    one_hot_labels = []
    for gt in gt_actions:
        one_hot = [0.0] * num_actions
        one_hot[int(gt)] = 1.0
        one_hot_labels.append(one_hot)
    brier_mses, brier_maes, preds = [], [], []
    total_invalid_preds = 0
    for i in range(len(pred_actions)):
        prob = probs[i]
        mae = calculate_brier_mae(prob, one_hot_labels[i])
        mse = calculate_brier_mse(prob, one_hot_labels[i])
        brier_maes.append(mae)
        brier_mses.append(mse)
        preds.append(pred_actions[i])
    return brier_mses, brier_maes, total_invalid_preds, preds


def calculate_final_metrics(timestep_mses, timestep_maes, preds, trues, num_actions):
    result = {}
    average_dataset_mae = calculate_mean(timestep_maes)
    normalized_maes = min_max_normalize(timestep_maes)
    average_normalized_mae = calculate_mean(normalized_maes)
    quantile_filtered_maes = quantile_filter(timestep_maes)
    normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
    average_normalized_quantile_filtered_mae = calculate_mean(normalized_quantile_filtered_maes)
    max_rel_mae = calculate_max_relative_mae(timestep_maes)
    prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)
    possible_actions = list(range(num_actions))
    tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(preds, trues, possible_actions)
    precision = get_micro_precision_from_counts(tp, fp)
    precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    f1_without_invalid = get_micro_f1(precision_without_invalid, recall)
    class_precisions = get_precision_per_class(preds, trues, possible_actions)
    class_recalls = get_recall_per_class(preds, trues, possible_actions)
    class_f1s = get_f1_per_class(class_precisions, class_recalls)
    macro_precision = get_macro_precision(class_precisions)
    macro_recall = get_macro_recall(class_recalls)
    macro_f1 = get_macro_f1(class_f1s)
    total_dataset_amse = sum(timestep_mses)
    num_timesteps = len(timestep_mses)
    avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else 0.0
    normalized_mses = min_max_normalize(timestep_mses)
    normalized_amse = calculate_mean(normalized_mses)
    exact_match_rate = get_exact_match_rate(preds, trues)
    result["exact_match"] = exact_match_rate
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
    result["recall"] = recall
    result["precision"] = precision
    result["precision_without_invalid"] = precision_without_invalid
    result["f1"] = f1
    result["f1_without_invalid"] = f1_without_invalid
    result["macro_precision"] = macro_precision
    result["macro_recall"] = macro_recall
    result["macro_f1"] = macro_f1
    result["class_precisions"] = class_precisions
    result["class_recalls"] = class_recalls
    result["class_f1s"] = class_f1s
    result["total_invalids"] = int(invalid_fp)
    result["percentage_invalids"] = (invalid_fp / len(preds)) * 100 if len(preds) > 0 else 0
    result["preds"] = [int(pred) for pred in preds]
    result["gt_actions"] = [int(true) for true in trues]
    return result


class DiscreteActionTokenizer(ActionTokenizer):
    def __init__(self, tokenizer, num_actions: int = 36):
        super().__init__(tokenizer, bins=num_actions, min_action=0, max_action=num_actions - 1)
        self.bins = np.linspace(-0.5, num_actions - 0.5, num_actions + 1)

    def encode_actions_to_token_aids(self, action: np.ndarray) -> np.ndarray:
        discretized = np.digitize(action, self.bins) - 1 
        return self.tokenizer.vocab_size - (self.n_bins - discretized)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        offset = self.tokenizer.vocab_size - action_token_ids
        actions = self.n_bins - offset
        return np.clip(actions, 0, self.n_bins - 1)


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
    logger.info(f"Loading data from: {args.data_file}")
    if not os.path.exists(args.data_file):
        logger.error(f"No data file found at {args.data_file}. Exiting.")
        return
    logger.info("Creating dataloader...")
    dataset, dataloader = get_overcooked_dataloader(args.data_file, args.batch_size, num_workers=0, by_episode=False)
    action_stats = dataset.action_stats
    num_actions = action_stats['num_actions']  # 36 for Overcooked joint actions.
    
    logger.info("Action stats calculated and dataloader is ready.")
    
    action_dim = 1  # Single discrete action index.
    generation_args = {"max_new_tokens": action_dim, "temperature": 0.0, "do_sample": False}
    
    all_mses = []
    all_maes = []
    all_preds = []
    all_trues = []
    total_batches = len(dataloader)
    logger.info(f"Starting evaluation on {total_batches} batches...")
    start_time = time.perf_counter()
    for batch_counter, batch in enumerate(dataloader, 1):
        try:
            images = [Image.fromarray(img) for img in batch["image_observation"]]
            env_names = batch["text_observation"]
            time_lefts = batch["time_left"]
            time_elapsed = batch["time_elapsed"]
            gt_actions = batch['action']
            
            prompts = []
            for i in range(len(env_names)):
                env_name = env_names[i]
                action_meaning = str(OverCookedDefinitions.ACTION_MEANINGS)
                action_space = str(OverCookedDefinitions.ACTION_SPACES['overcooked_ai']['default'])
                additional_inst = OverCookedDefinitions.ADDITIONAL_INSTRUCTIONS.get(env_name, "")
                inst = format_instruction_prompt(
                    env_name,
                    action_meaning,
                    action_space,
                    time_lefts[i],
                    time_elapsed[i],
                    additional_inst
                )
                prompt = processor.tokenizer.apply_chat_template([{"role": "user", "content": f"<image>\n{inst}"}], tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
            inputs = processor(texts=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                action_logits = logits[:, -num_actions:]  # Last 36 logits for actions.
                probs = torch.softmax(action_logits, dim=-1).cpu().numpy()
                generate_ids = model.generate(**inputs, **generation_args)
            output_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            action_tokenizer = DiscreteActionTokenizer(processor.tokenizer, num_actions)
            decoded_actions = action_tokenizer.decode_token_ids_to_actions(output_ids.cpu().numpy())
            pred_actions = np.round(decoded_actions).astype(int).flatten()  # Flatten to 1D array of actions.
            logger.debug(f"Batch {batch_counter}: gt_actions = {gt_actions}")
            logger.debug(f"Batch {batch_counter}: output_ids shape = {output_ids.shape}")
            logger.debug(f"Batch {batch_counter}: decoded_actions = {decoded_actions}")
            logger.debug(f"Batch {batch_counter}: pred_actions = {pred_actions}")

            mses, maes, invalid_preds, batch_preds = calculate_batch_metrics(pred_actions, gt_actions, probs, num_actions)
            all_mses.extend(mses)
            all_maes.extend(maes)
            all_preds.extend(batch_preds)
            all_trues.extend(gt_actions)
            logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
            
        except Exception as e:
            logger.error(f"Error in batch {batch_counter}: {e}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    end_time = time.perf_counter()
    logger.info(f"Evaluation time: {end_time - start_time:.2f} seconds")
    logger.info("Evaluation loop finished. Calculating final metrics...")
    final_metrics = calculate_final_metrics(all_mses, all_maes, all_preds, all_trues, num_actions)
    
    results_file = os.path.join(args.output_dir, args.results_filename)
    with open(results_file, 'w') as f:
        json.dump({args.dataset_name: final_metrics}, f, indent=4)
    logger.info(f"Success! Detailed results saved to {results_file}")
    logger.info("\n--- Detailed Metrics ---\n" + json.dumps(final_metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run Magma model evaluation on Overcooked dataset.")
    
    parser.add_argument('--data_file', type=str, required=True, help='Path to the single pickle file containing Overcooked data.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the output results JSON file.')
    parser.add_argument('--dataset_name', type=str, default='overcooked', help='Name of the dataset being evaluated.')
    parser.add_argument('--results_filename', type=str, default='magma_overcooked_results.json', help='Name for the output results file.')
    
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference.')
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()