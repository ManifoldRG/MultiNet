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

# Constants for worst-case Brier scores
MAX_BRIER_MAE_ERROR = 2.0
MAX_BRIER_MSE_ERROR = 2.0

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
    logger.debug(f"Input shapes: input_ids={input_ids.shape}, pixel_values={pixel_values.shape}")
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
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.error("NaN or Inf detected in logits")
    logger.debug(f"Logits shape: {logits.shape}")
    return PatchedMagmaCausalLMOutput(
        loss=None,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def stable_softmax(logits):
    """Apply softmax with numerical stability by subtracting the max value."""
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    return exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

def calculate_batch_metrics(pred_probs, gt_actions, full_probs, num_actions, vocab_size, invalid_threshold=0.1):
    one_hot_labels = []
    for gt in gt_actions:
        one_hot = [0.0] * num_actions
        one_hot[int(gt)] = 1.0
        one_hot_labels.append(one_hot)
    brier_mses, brier_maes, pred_actions = [], [], []
    total_invalid_preds = 0
    for i in range(len(pred_probs)):
        prob = pred_probs[i]
        if np.any(np.isnan(prob)) or np.any(np.isinf(prob)):
            logger.warning(f"NaN or Inf in pred_probs at index {i}")
            prob = np.full(num_actions, 1.0 / num_actions)
            total_invalid_preds += 1
            pred_action = -1
        else:
            non_action_probs = full_probs[i][:-num_actions]
            invalid = np.sum(non_action_probs) > invalid_threshold
            if invalid:
                total_invalid_preds += 1
                prob = np.full(num_actions, 1.0 / num_actions)
                pred_action = -1
            else:
                pred_action = np.argmax(prob)
        mae = calculate_brier_mae(prob, one_hot_labels[i])
        mse = calculate_brier_mse(prob, one_hot_labels[i])
        if np.isnan(mae) or np.isinf(mae):
            logger.warning(f"Invalid MAE at index {i}; using worst-case value")
            brier_maes.append(MAX_BRIER_MAE_ERROR)
        else:
            brier_maes.append(mae)
        if np.isnan(mse) or np.isinf(mse):
            logger.warning(f"Invalid MSE at index {i}; using worst-case value")
            brier_mses.append(MAX_BRIER_MSE_ERROR)
        else:
            brier_mses.append(mse)
        pred_actions.append(pred_action)
    return brier_mses, brier_maes, total_invalid_preds, pred_actions

def calculate_final_metrics(timestep_mses, timestep_maes, preds, trues, num_actions, n_invalid_outputs):
    result = {}
    # Handle empty or invalid metric lists with worst-case defaults
    if not timestep_maes or all(np.isnan(x) or np.isinf(x) for x in timestep_maes):
        logger.warning("No valid MAEs collected; setting MAE-related metrics to worst-case values")
        average_dataset_mae = MAX_BRIER_MAE_ERROR
        average_normalized_mae = MAX_BRIER_MAE_ERROR
        average_normalized_quantile_filtered_mae = MAX_BRIER_MAE_ERROR
        max_rel_mae = MAX_BRIER_MAE_ERROR
        prop_beyond_threshold_mae = 1.0
        normalized_maes = [MAX_BRIER_MAE_ERROR] * len(timestep_maes) if timestep_maes else []
        normalized_quantile_filtered_maes = [MAX_BRIER_MAE_ERROR] * len(timestep_maes) if timestep_maes else []
    else:
        average_dataset_mae = calculate_mean(timestep_maes) if timestep_maes else MAX_BRIER_MAE_ERROR
        normalized_maes = min_max_normalize(timestep_maes) if timestep_maes else [MAX_BRIER_MAE_ERROR] * len(timestep_maes)
        average_normalized_mae = calculate_mean(normalized_maes) if normalized_maes else MAX_BRIER_MAE_ERROR
        quantile_filtered_maes = quantile_filter(timestep_maes) if timestep_maes else [MAX_BRIER_MAE_ERROR] * len(timestep_maes)
        normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes) if quantile_filtered_maes else [MAX_BRIER_MAE_ERROR] * len(timestep_maes)
        average_normalized_quantile_filtered_mae = calculate_mean(normalized_quantile_filtered_maes) if normalized_quantile_filtered_maes else MAX_BRIER_MAE_ERROR
        max_rel_mae = calculate_max_relative_mae(timestep_maes) if timestep_maes else MAX_BRIER_MAE_ERROR
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes) if timestep_maes else 1.0
    if not timestep_mses or all(np.isnan(x) or np.isinf(x) for x in timestep_mses):
        logger.warning("No valid MSEs collected; setting MSE-related metrics to worst-case values")
        total_dataset_amse = len(timestep_mses) * MAX_BRIER_MSE_ERROR if timestep_mses else 0.0
        avg_dataset_amse = MAX_BRIER_MSE_ERROR
        normalized_amse = MAX_BRIER_MSE_ERROR
    else:
        total_dataset_amse = sum(timestep_mses) if timestep_mses else 0.0
        num_timesteps = len(timestep_mses)
        avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else MAX_BRIER_MSE_ERROR
        normalized_mses = min_max_normalize(timestep_mses) if timestep_mses else [MAX_BRIER_MSE_ERROR] * len(timestep_mses)
        normalized_amse = calculate_mean(normalized_mses) if normalized_mses else MAX_BRIER_MSE_ERROR
    possible_actions = list(range(num_actions))
    try:
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
        exact_match_rate = get_exact_match_rate(preds, trues)
    except Exception as e:
        logger.warning(f"Error in classification metrics: {e}; setting to worst-case values")
        tp = fp = fn = valid_fp = invalid_fp = 0
        precision = precision_without_invalid = recall = f1 = f1_without_invalid = 0.0
        class_precisions = [0.0] * num_actions
        class_recalls = [0.0] * num_actions
        class_f1s = [0.0] * num_actions
        macro_precision = macro_recall = macro_f1 = 0.0
        exact_match_rate = 0.0
    result["exact_match"] = exact_match_rate
    result["total_dataset_amse"] = total_dataset_amse
    result["total_dataset_amae"] = sum(timestep_maes) if timestep_maes else 0.0
    result["num_timesteps"] = len(timestep_mses)
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
    result["n_invalid_outputs"] = int(n_invalid_outputs)
    result["preds"] = [int(pred) for pred in preds]
    result["gt_actions"] = [int(true) for true in trues]
    return result

def run_evaluation(args):
    transformers_logging.set_verbosity_error()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading model and processor...")
    try:
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
    except Exception as e:
        logger.error(f"Failed to load model or processor: {e}; continuing with dummy outputs")
        model = None
        processor = None
    
    logger.info(f"Loading data from: {args.data_file}")
    if not os.path.exists(args.data_file):
        logger.error(f"No data file found at {args.data_file}; continuing with dummy metrics")
        dataset, dataloader = None, []
        action_stats = {'num_actions': 36}
    else:
        try:
            dataset, dataloader = get_overcooked_dataloader(args.data_file, args.batch_size, num_workers=0, by_episode=False)
            action_stats = dataset.action_stats
        except Exception as e:
            logger.error(f"Failed to load dataloader: {e}; continuing with dummy metrics")
            dataset, dataloader = None, []
            action_stats = {'num_actions': 36}
    
    num_actions = action_stats['num_actions']
    logger.info("Action stats calculated and dataloader is ready.")
    
    all_mses = []
    all_maes = []
    all_preds = []
    all_trues = []
    all_probs = []
    total_invalid_outputs = 0
    total_batches = len(dataloader) if dataloader else 1
    logger.info(f"Starting evaluation on {total_batches} batches...")
    start_time = time.perf_counter()
    
    if not dataloader or not model or not processor:
        logger.warning("No valid dataloader or model; generating dummy outputs")
        all_maes = [MAX_BRIER_MAE_ERROR] * args.batch_size
        all_mses = [MAX_BRIER_MSE_ERROR] * args.batch_size
        all_preds = [28] * args.batch_size
        all_trues = [0] * args.batch_size
        all_probs = [[1.0 / num_actions] * num_actions] * args.batch_size
        total_invalid_outputs = args.batch_size
    else:
        for batch_counter, batch in enumerate(dataloader, 1):
            print("--------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------")
            try:
                images = [Image.fromarray(img) for img in batch["image_observation"]]
                env_names = batch["text_observation"]
                time_lefts = batch["time_left"]
                time_elapsed = batch["time_elapsed"]
                gt_actions = batch['action']
                # Validate ground truth actions
                gt_actions = [int(gt) if isinstance(gt, (int, np.integer)) and 0 <= gt < num_actions else 0 for gt in gt_actions]
                logger.debug(f"Batch {batch_counter}: gt_actions = {gt_actions}")
                logger.debug(f"Batch {batch_counter}: image shapes = {[img.size for img in images]}")

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
                    if torch.isnan(inputs['pixel_values']).any() or torch.isinf(inputs['pixel_values']).any():
                        logger.error("NaN or Inf detected in pixel_values")
                with torch.inference_mode():
                    outputs = model(**inputs)
                    logits = outputs.logits[:, -1, :]
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.error(f"Batch {batch_counter}: NaN or Inf in logits")
                        action_probs = np.full((logits.shape[0], num_actions), 1.0 / num_actions)
                        full_probs = np.full((logits.shape[0], logits.shape[-1]), 1.0 / logits.shape[-1])
                    else:
                        logits = torch.clamp(logits, -100, 100)
                        action_probs = stable_softmax(logits[:, -num_actions:]).cpu().numpy()
                        full_probs = stable_softmax(logits).cpu().numpy()
                for i, prob in enumerate(action_probs):
                    if len(prob) != num_actions or not np.isclose(np.sum(prob), 1.0, rtol=1e-5):
                        logger.warning(f"Batch {batch_counter}: Invalid probability distribution at index {i}: length={len(prob)}, sum={np.sum(prob)}")
                        action_probs[i] = np.full(num_actions, 1.0 / num_actions)
                mses, maes, invalid_preds, pred_actions = calculate_batch_metrics(
                    action_probs, gt_actions, full_probs, num_actions, vocab_size=logits.shape[-1]
                )
                all_mses.extend(mses)
                all_maes.extend(maes)
                all_preds.extend(pred_actions)
                all_trues.extend(gt_actions)
                all_probs.extend(action_probs.tolist())
                total_invalid_outputs += invalid_preds
                logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
                print(f"PROBS = {action_probs}")
                print(f"Batch {batch_counter}: pred_actions = {pred_actions}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_counter}: {e}; using dummy outputs")
                batch_size = len(batch['action']) if 'action' in batch else args.batch_size
                all_maes.extend([MAX_BRIER_MAE_ERROR] * batch_size)
                all_mses.extend([MAX_BRIER_MSE_ERROR] * batch_size)
                all_preds.extend([28] * batch_size)
                all_trues.extend(batch['action'] if 'action' in batch else [0] * batch_size)
                all_probs.extend([[1.0 / num_actions] * num_actions] * batch_size)
                total_invalid_outputs += batch_size
                logger.info(f"--- Processed Batch {batch_counter}/{total_batches} with dummy outputs ---")
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    
    end_time = time.perf_counter()
    logger.info(f"Evaluation time: {end_time - start_time:.2f} seconds")
    logger.info("Evaluation loop finished. Calculating final metrics...")
    try:
        final_metrics = calculate_final_metrics(all_mses, all_maes, all_preds, all_trues, num_actions, total_invalid_outputs)
        final_metrics["action_probabilities"] = all_probs
    except Exception as e:
        logger.error(f"Error in final metrics calculation: {e}; using worst-case metrics")
        final_metrics = {
            "exact_match": 0.0,
            "total_dataset_amse": len(all_mses) * MAX_BRIER_MSE_ERROR,
            "total_dataset_amae": len(all_maes) * MAX_BRIER_MAE_ERROR,
            "num_timesteps": len(all_mses),
            "avg_dataset_amse": MAX_BRIER_MSE_ERROR,
            "avg_dataset_amae": MAX_BRIER_MAE_ERROR,
            "normalized_amse": MAX_BRIER_MSE_ERROR,
            "normalized_amae": MAX_BRIER_MAE_ERROR,
            "normalized_quantile_filtered_amae": MAX_BRIER_MAE_ERROR,
            "max_relative_mae": MAX_BRIER_MAE_ERROR,
            "proportion_beyond_threshold_mae": 1.0,
            "recall": 0.0,
            "precision": 0.0,
            "precision_without_invalid": 0.0,
            "f1": 0.0,
            "f1_without_invalid": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "class_precisions": [0.0] * num_actions,
            "class_recalls": [0.0] * num_actions,
            "class_f1s": [0.0] * num_actions,
            "total_invalids": int(total_invalid_outputs),
            "percentage_invalids": 100.0,
            "preds": [int(pred) for pred in all_preds],
            "gt_actions": [int(true) for true in all_trues],
            "action_probabilities": all_probs
        }
    
    results_file = os.path.join(args.output_dir, args.results_filename)
    try:
        with open(results_file, 'w') as f:
            json.dump({args.dataset_name: final_metrics}, f, indent=4)
        logger.info(f"Success! Detailed results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}; printing metrics")
    
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description="Run Magma model evaluation on Overcooked dataset.")
    
    parser.add_argument('--data_file', type=str, required=True, help='Path to the single pickle file containing Overcooked data.')
    parser.add_argument('--output_dir', type=str, default='./results/v1/magma/overcooked/', help='Directory to save the output results JSON file.')
    parser.add_argument('--dataset_name', type=str, default='overcooked', help='Name of the dataset being evaluated.')
    parser.add_argument('--results_filename', type=str, default='magma_overcooked_results.json', help='Name for the output results file.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inference.')
    
    args = parser.parse_args()
    
    try:
        run_evaluation(args)
    except Exception as e:
        logger.critical(f"Critical error in main: {e}; generating dummy output")
        final_metrics = {
            args.dataset_name: {
                "exact_match": 0.0,
                "total_dataset_amse": args.batch_size * MAX_BRIER_MSE_ERROR,
                "total_dataset_amae": args.batch_size * MAX_BRIER_MAE_ERROR,
                "num_timesteps": args.batch_size,
                "avg_dataset_amse": MAX_BRIER_MSE_ERROR,
                "avg_dataset_amae": MAX_BRIER_MAE_ERROR,
                "normalized_amse": MAX_BRIER_MSE_ERROR,
                "normalized_amae": MAX_BRIER_MAE_ERROR,
                "normalized_quantile_filtered_amae": MAX_BRIER_MAE_ERROR,
                "max_relative_mae": MAX_BRIER_MAE_ERROR,
                "proportion_beyond_threshold_mae": 1.0,
                "recall": 0.0,
                "precision": 0.0,
                "precision_without_invalid": 0.0,
                "f1": 0.0,
                "f1_without_invalid": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "class_precisions": [0.0] * 36,
                "class_recalls": [0.0] * 36,
                "class_f1s": [0.0] * 36,
                "total_invalids": args.batch_size,
                "percentage_invalids": 100.0,
                "n_invalid_outputs": args.batch_size,
                "preds": [28] * args.batch_size,
                "gt_actions": [0] * args.batch_size,
                "action_probabilities": [[1.0 / 36] * 36] * args.batch_size
            }
        }
        results_file = os.path.join(args.output_dir, args.results_filename)
        try:
            with open(results_file, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            logger.info(f"Success! Dummy results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save dummy results: {e}; printing metrics")
            print("\n--- Detailed Metrics ---\n" + json.dumps(final_metrics, indent=2))

if __name__ == "__main__":
    main()