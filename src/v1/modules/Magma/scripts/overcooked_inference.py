# File: magma_overcooked.py
import os
import logging
import time
import types
import json
import gc
import warnings
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

# Suppress warnings to reduce log clutter
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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

class MagmaOvercookedInferenceClass:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.processor = None
        self.dataset = None
        self.dataloader = None
        self.action_stats = {'num_actions': 36}
        self.num_actions = self.action_stats['num_actions']
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_content = None

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
        image_token_index = self.model.config.image_token_index
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[input_ids == image_token_index] = 0
        inputs_embeds = self.model.get_input_embeddings()(for_inputs_embeds_ids)
        if pixel_values is not None and (input_ids == image_token_index).any():
            vision_output = self.model.vision_tower(pixel_values)
            image_features = vision_output['clip_vis_dense']
            b, c, h, w = image_features.shape
            image_features = image_features.flatten(2).transpose(1, 2)
            image_features = self.model.multi_modal_projector(image_features)
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
        outputs = self.model.language_model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = self.model.language_model.lm_head(outputs[0]).float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error("NaN or Inf detected in logits")
        logger.debug(f"Logits shape: {logits.shape}")
        return PatchedMagmaCausalLMOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def load_model_and_processor(self):
        logger.info("Loading model and processor...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Magma-8B",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
            self.model.forward = types.MethodType(self.fixed_magma_forward, self.model)
            logger.info("Final patched forward function has been applied to the model.")
        except Exception as e:
            logger.error(f"Failed to load model or processor: {e}; continuing with dummy outputs")
            self.model = None
            self.processor = None

    def load_dataloader(self):
        logger.info(f"Loading data from: {self.args.data_file}")
        if not os.path.exists(self.args.data_file):
            logger.error(f"No data file found at {self.args.data_file}; continuing with dummy metrics")
            self.dataset, self.dataloader = None, []
        else:
            try:
                self.dataset, self.dataloader = get_overcooked_dataloader(self.args.data_file, self.args.batch_size, num_workers=0, by_episode=False)
                self.action_stats = self.dataset.action_stats
                self.num_actions = self.action_stats['num_actions']
            except Exception as e:
                logger.error(f"Failed to load dataloader: {e}; continuing with dummy metrics")
                self.dataset, self.dataloader = None, []
        logger.info("Action stats calculated and dataloader is ready.")

    def validate_probabilities(self, probs):
        if len(probs) != self.num_actions or np.any(np.isnan(probs)) or np.any(np.isinf(probs)) or not np.isclose(np.sum(probs), 1.0, rtol=1e-5):
            return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)
        return probs

    def calculate_batch_metrics(self, pred_probs, gt_actions):
        one_hot_labels = np.zeros((len(gt_actions), self.num_actions), dtype=np.float32)
        for i, gt in enumerate(gt_actions):
            one_hot_labels[i, int(gt)] = 1.0

        brier_mses, brier_maes, pred_actions = [], [], []
        total_invalid_preds = 0

        for i in range(len(pred_probs)):
            prob = self.validate_probabilities(np.array(pred_probs[i], dtype=np.float32))
            if np.array_equal(prob, np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)):
                total_invalid_preds += 1
                pred_action = -1
            else:
                pred_action = np.argmax(prob)

            try:
                mae = float(calculate_brier_mae(prob, one_hot_labels[i]))
                mse = float(calculate_brier_mse(prob, one_hot_labels[i]))
            except Exception as e:
                logger.warning(f"Error computing MAE/MSE at index {i}: {e}; using worst-case values")
                mae = MAX_BRIER_MAE_ERROR
                mse = MAX_BRIER_MSE_ERROR
                total_invalid_preds += 1
                pred_action = -1

            if np.isnan(mae) or np.isinf(mae):
                mae = MAX_BRIER_MAE_ERROR
            if np.isnan(mse) or np.isinf(mse):
                mse = MAX_BRIER_MSE_ERROR

            brier_maes.append(mae)
            brier_mses.append(mse)
            pred_actions.append(pred_action)

        return brier_mses, brier_maes, total_invalid_preds, pred_actions

    def calculate_final_metrics(self, timestep_mses, timestep_maes, preds, trues, n_invalid_outputs):
        result = {}
        timestep_maes = [float(x) for x in timestep_maes]
        timestep_mses = [float(x) for x in timestep_mses]

        # MAE metrics
        if not timestep_maes or any(np.isnan(x) or np.isinf(x) for x in timestep_maes):
            average_dataset_mae = MAX_BRIER_MAE_ERROR
            average_normalized_mae = MAX_BRIER_MAE_ERROR
            average_normalized_quantile_filtered_mae = MAX_BRIER_MAE_ERROR
            max_rel_mae = MAX_BRIER_MAE_ERROR
            prop_beyond_threshold_mae = 1.0
        else:
            try:
                average_dataset_mae = float(calculate_mean(timestep_maes))
                normalized_maes = min_max_normalize(timestep_maes)
                average_normalized_mae = float(calculate_mean(normalized_maes))
                quantile_filtered_maes = quantile_filter(timestep_maes)
                normalized_quantile_filtered_maes = min_max_normalize(quantile_filtered_maes)
                average_normalized_quantile_filtered_mae = float(calculate_mean(normalized_quantile_filtered_maes))
                max_rel_mae = float(calculate_max_relative_mae(timestep_maes))
                prop_beyond_threshold_mae = float(calculate_proportion_beyond_mae_threshold(timestep_maes))
            except Exception:
                average_dataset_mae = MAX_BRIER_MAE_ERROR
                average_normalized_mae = MAX_BRIER_MAE_ERROR
                average_normalized_quantile_filtered_mae = MAX_BRIER_MAE_ERROR
                max_rel_mae = MAX_BRIER_MAE_ERROR
                prop_beyond_threshold_mae = 1.0

        # MSE metrics
        if not timestep_mses or any(np.isnan(x) or np.isinf(x) for x in timestep_mses):
            total_dataset_amse = len(timestep_mses) * MAX_BRIER_MSE_ERROR
            avg_dataset_amse = MAX_BRIER_MSE_ERROR
            normalized_amse = MAX_BRIER_MSE_ERROR
        else:
            try:
                total_dataset_amse = float(sum(timestep_mses))
                num_timesteps = len(timestep_mses)
                avg_dataset_amse = total_dataset_amse / num_timesteps if num_timesteps > 0 else MAX_BRIER_MSE_ERROR
                normalized_mses = min_max_normalize(timestep_mses)
                normalized_amse = float(calculate_mean(normalized_mses))
            except Exception:
                total_dataset_amse = len(timestep_mses) * MAX_BRIER_MSE_ERROR
                avg_dataset_amse = MAX_BRIER_MSE_ERROR
                normalized_amse = MAX_BRIER_MSE_ERROR

        # Classification metrics
        possible_actions = list(range(self.num_actions))
        try:
            tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(preds, trues, possible_actions)
            precision = float(get_micro_precision_from_counts(tp, fp))
            precision_without_invalid = float(get_micro_precision_from_counts(tp, valid_fp))
            recall = float(get_micro_recall_from_counts(tp, fn))
            f1 = float(get_micro_f1(precision, recall))
            f1_without_invalid = float(get_micro_f1(precision_without_invalid, recall))
            class_precisions = get_precision_per_class(preds, trues, possible_actions)
            class_recalls = get_recall_per_class(preds, trues, possible_actions)
            class_f1s = get_f1_per_class(class_precisions, class_recalls)
            macro_precision = float(get_macro_precision(class_precisions))
            macro_recall = float(get_macro_recall(class_recalls))
            macro_f1 = float(get_macro_f1(class_f1s))
            exact_match_rate = float(get_exact_match_rate(preds, trues))
        except Exception:
            tp = fp = fn = valid_fp = invalid_fp = 0
            precision = precision_without_invalid = recall = f1 = f1_without_invalid = 0.0
            class_precisions = [0.0] * self.num_actions
            class_recalls = [0.0] * self.num_actions
            class_f1s = [0.0] * self.num_actions
            macro_precision = macro_recall = macro_f1 = 0.0
            exact_match_rate = 0.0

        result["exact_match"] = exact_match_rate
        result["total_dataset_amse"] = total_dataset_amse
        result["total_dataset_amae"] = float(sum(timestep_maes))
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

    def generate_dummy_batch_outputs(self, batch_size):
        return (
            [MAX_BRIER_MSE_ERROR] * batch_size,
            [MAX_BRIER_MAE_ERROR] * batch_size,
            [28] * batch_size,
            [0] * batch_size,
            [[1.0 / self.num_actions] * self.num_actions] * batch_size,
            batch_size
        )

    def process_batch(self, batch, batch_counter, total_batches):
        try:
            images = [Image.fromarray(img) for img in batch["image_observation"]]
            env_names = batch["text_observation"]
            time_lefts = batch["time_left"]
            time_elapsed = batch["time_elapsed"]
            gt_actions = [int(gt) if 0 <= int(gt) < self.num_actions else 0 for gt in batch['action']]
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
                self.system_content = inst
                convs = [
                    {"role": "system", "content": self.system_content},
                    {"role": "user", "content": "Here is the image representing the states. Please output probability values for each of the 36 action spaces : <image>"},
                ]
                prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)

            inputs = self.processor(texts=prompts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)
                if torch.isnan(inputs['pixel_values']).any() or torch.isinf(inputs['pixel_values']).any():
                    logger.error("NaN or Inf detected in pixel_values")

            base_input_ids = inputs['input_ids']
            base_attention_mask = inputs['attention_mask']
            pixel_values = inputs.get('pixel_values')
            image_sizes = inputs.get('image_sizes')
            batch_size = base_input_ids.shape[0]

            action_prefix_str = "The action number is: "
            prefix_tokens = self.processor.tokenizer(action_prefix_str, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            base_input_ids = torch.cat([base_input_ids, prefix_tokens.repeat(batch_size, 1)], dim=1)
            base_attention_mask = torch.cat([base_attention_mask, torch.ones(batch_size, prefix_tokens.shape[1], dtype=base_attention_mask.dtype, device=self.device)], dim=1)
            base_len = base_input_ids.shape[1]

            all_logprobs = torch.full((batch_size, self.num_actions), float('-inf'), device=self.device, dtype=torch.float32)
            for action_idx in range(self.num_actions):
                action_str = str(action_idx)
                action_tokens = self.processor.tokenizer(action_str, add_special_tokens=False).input_ids
                action_tokens_tensor = torch.tensor([action_tokens] * batch_size, device=self.device)
                l = action_tokens_tensor.shape[1]
                full_input_ids = torch.cat([base_input_ids, action_tokens_tensor], dim=1)
                full_attention_mask = torch.cat([base_attention_mask, torch.ones(batch_size, l, dtype=base_attention_mask.dtype, device=self.device)], dim=1)
                inputs_dict = {
                    'input_ids': full_input_ids,
                    'attention_mask': full_attention_mask,
                }
                if pixel_values is not None:
                    inputs_dict['pixel_values'] = pixel_values
                if image_sizes is not None:
                    inputs_dict['image_sizes'] = image_sizes
                with torch.inference_mode():
                    outputs = self.fixed_magma_forward(**inputs_dict)
                logits = outputs.logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.error(f"Batch {batch_counter}: NaN or Inf in logits")
                action_logits = logits[:, base_len:base_len + l, :]
                log_probs = torch.log_softmax(action_logits, dim=-1)
                gather_index = action_tokens_tensor.unsqueeze(-1)
                token_logprobs = log_probs.gather(dim=-1, index=gather_index).squeeze(-1)
                sum_logprobs = token_logprobs.sum(dim=-1)
                all_logprobs[:, action_idx] = sum_logprobs

            max_log = all_logprobs.max(dim=-1, keepdim=True)[0]
            exp_log = torch.exp(all_logprobs - max_log)
            sum_exp = exp_log.sum(dim=-1, keepdim=True)
            action_probs = exp_log / sum_exp
            action_probs_np = action_probs.cpu().numpy()

            for i, prob in enumerate(action_probs_np):
                action_probs_np[i] = self.validate_probabilities(prob)

            mses, maes, invalid_preds, pred_actions = self.calculate_batch_metrics(
                action_probs_np, gt_actions
            )

            logger.info(f"--- Processed Batch {batch_counter}/{total_batches} ---")
            print(f"PROBS = {action_probs_np}")
            print(f"Batch {batch_counter}: pred_actions = {pred_actions}")
            print(f"Batch {batch_counter}: ground_truth_actions = {gt_actions}")

            return mses, maes, pred_actions, gt_actions, action_probs_np.tolist(), invalid_preds

        except Exception as e:
            logger.error(f"Error in batch {batch_counter}: {e}; using dummy outputs")
            batch_size = len(batch['action']) if 'action' in batch else self.args.batch_size
            mses, maes, pred_actions, gt_actions, probs, invalid_preds = self.generate_dummy_batch_outputs(batch_size)
            gt_actions = batch['action'] if 'action' in batch else gt_actions
            logger.info(f"--- Processed Batch {batch_counter}/{total_batches} with dummy outputs ---")
            return mses, maes, pred_actions, gt_actions, probs, invalid_preds

    def run_evaluation(self):
        transformers_logging.set_verbosity_error()
        os.makedirs(self.args.output_dir, exist_ok=True)

        self.load_model_and_processor()
        self.load_dataloader()

        all_mses = []
        all_maes = []
        all_preds = []
        all_trues = []
        all_probs = []
        total_invalid_outputs = 0

        if not self.dataloader or not self.model or not self.processor:
            logger.warning("No valid dataloader or model; generating dummy outputs")
            dummy_size = self.args.batch_size
            mses, maes, preds, trues, probs, invalids = self.generate_dummy_batch_outputs(dummy_size)
            all_mses.extend(mses)
            all_maes.extend(maes)
            all_preds.extend(preds)
            all_trues.extend(trues)
            all_probs.extend(probs)
            total_invalid_outputs += invalids
        else:
            total_batches = len(self.dataloader)
            logger.info(f"Starting evaluation on {total_batches} batches...")
            start_time = time.perf_counter()
            for batch_counter, batch in enumerate(self.dataloader, 1):
                print("--------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------")
                mses, maes, pred_actions, gt_actions, probs, invalid_preds = self.process_batch(batch, batch_counter, total_batches)
                all_mses.extend(mses)
                all_maes.extend(maes)
                all_preds.extend(pred_actions)
                all_trues.extend(gt_actions)
                all_probs.extend(probs)
                total_invalid_outputs += invalid_preds
                gc.collect()
                torch.cuda.empty_cache()

            end_time = time.perf_counter()
            logger.info(f"Evaluation time: {end_time - start_time:.2f} seconds")

        logger.info("Evaluation loop finished. Calculating final metrics...")
        try:
            final_metrics = self.calculate_final_metrics(all_mses, all_maes, all_preds, all_trues, total_invalid_outputs)
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
                "class_precisions": [0.0] * self.num_actions,
                "class_recalls": [0.0] * self.num_actions,
                "class_f1s": [0.0] * self.num_actions,
                "total_invalids": int(total_invalid_outputs),
                "percentage_invalids": 100.0,
                "n_invalid_outputs": int(total_invalid_outputs),
                "preds": [int(pred) for pred in all_preds],
                "gt_actions": [int(true) for true in all_trues],
                "action_probabilities": all_probs
            }

        results_file = os.path.join(self.args.output_dir, self.args.results_filename)
        try:
            with open(results_file, 'w') as f:
                json.dump({self.args.dataset_name: final_metrics}, f, indent=4)
            logger.info(f"Success! Detailed results saved to {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}; printing metrics")
            print("\n--- Detailed Metrics ---\n" + json.dumps({self.args.dataset_name: final_metrics}, indent=2))

        return final_metrics