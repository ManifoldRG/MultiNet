import os
import sys
import re
import time
import gc
import glob
import json
import logging

from dataclasses import dataclass, field, fields
from typing import List

import numpy as np
import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, PreTrainedTokenizerBase

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any

import types
from typing import Optional, Tuple, Union
from transformers.utils import ModelOutput
from transformers.utils import logging as transformers_logging


from src.v1.OpenX_Magma.action_tokenizer import ActionTokenizer
from src.data_utils.openx_dataloader import get_openx_dataloader
from src.eval_utils import (quantile_filter, calculate_mean, min_max_normalize, 
                            calculate_mse, calculate_mae, calculate_max_relative_mae, 
                            calculate_proportion_beyond_mae_threshold)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set format
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Magma-8B",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,   # fp16, much lighter than bf16/fp32
    low_cpu_mem_usage=True,
)

processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)

# Define the float data type
dtype = torch.float16

class ModelConfig:
    GENERATION_ARGS = {"max_new_tokens": 7, "temperature": 0.0, "do_sample": False}

class DatasetConfig:
    RESULTS_FILENAME = 'magma_base_openx_results.json'

@dataclass
class DatasetResults:
    """A dataclass to hold and manage evaluation results for a dataset."""
    total_timesteps: int = 0
    eval_time: float = 0
    total_emr: float = 0
    avg_emr: float = 0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

class MagmaOpenXInference:
    """Manages the inference process for the MAGMA model on OpenX datasets."""
    def __init__(self, model, processor: AutoProcessor, action_tokenizer: ActionTokenizer):
        self.model = model
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.dtype = torch.float16

    def prepare_observation(self, batch: dict) -> list:
        """Prepares a batch of data for the MAGMA model."""
        images = [Image.fromarray(img_array) for img_array in batch["image_observation"]]
        text_obs = [text.strip() for text in batch["text_observation"]]

        inputs_list = []
        for img, instruction in zip(images, text_obs):
            width, height = img.size
            correct_image_size = torch.tensor([[height, width]])

            convs = [
                {"role": "system", "content": "You are agent that can see, talk and act."},
                {"role": "user", "content": f"<image_start><image><image_end>\\n{instruction}"},
            ]
            prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(images=[img], texts=prompt, return_tensors="pt")

            inputs['image_sizes'] = correct_image_size

            inputs_list.append(inputs)
        return inputs_list

    def _cleanup_memory(self):
        """Clean up memory to prevent OOM errors."""
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate_model(self, dataloader: tf.data.Dataset) -> dict:
        """Runs the evaluation loop for the MAGMA model on a given dataloader."""
        counter = 0
        dataset_results = DatasetResults()
        start_time = time.perf_counter()

        total_batches = len(dataloader)

        for batch in dataloader:
            actual_batch_size = len(batch['image_observation'])
            inputs_list = self.prepare_observation(batch)

            processed_actions_list = []
            with torch.inference_mode():
                for inputs in inputs_list:
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)

                    generate_ids = self.model.generate(**inputs, **ModelConfig.GENERATION_ARGS)
                    output_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                    robot_action = self.action_tokenizer.decode_token_ids_to_actions(output_ids.cpu().numpy()[0])
                    processed_actions_list.append(robot_action)

            gt_actions = np.array([action.numpy() for action in batch['action']])
            pred_actions = np.array(processed_actions_list)

            effective_dim = min(pred_actions.shape[1], gt_actions.shape[1])
            pred_clipped = pred_actions[:, :effective_dim]
            gt_clipped = gt_actions[:, :effective_dim]

            mse = np.mean((pred_clipped - gt_clipped) ** 2)
            gt_variance = np.var(gt_clipped) + 1e-8
            pseudo_accuracy = np.exp(-mse / gt_variance)

            dataset_results.total_emr += (pseudo_accuracy * actual_batch_size)
            dataset_results.total_batches += 1
            dataset_results.total_timesteps += actual_batch_size
            counter += 1
            logger.info(f"Processed batch {counter}/{total_batches}")
            self._cleanup_memory()

        dataset_results.eval_time = time.perf_counter() - start_time
        if dataset_results.total_timesteps > 0:
            dataset_results.avg_emr = dataset_results.total_emr / dataset_results.total_timesteps

        return dataset_results.to_dict()
    

def _get_sorted_shard_paths(dataset_dir: str) -> list[str]:
    """Get sorted shard paths using robust regex parsing."""
    shard_pattern = re.compile(r'translated_shard_(\d+)$')
    shard_dirs = []

    try:
        test_dir = os.path.join(dataset_dir, 'test')
        search_dir = test_dir if os.path.exists(test_dir) else dataset_dir

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

OUTPUT_DIR = '/content/results'
DATASET_DIR = '/content/drive/My Drive/'
DATASET_NAME = 'openx'
BATCH_SIZE = 2 
NUM_SHARDS = 1

class MagmaInference:
    def __init__(self, model, processor, action_tokenizer):
        self.model = model
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate_model(self, dataloader: tf.data.Dataset) -> dict:
        counter = 0
        dataset_results = DatasetResults()
        start_time = time.perf_counter()
        total_batches = len(dataloader)

        for batch in dataloader:
            actual_batch_size = len(batch['image_observation'])
            images = [Image.fromarray(img_array) for img_array in batch["image_observation"]]
            instructions = [text.strip() for text in batch["text_observation"]]
            gt_actions = np.array([action.numpy() for action in batch['action']])

            pred_actions_list = []
            for i in range(actual_batch_size):
                image = images[i]
                instruction = instructions[i]

                convs = [
                    {"role": "system", "content": "You are agent that can see, talk and act."},
                    {"role": "user", "content": f"<image_start><image><image_end>\\n{instruction}"},
                ]
                prompt = self.processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(images=[image], texts=prompt, return_tensors="pt")

                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
                inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)

                inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
                inputs['pixel_values'] = inputs['pixel_values'].to(self.dtype)

                with torch.inference_mode():
                    generate_ids = self.model.generate(**inputs, max_new_tokens=128, temperature=0.0)

                output_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                pred_action = self.action_tokenizer.decode_token_ids_to_actions(output_ids.cpu().numpy()[0])
                pred_actions_list.append(pred_action)

            pred_actions = np.array(pred_actions_list)
            effective_dim = min(pred_actions.shape[1], gt_actions.shape[1])
            pred_clipped = pred_actions[:, :effective_dim]
            gt_clipped = gt_actions[:, :effective_dim]

            mse = np.mean((pred_clipped - gt_clipped) ** 2)
            gt_variance = np.var(gt_clipped) + 1e-8
            pseudo_accuracy = np.exp(-mse / gt_variance)

            dataset_results.total_emr += pseudo_accuracy * actual_batch_size
            dataset_results.total_batches += 1
            dataset_results.total_timesteps += actual_batch_size
            counter += 1
            logger.info(f"Processed batch {counter}/{total_batches}")
            self._cleanup_memory()

        dataset_results.eval_time = time.perf_counter() - start_time
        if dataset_results.total_timesteps > 0:
            dataset_results.avg_emr = dataset_results.total_emr / dataset_results.total_timesteps

        return dataset_results.to_dict()

def fixed_magma_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: Union[torch.FloatTensor, List[torch.FloatTensor], List[List[torch.FloatTensor]]] = None,
    image_sizes: Union[torch.LongTensor, List[torch.LongTensor], List[List[torch.LongTensor]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_config['vision_feature_layer']
    )

    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if inputs_embeds is None:
        for_inputs_embeds_ids = input_ids.clone()
        for_inputs_embeds_ids[(input_ids == self.config.image_token_index)] = 0
        inputs_embeds = self.get_input_embeddings()(for_inputs_embeds_ids)

        if pixel_values is not None and input_ids.shape[1] != 1 and len(pixel_values) > 0:
            selected_image_features = [] # FIX 1: Initialize the list
            if type(pixel_values) == list:
                n_imgs_per_sample = [len(pv) for pv in pixel_values]
                pixels_values_list = sum(pixel_values, [])
                image_sizes_list = sum(image_sizes, [])
            else:
                image_num_patches = [(imsize[imsize.sum(1) > 0,0] * imsize[imsize.sum(1) > 0,1]).tolist() for imsize in image_sizes]
                if pixel_values.dim() == 5:
                    _pixel_values_list = [
                        pix_val[:sum(num_patch)].split(num_patch, dim=0) for pix_val, num_patch in zip(pixel_values, image_num_patches)
                    ]
                    _image_sizes_list = [image_size[image_size.sum(-1) > 0].tolist() for image_size in image_sizes]
                elif pixel_values.dim() != 4:
                    raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

            if self.config.vision_config['img_anyres_strategy'] == "global":
                for idx, (image_size_for_instance, pixel_values_for_instance) in enumerate(zip(_image_sizes_list, _pixel_values_list)):
                    for image_size, pixel_values_for_image in zip(image_size_for_instance, pixel_values_for_instance):
                        pass
            elif self.config.vision_config['img_anyres_strategy'] == "crop":
                pass

            if selected_image_features:
                feature_lens = [elem.shape[0] for elem in selected_image_features]
                image_features = torch.cat(selected_image_features, 0)
                feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)

                inputs_embeds, attention_mask, position_ids, labels = self._merge_input_ids_with_image_features(
                    image_features,
                    feature_lens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )
        elif pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) == 0:
            pass
        elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            pass

    @dataclass
    class PatchedMagmaCausalLMOutputWithPast(ModelOutput):
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        past_key_values: Optional[List[torch.FloatTensor]] = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    outputs = self.language_model.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )

    hidden_states = outputs[0]
    logits = self.language_model.lm_head(hidden_states)
    logits = logits.float()
    loss = None
    if labels is not None:
        pass

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return PatchedMagmaCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

model.forward = types.MethodType(fixed_magma_forward, model)

def _calculate_final_metrics(timestep_mses: list[float], timestep_maes: list[float], action_success: list[int]) -> dict:
    """Calculate comprehensive final metrics for OpenX evaluation."""
    result = {}
    num_timesteps = len(timestep_mses)

    total_dataset_mse = sum(timestep_mses)
    avg_dataset_mse = calculate_mean(timestep_mses)
    normalized_amse = 0.0
    if num_timesteps > 1:
        normalized_mses = min_max_normalize(timestep_mses)
        normalized_amse = calculate_mean(normalized_mses)

    total_dataset_mae = sum(timestep_maes)
    avg_dataset_mae = calculate_mean(timestep_maes)
    normalized_amae = 0.0
    normalized_quantile_filtered_amae = 0.0
    max_rel_mae = 0.0
    prop_beyond_threshold_mae = 0.0

    if num_timesteps > 1:
        # Normalized MAE
        normalized_maes = min_max_normalize(timestep_maes)
        normalized_amae = calculate_mean(normalized_maes)

        # Quantile Filtered MAE
        quantile_filtered_maes = quantile_filter(timestep_maes)
        if quantile_filtered_maes: # Check if the list is not empty after filtering
            normalized_quantile_filtered_maes = calculate_mean(min_max_normalize(quantile_filtered_maes))

        # Additional MAE metrics
        max_rel_mae = calculate_max_relative_mae(timestep_maes)
        prop_beyond_threshold_mae = calculate_proportion_beyond_mae_threshold(timestep_maes)

    # --- Action Success Rate ---
    action_success_rate = calculate_mean(action_success) * 100 if action_success else 0.0

    # --- Populate the final result dictionary ---
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

# --- Start of Evaluation ---
transformers_logging.set_verbosity_error()
action_tokenizer = ActionTokenizer(processor.tokenizer)
device = model.device
dtype = torch.float16

all_timestep_mses = []
all_timestep_maes = []
all_action_successes = []

shard_paths = _get_sorted_shard_paths(DATASET_DIR)
if NUM_SHARDS:
    shard_paths = shard_paths[:NUM_SHARDS]

if not shard_paths:
    print(f"Error: No data shards found in {DATASET_DIR}.")
else:
    print(f"Found {len(shard_paths)} real data files.")
    print("Creating the dataloader from real data...")

    dataset, dataloader = get_openx_dataloader(shard_paths, BATCH_SIZE, DATASET_NAME)

    total_batches = len(dataloader)
    print(f"Dataloader created. Starting evaluation on {total_batches} batches.")

    start_time = time.perf_counter()
    batch_counter = 0

    try:
        for batch in dataloader:
            batch_counter += 1
            print(f"\n--- Processing Batch {batch_counter}/{total_batches} ---")

            images = [Image.fromarray(img) for img in batch["image_observation"]]
            instructions = [txt for txt in batch["text_observation"]]
            gt_actions = np.array([action.numpy() for action in batch['action']])

            for i in range(len(images)):
                print(f"  -> Generating output for image {i+1}/{len(images)} in batch {batch_counter}...")
                image, instruction = images[i], instructions[i]

                convs = [{"role": "user", "content": f"<image>\n{instruction}"}]
                prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
                inputs = processor(texts=[prompt], images=[image], return_tensors="pt")
                inputs['image_sizes'] = [torch.tensor([[image.height, image.width]])]

                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else [t.to(device) for t in v] for k, v in inputs.items()}
                if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                    inputs['pixel_values'] = inputs['pixel_values'].to(dtype)

                with torch.inference_mode():
                    generate_ids = model.generate(**inputs, **ModelConfig.GENERATION_ARGS)

                output_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
                pred_action = action_tokenizer.decode_token_ids_to_actions(output_ids.cpu().numpy()[0])

                gt_action = gt_actions[i]
                timestep_mse = np.mean((pred_action - gt_action) ** 2)
                timestep_mae = np.mean(np.abs(pred_action - gt_action))
                action_success = 1 if timestep_mae < 0.05 else 0

                all_timestep_mses.append(timestep_mse)
                all_timestep_maes.append(timestep_mae)
                all_action_successes.append(action_success)

                print("\n" + "="*50)
                print(f"INSTRUCTION: {instruction}")
                print(f"GROUND TRUTH ACTION: {gt_action}")
                print(f"MODEL PREDICTION:    {pred_action}")
                print(f"   (MSE: {timestep_mse:.4f}, MAE: {timestep_mae:.4f}, Success: {action_success})")
                print("="*50 + "\n")

            print(f"--- Finished Batch {batch_counter}/{total_batches} ---")
            gc.collect()
            torch.cuda.empty_cache()

        final_metrics = _calculate_final_metrics(
            all_timestep_mses,
            all_timestep_maes,
            all_action_successes
        )

        results_file = os.path.join(OUTPUT_DIR, DatasetConfig.RESULTS_FILENAME)
        with open(results_file, 'w') as f:
            json.dump({DATASET_NAME: final_metrics}, f, indent=4)

        print("\n--- Detailed Metrics ---")
        print(json.dumps(final_metrics, indent=2))

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()