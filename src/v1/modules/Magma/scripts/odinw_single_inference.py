import argparse
import json
import logging
import pprint
import re
import sys

from pathlib import Path
from time import time
from typing import Dict, Any, List
from glob import glob
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from sentence_transformers import SentenceTransformer, util

project_dir = next(p for p in Path(__file__).parents if p.parts[-1]=='MultiNet')
sys.path.append(str(project_dir))

from src.eval_utils import (
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
from definitions.odinw import ODinWDefinitions
from src.data_utils.odinw_dataloader import get_odinw_dataloader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _find_sub_dir(data_path: str, dataset: str) -> str:
    p = f"{data_path}/{dataset}" 
    if Path(p).exists():
        return p
    else:
        return None

def _validate_output(output, possible_outputs) -> bool:
    """Validate that output is a valid integer within the possible outputs"""
    # Handle string outputs that might contain the choice
    if isinstance(output, str):
        try:
            numbers = re.findall(r'\d+', output)
            int_num = int(numbers[0])
            return int_num in possible_outputs
        except Exception:
            return False

    return False


def _validate_outputs_and_parse(outputs: list, possible_outputs: List[int]) -> list:
    preds = []
    for output in outputs:
        if _validate_output(output, possible_outputs):
            numbers = re.findall(r'\d+', output)
            preds.append(int(numbers[0]))
        else:
            preds.append(-1)
    return preds

def _calculate_final_metrics(preds: List[int], trues: List[int], possible_outputs: List[int]) -> Dict[str, Any]:
    result = {}
    valid_preds, valid_trues = [], []
    invalid_count = 0

    for pred, true in zip(preds, trues):
        if pred == -1:
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)

    preds = np.array([int(pred) for pred in preds])
    labels = np.array([int(true) for true in trues])

    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0

     # Calculate metrics counts
    tp, fp, fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
        preds, labels, possible_outputs
    )
    
    precision = get_micro_precision_from_counts(tp, fp)
    precision_without_invalid = get_micro_precision_from_counts(tp, valid_fp)
    recall = get_micro_recall_from_counts(tp, fn)
    f1 = get_micro_f1(precision, recall)
    f1_without_invalid = get_micro_f1(precision_without_invalid, recall)
    
    # Calculate class-wise metrics
    class_precisions = get_precision_per_class(preds, labels, possible_outputs)
    class_recalls = get_recall_per_class(preds, labels, possible_outputs)
    class_f1s = get_f1_per_class(class_precisions, class_recalls)
    
    # Calculate macro metrics
    macro_precision = get_macro_precision(class_precisions)
    macro_recall = get_macro_recall(class_recalls)
    macro_f1 = get_macro_f1(class_f1s)
    exact_match_rate_with_invalids = get_exact_match_rate(np.array(preds), np.array(trues))
    

    result["exact_match_rate_without_invalids"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["recall"] = recall
    result["precision"] = precision
    result["precision_without_invalids"] = precision_without_invalid
    result["f1"] = f1
    result["f1_without_invalids"] = f1_without_invalid
    result["macro_precision"] = macro_precision
    result["macro_recall"] = macro_recall
    result["macro_f1"] = macro_f1
    result["class_precisions"] = class_precisions
    result["class_recalls"] = class_recalls
    result["class_f1s"] = class_f1s
    result["total_predictions"] = len(preds)
    result["invalid_predictions"] = invalid_count
    result["percentage_invalids"] = (invalid_count / len(preds)) * 100
    result["preds"] = [int(p) for p in preds]
    result["gt_labels"] = [int(t) for t in trues]

    return result


def main(args):
    logging.info("Starting odinw evaluation script.")
    pprint.pprint(vars(args))

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    all_datasets = ODinWDefinitions.SUB_DATASET_CATEGORIES.keys() if args.dataset is None else [args.dataset]

    sub_dirs = {}
    for ds in all_datasets:
        sub_dir = _find_sub_dir(args.data_path, ds)
        if sub_dir is not None:
            sub_dirs[ds] = sub_dir

    if not sub_dirs:
        raise FileNotFoundError(f"No test dataset found")

    # Check if the files already exist before running any inference to avoid partial runs
    output_paths = {}
    for ds in sub_dirs.keys():
        output_paths[ds] = Path(args.output_dir) / f"{ds}.json"
        if output_paths[ds].exists():
            raise FileExistsError(f"Results file already exists: {output_paths[ds]}. "
                                  "Specify a different output directory or delete the file.")

    # Run inference for each dataset
    for ds_name, sub_dir in sub_dirs.items():
        _, dataloader = get_odinw_dataloader(
            sub_dir, args.batch_size
        )

        raw_predictions, gt_labels = [], []
        generation_args = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "use_cache": False,
            "past_key_values": None,
        }

        start_time = time()
        for batch in tqdm(dataloader, desc="Running Inference"):
            questions = batch["question"]
            labels = batch["correct_option_idx"]
            images = batch["image"]

            for t in range(len(questions)):
                img = images[t]
                if img is None:
                    continue

                inst = questions[t]
                label = labels[t]
                img = Image.fromarray(img)
        
                system_prompt = {"role": "system", "content": ODinWDefinitions.SYSTEM_PROMPT}
                inst_content = f"<image_start><image><image_end>\n{inst}"
                user_prompt = {"role": "user", "content": inst_content}
                prompt_content = [system_prompt, user_prompt]
            
                prompt = processor.tokenizer.apply_chat_template(prompt_content, tokenize=False, add_generation_prompt=True)
                inputs = processor(images=[img], texts=prompt, return_tensors="pt")
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
                inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
                
                inputs = inputs.to(model.device)
                inputs['pixel_values'] = inputs['pixel_values'].to(torch_dtype)

                with torch.inference_mode():
                    generate_ids = model.generate(**inputs, **generation_args)

                generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
                response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

                raw_predictions.append(response)
                gt_labels.append(label)

        possible_outputs = list(range(ODinWDefinitions.SUB_DATASET_CATEGORIES[ds_name]))
        parsed_choices = _validate_outputs_and_parse(raw_predictions, possible_outputs)
        final_report = _calculate_final_metrics(parsed_choices, gt_labels, possible_outputs)
        final_report['all_outs'] = raw_predictions
        final_report['eval_time'] = time() - start_time

        output_path = output_paths[ds_name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4)
        logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the odinw dataset.")
    parser.add_argument('--model_name_or_path', type=str, default="microsoft/Magma-8B", help="Model identifier.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the odinw test data folder.")
    parser.add_argument('--dataset', type=str, default=None, help="Name of the odinw dataset. Default is all datasets.")
    parser.add_argument('--dtype', type=str, default="bf16", choices=['fp16', 'bf16', 'fp32'], help="Model data type.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference.")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument('--max_new_tokens', type=int, default=75, help="Max new tokens for generation.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling for generation.")
    parser.add_argument('--output_dir', type=str, default="./results/odinw", help="Directory to save the results file.")
    args = parser.parse_args()
    main(args)
