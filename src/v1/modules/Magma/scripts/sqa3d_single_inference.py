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

project_dir = next(p for p in Path(__file__).parents if p.parts[-1]=='MultiNet')
sys.path.append(str(project_dir))

from src.eval_utils import get_exact_match_rate
from definitions.sqa3d_prompt import SQA3DDefinitions
from src.data_utils.sqa3d_dataloader import get_sqa3d_dataloader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _validate_text_output(output: Any) -> bool:
    """Validate that output is a valid text string."""
    return isinstance(output, str) and len(output.strip()) > 0

def _normalize_text(text: Any) -> str:
    """Normalize text for comparison by removing punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())

    numbers = "zero one two three four five six seven eight nine".split()
    int_numbers = [str(i) for i in range(10)]
    if text in int_numbers:
        text = numbers[int(text)]

    return text

def _validate_outputs_and_parse(outputs: list) -> list:
    preds = []
    for output in outputs:
        if _validate_text_output(output):
            preds.append(_normalize_text(output))
        else:
            preds.append(None)
    return preds

def _calculate_final_metrics(preds: List[int], trues: List[int]) -> Dict[str, Any]:
    result = {}
    valid_preds, valid_trues = [], []
    invalid_count = 0
    total_samples = len(preds)

    for pred, true in zip(preds, trues):
        if pred is None:
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)
    percentage_invalids = (invalid_count / total_samples) * 100 if total_samples > 0 else 0.0

    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0

    exact_match_rate_with_invalids = get_exact_match_rate(np.array(preds), np.array(trues))

    result["exact_match_rate_without_invalids"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["total_predictions"] = total_samples
    result["invalid_predictions"] = invalid_count
    result["percentage_invalids"] = percentage_invalids
    result["preds"] = [p for p in preds]
    result["gt_labels"] = [t for t in trues]

    return result


def main(args):
    logging.info("Starting sqa3d evaluation script.")
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

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    questions_file = Path(args.data_path) / "v1_balanced_questions_test_scannetv2.json"
    annotations_file = Path(args.data_path) / "v1_balanced_sqa_annotations_test_scannetv2.json"

    if not questions_file.exists():
        raise FileNotFoundError(f"Test questions file not found: {questions_file}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Test annotations file not found: {annotations_file}")
    
    images_dir = Path(args.data_path)
    dataset, dataloader = get_sqa3d_dataloader(
        questions_file, annotations_file, images_dir, args.batch_size
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
        answers = batch["answer"]
        scene_images = batch.get("scene_image", [None] * len(questions))

        for t in range(len(questions)):
            img = scene_images[t]
            if img is None:
                continue

            inst = questions[t]
            label = answers[t]
            img = Image.fromarray(img)

            system_prompt = {"role": "system", "content": SQA3DDefinitions.SYSTEM_PROMPT}
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

    parsed_choices = _validate_outputs_and_parse(raw_predictions)
    final_report = _calculate_final_metrics(parsed_choices, gt_labels)
    final_report['all_outs'] = raw_predictions
    final_report['eval_time'] = time() - start_time

    output_path = Path(args.output_dir) / args.results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the sqa3d dataset.")
    parser.add_argument('--model_name_or_path', type=str, default="microsoft/Magma-8B", help="Model identifier.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the sqa3d test data folder.")
    parser.add_argument('--dtype', type=str, default="bf16", choices=['fp16', 'bf16', 'fp32'], help="Model data type.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference.")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument('--max_new_tokens', type=int, default=75, help="Max new tokens for generation.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling for generation.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the results file.")
    parser.add_argument('--results_filename', type=str, default="sqa3d_results.json", help="Name of the output results file.")
    args = parser.parse_args()
    main(args)
