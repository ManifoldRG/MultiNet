import argparse
import json
import logging
import pprint
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from src.data_utils.piqa_dataloader import get_piqa_dataloader
from src.eval_utils import get_exact_match_rate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _validate_outputs_and_parse(outputs: list) -> list:
    preds = []
    for output in outputs:
        numbers = re.findall(r'\d+', str(output))
        if numbers:
            preds.append(int(numbers[0]))
        else:
            preds.append(-1)
    return preds


def _calculate_final_metrics(preds: List[int], trues: List[int]) -> Dict[str, Any]:
    result = {}
    valid_preds, valid_trues = [], []
    invalid_count = 0

    for pred, true in zip(preds, trues):
        if pred == -1:
            invalid_count += 1
        else:
            valid_preds.append(pred)
            valid_trues.append(true)

    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0

    total_predictions = len(preds)
    correct_predictions = sum(1 for pred, true in zip(preds, trues) if pred == true and pred != -1)
    exact_match_rate_with_invalids = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    result["exact_match_rate"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["total_predictions"] = total_predictions
    result["valid_predictions"] = len(valid_preds)
    result["invalid_predictions"] = invalid_count
    result["percentage_invalids"] = (invalid_count / total_predictions) * 100 if total_predictions > 0 else 0.0
    result["preds"] = [int(p) for p in preds]
    result["gt_labels"] = [int(t) for t in trues]

    return result


def main(args):
    logging.info("Starting PIQA evaluation script.")
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

    dataset, dataloader = get_piqa_dataloader(
        data_file=args.data_path,
        batch_size=args.batch_size,
    )

    raw_predictions, gt_labels = [], []
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }

    for batch in tqdm(dataloader, desc="Running Inference"):
        prompts_text = batch["question"]
        labels = batch["label"]

        chats = [[{"role": "user", "content": p}] for p in prompts_text]

        input_ids = processor.tokenizer.apply_chat_template(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            add_generation_prompt=True,
        )

        input_ids = input_ids.to(model.device)

        with torch.inference_mode():
            generate_ids = model.generate(input_ids=input_ids, **generation_args)
            input_token_len = input_ids.shape[1]
            responses = processor.batch_decode(
                generate_ids[:, input_token_len:],
                skip_special_tokens=True,
            )

        raw_predictions.extend([res.strip() for res in responses])
        gt_labels.extend(labels)

    parsed_choices = _validate_outputs_and_parse(raw_predictions)
    final_report = _calculate_final_metrics(parsed_choices, gt_labels)

    print("\n--- PIQA Evaluation Report ---")
    pprint.pprint(final_report)

    output_path = Path(args.output_dir) / args.results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the PIQA dataset.")
    parser.add_argument('--model_name_or_path', type=str, default="microsoft/phi-2", help="Model identifier.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the PIQA data file (e.g., test.jsonl).")
    parser.add_argument('--dtype', type=str, default="fp16", choices=['fp16', 'bf16', 'fp32'], help="Model data type.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for inference.")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument('--max_new_tokens', type=int, default=5, help="Max new tokens for generation.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling for generation.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the results file.")
    parser.add_argument('--results_filename', type=str, default="piqa_results.json", help="Name of the output results file.")
    args = parser.parse_args()
    main(args)
