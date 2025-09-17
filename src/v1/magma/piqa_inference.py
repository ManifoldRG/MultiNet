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

from src.eval_utils import (
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
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PIQADataset(Dataset):
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.samples = []
        self._load_all_data()

    def _load_all_data(self):
        logging.info(f"Loading PIQA data from {self.data_file}")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(self.samples)} PIQA samples")
    
    def _process_sample(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'goal': sample_data.get('goal', ''),
            'sol1': sample_data.get('sol1', ''),
            'sol2': sample_data.get('sol2', ''),
            'label': sample_data.get('label', 0),
            'sample_id': sample_data.get('id', ''),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._process_sample(self.samples[idx])

def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return result

class MCQMetricsCalculator:
    def __init__(self, num_choices: int = 2):
        self.num_choices = num_choices
        self.valid_labels = list(range(num_choices))

    def _extract_choice(self, output: Any) -> int:
        if isinstance(output, str):
            numbers = re.findall(r'\d+', output.strip())
            if numbers:
                try:
                    choice_val = int(numbers[0])
                    if choice_val == 1: return 0
                    if choice_val == 2: return 1
                    if 0 <= choice_val < self.num_choices: return choice_val
                except ValueError:
                    return -1
        return -1

    def calculate_metrics(self, predictions: List[Any], ground_truth_choices: List[int]) -> Dict[str, Any]:
        if len(predictions) != len(ground_truth_choices):
            raise ValueError("Number of predictions must match number of ground truth choices")

        predicted_choices = np.array([self._extract_choice(p) for p in predictions])
        ground_truth_choices = np.array(ground_truth_choices)
        
        overall_accuracy = get_exact_match_rate(predicted_choices, ground_truth_choices)

        total_tp, total_fp, total_fn, valid_fp, invalid_fp = calculate_tp_fp_fn_counts(
            predicted_choices, ground_truth_choices, self.valid_labels
        )
        micro_precision = get_micro_precision_from_counts(total_tp, total_fp)
        micro_recall = get_micro_recall_from_counts(total_tp, total_fn)
        micro_f1 = get_micro_f1(micro_precision, micro_recall)

        class_precisions = get_precision_per_class(predicted_choices, ground_truth_choices, self.valid_labels)
        class_recalls = get_recall_per_class(predicted_choices, ground_truth_choices, self.valid_labels)
        class_f1s = get_f1_per_class(class_precisions, class_recalls)

        macro_precision = get_macro_precision(class_precisions)
        macro_recall = get_macro_recall(class_recalls)
        macro_f1 = get_macro_f1(class_f1s)

        return {
            'overall_accuracy': overall_accuracy,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'precision_per_class': class_precisions,
            'recall_per_class': class_recalls,
            'f1_per_class': class_f1s,
            'total_samples': len(predictions),
            'total_invalid_preds': np.sum(predicted_choices == -1),
        }

def main(args):
    logging.info("Starting PIQA evaluation script."); pprint.pprint(vars(args))
    
    torch_dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch_dtype)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    dataset = PIQADataset(data_file=args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
    
    raw_predictions, gt_labels = [], []
    generation_args = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "do_sample": args.do_sample}
    
    for batch in tqdm(dataloader, desc="Running Inference"):
        prompts = [
            processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are an expert. Your answer must be only the number of the correct solution, either 1 or 2."},
                 {"role": "user", "content": f"Problem: {goal}\n\nWhich is the correct solution?\n1. {sol1}\n2. {sol2}"}],
                tokenize=False, add_generation_prompt=True
            ) for goal, sol1, sol2 in zip(batch['goal'], batch['sol1'], batch['sol2'])
        ]
        
        inputs = processor(texts=prompts, images=None, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            generate_ids = model.generate(**inputs, **generation_args)
            responses = processor.batch_decode(generate_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        raw_predictions.extend([res.strip() for res in responses])
        gt_labels.extend(batch['label'])
    
    calculator = MCQMetricsCalculator(num_choices=2)
    metrics_report = calculator.calculate_metrics(raw_predictions, gt_labels)
    
    print("\n--- PIQA Evaluation Report ---"); pprint.pprint(metrics_report)
    
    output_path = Path(args.output_dir) / args.results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, indent=4)
    logging.info(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run modular evaluation on the PIQA dataset.")
    parser.add_argument('--model_name_or_path', type=str, default="microsoft/Magma-8B", help="Model identifier.")
    parser.add_argument('--dtype', type=str, default="fp16", choices=['fp16', 'bf16', 'fp32'], help="Model data type.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the PIQA test JSONL file.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for inference. Adjust based on VRAM.")
    parser.add_argument('--max_new_tokens', type=int, default=10, help="Max new tokens for generation.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory for saving results.")
    parser.add_argument('--results_filename', type=str, default="piqa_results.json", help="Output results filename.")
    args = parser.parse_args()
    main(args)
