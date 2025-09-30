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

from src.data_utils.openx_dataloader import get_openx_dataloader
from src.eval_utils import get_exact_match_rate
from definitions.robovqa_prompt import ROBOVQA_PROMPT



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_shards(dataset_dir: str) -> list[str]:
    try:
        shard_files = glob(f"{dataset_dir}/translated_shard_*")
        tfds_shards = sorted(shard_files, key=lambda x: int(x.split('_')[-1]))
        return tfds_shards
    except IndexError:
        print("Cannot identify the directory to the dataset. Skipping this dataset.")
        return []

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

def _calculate_similarity(pred: str, true: str, similarity_model: SentenceTransformer) -> float:
    emb1 = similarity_model.encode(pred, convert_to_tensor=True)
    emb2 = similarity_model.encode(true, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

def _calculate_final_metrics(preds: List[int], trues: List[int], similarity_model: SentenceTransformer) -> Dict[str, Any]:
    result = {}
    valid_preds, valid_trues = [], []
    invalid_count = 0
    total_samples = len(preds)

    similarity_scores = []
    for pred, true in zip(preds, trues):
        if pred is None:
            invalid_count += 1
            similarity_scores.append(0.0)
        else:
            valid_preds.append(pred)
            valid_trues.append(true)
            similarity_scores.append(_calculate_similarity(pred, true, similarity_model))
    percentage_invalids = (invalid_count / total_samples) * 100 if total_samples > 0 else 0.0

    if len(valid_preds) > 0:
        exact_match_rate = get_exact_match_rate(np.array(valid_preds), np.array(valid_trues))
    else:
        exact_match_rate = 0.0

    exact_match_rate_with_invalids = get_exact_match_rate(np.array(preds), np.array(trues))

    result["avg_similarity_score"] = np.mean(similarity_scores)
    result["max_similarity_score"] = np.max(similarity_scores)
    result["min_similarity_score"] = np.min(similarity_scores)
    result["similarity_std"] = np.std(similarity_scores)
    result["high_similarity_percentage"] = np.mean(similarity_scores >= 0.8)
    result["high_similarity_threshold"] = 0.8
    result["exact_match_rate_without_invalids"] = exact_match_rate
    result["exact_match_rate_with_invalids"] = exact_match_rate_with_invalids
    result["total_predictions"] = total_samples
    result["invalid_predictions"] = invalid_count
    result["percentage_invalids"] = percentage_invalids
    result["preds"] = [p for p in preds]
    result["gt_labels"] = [t for t in trues]

    return result


def main(args):
    logging.info("Starting RoboVQA evaluation script.")
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

    shard_paths = find_shards(args.data_path)
    dataset, dataloader = get_openx_dataloader(
        tfds_shards=shard_paths,
        batch_size=args.batch_size,
        dataset_name="robot_vqa"
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
        for t in range(len(batch['text_observation'])):
            inst = batch['text_observation'][t]
            label = batch["text_answer"][t]
            img = Image.fromarray(batch['image_observation'][t])

            system_prompt = {"role": "system", "content": ROBOVQA_PROMPT}
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

    similarity_model = SentenceTransformer("all-MiniLM-L6-v2", tokenizer_kwargs={"clean_up_tokenization_spaces": True})
    final_report = _calculate_final_metrics(parsed_choices, gt_labels, similarity_model)
    final_report['all_outs'] = raw_predictions
    final_report['eval_time'] = time() - start_time

    output_path = Path(args.output_dir) / args.results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the RoboVQA dataset.")
    parser.add_argument('--model_name_or_path', type=str, default="microsoft/Magma-8B", help="Model identifier.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the RoboVQA test data folder.")
    parser.add_argument('--dtype', type=str, default="bf16", choices=['fp16', 'bf16', 'fp32'], help="Model data type.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference.")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument('--max_new_tokens', type=int, default=75, help="Max new tokens for generation.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling for generation.")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the results file.")
    parser.add_argument('--results_filename', type=str, default="robovqa_results.json", help="Name of the output results file.")
    args = parser.parse_args()
    main(args)
