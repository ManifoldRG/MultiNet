#!/usr/bin/env python3
"""
BFCL Inference with Magma Model

This script evaluates the Magma model on the BFCL dataset for multi-turn function-calling tasks.
It generates function call sequences and computes metrics including exact match accuracy,
function-level accuracy, and similarity scores.

Usage:
    python bfcl_magma_inference.py --dataset_dir /path/to/bfcl/test --output_dir ./results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from sentence_transformers import SentenceTransformer, util

# Add project root to path
project_dir = next(p for p in Path(__file__).parents if p.parts[-1] == 'MultiNet')
sys.path.append(str(project_dir))

# Import BFCL dataloader
from src.data_utils.bfcl_dataloader import get_bfcl_test_dataloader

# BFCL system prompt for function calling tasks
BFCL_SYSTEM_PROMPT = """You are an AI assistant that can call functions to complete tasks. You will be presented with multi-turn conversations where each turn may require function calls.

For each turn, analyze the user's request and output the exact sequence of function calls needed for each turn.
Format each function call as: function_name(param1=value1, param2=value2, ...)
Use only the exact function names available in the provided set of functions and append appropriate parameters.
Output only the function calls, no explanations or additional text."""

@dataclass
class DatasetResults:
    """Results from BFCL model inference evaluation"""
    all_exact_matches: List[float] = field(default_factory=list)
    all_similarity_scores: List[float] = field(default_factory=list)
    all_predicted_calls: List[List[str]] = field(default_factory=list)
    all_ground_truth_calls: List[List[List[str]]] = field(default_factory=list)  # Nested: sample -> turn -> calls
    all_original_outputs: List[str] = field(default_factory=list)
    
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_predictions: int = 0
    
    # Final metrics
    exact_match_accuracy: float = 0
    exact_match_accuracy_without_invalids: float = 0
    function_level_accuracy: float = 0
    avg_similarity_score: float = 0
    max_similarity_score: float = 0
    min_similarity_score: float = 0
    similarity_std: float = 0
    high_similarity_percentage: float = 0
    high_similarity_threshold: float = 0.8
    invalid_percentage: float = 0
    total_predicted_functions: int = 0
    total_ground_truth_functions: int = 0
    avg_predicted_functions_per_sample: float = 0
    avg_ground_truth_functions_per_sample: float = 0

    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

def _validate_output(output: str) -> bool:
    """Validate that output contains content."""
    return isinstance(output, str) and bool(output.strip())

def _validate_outputs_and_calculate_metrics(similarity_model: SentenceTransformer, outputs: List[str], ground_truth_list: List[List[List[str]]]):
    """Validate outputs and calculate exact match and similarity metrics for BFCL.
    
    Args:
        similarity_model: SentenceTransformer model for computing similarity scores
        outputs: List of model outputs (one per sample)
        ground_truth_list: List of ground truth function calls per sample.
                          Each sample contains multiple turns, each turn contains multiple function calls.
                          Format: [[[turn1_func1, turn1_func2], [turn2_func1]], ...]
    """
    exact_matches = []
    similarity_scores = []
    predicted_calls_list = []
    total_invalid_preds = 0
    
    for i, output in enumerate(outputs):
        if _validate_output(output):
            # Split output into lines (predicted function calls)
            predicted_calls = [line.strip() for line in output.strip().split('\n') if line.strip()]
            predicted_calls_list.append(predicted_calls)
            
            # Get ground truth for this sample
            gt_turns = ground_truth_list[i] if i < len(ground_truth_list) else []
            
            # Flatten ground truth turns into a single list of function calls
            flattened_gt_calls = []
            for turn_calls in gt_turns:
                if isinstance(turn_calls, list):
                    for call in turn_calls:
                        if isinstance(call, str) and call.strip():
                            flattened_gt_calls.append(call.strip())
                elif isinstance(turn_calls, str) and turn_calls.strip():
                    flattened_gt_calls.append(turn_calls.strip())
            
            # Calculate exact match (order matters for function sequences)
            exact_match = 1.0 if predicted_calls == flattened_gt_calls else 0.0
            exact_matches.append(exact_match)
            
            # For similarity scoring, join function calls into text
            predicted_text = "\n".join(predicted_calls)
            gt_text = "\n".join(flattened_gt_calls)
            
            # Calculate similarity score using sentence embeddings
            try:
                emb1 = similarity_model.encode(predicted_text, convert_to_tensor=True)
                emb2 = similarity_model.encode(gt_text, convert_to_tensor=True)
                similarity = util.cos_sim(emb1, emb2).item()
                similarity_scores.append(similarity)
            except Exception as e:
                print(f"Warning: Similarity calculation failed for sample {i}: {e}")
                similarity_scores.append(0.0)
        else:
            # Invalid output
            exact_matches.append(0.0)
            similarity_scores.append(0.0)
            predicted_calls_list.append([])
            total_invalid_preds += 1
    
    return exact_matches, similarity_scores, predicted_calls_list, total_invalid_preds

def _calculate_final_metrics(exact_matches: List[float], similarity_scores: List[float], total_invalid_preds: int, 
                           predicted_calls: List[List[str]], ground_truth_calls: List[List[List[str]]]) -> Dict[str, Any]:
    """Calculate comprehensive final metrics for BFCL evaluation.
    
    Args:
        exact_matches: List of exact match scores (0.0 or 1.0) per sample
        similarity_scores: List of similarity scores per sample
        total_invalid_preds: Number of invalid predictions
        predicted_calls: List of predicted function calls per sample
        ground_truth_calls: List of ground truth function calls per sample (nested: sample -> turn -> calls)
    """
    result = {}
    
    total_samples = len(exact_matches)
    
    # Calculate accuracy metrics
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
    exact_match_accuracy_without_invalids = sum(exact_matches) / (total_samples - total_invalid_preds) if total_samples - total_invalid_preds > 0 else 0.0
    
    # Flatten ground truth for function-level metrics
    flattened_ground_truth = []
    for sample_gt in ground_truth_calls:
        sample_flattened = []
        for turn_calls in sample_gt:
            if isinstance(turn_calls, list):
                for call in turn_calls:
                    if isinstance(call, str) and call.strip():
                        sample_flattened.append(call.strip())
            elif isinstance(turn_calls, str) and turn_calls.strip():
                sample_flattened.append(turn_calls.strip())
        flattened_ground_truth.append(sample_flattened)
    
    # Calculate function-level metrics
    total_predicted_functions = sum(len(calls) for calls in predicted_calls)
    total_ground_truth_functions = sum(len(calls) for calls in flattened_ground_truth)
    
    # Calculate per-turn metrics
    correct_function_counts = 0
    total_function_comparisons = 0
    
    for pred_calls, gt_calls in zip(predicted_calls, flattened_ground_truth):
        # Count correct functions (regardless of order for this metric)
        for pred_call in pred_calls:
            if pred_call in gt_calls:
                correct_function_counts += 1
        total_function_comparisons += max(len(pred_calls), len(gt_calls))
    
    function_level_accuracy = correct_function_counts / total_function_comparisons if total_function_comparisons > 0 else 0.0
    
    # Calculate similarity metrics
    avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
    max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
    min_similarity_score = min(similarity_scores) if similarity_scores else 0.0
    similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
    
    # Calculate percentage of high similarity matches (threshold-based)
    high_similarity_threshold = 0.8
    high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
    high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0
    
    # Calculate invalid prediction percentage
    invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
    
    result['exact_match_accuracy'] = exact_match_accuracy
    result['exact_match_accuracy_without_invalids'] = exact_match_accuracy_without_invalids
    result['function_level_accuracy'] = function_level_accuracy
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_samples'] = total_samples
    result['total_invalid_preds'] = total_invalid_preds
    result['invalid_percentage'] = invalid_percentage
    result['total_predicted_functions'] = total_predicted_functions
    result['total_ground_truth_functions'] = total_ground_truth_functions
    result['avg_predicted_functions_per_sample'] = total_predicted_functions / total_samples if total_samples > 0 else 0.0
    result['avg_ground_truth_functions_per_sample'] = total_ground_truth_functions / total_samples if total_samples > 0 else 0.0
    result['exact_matches'] = exact_matches
    result['similarity_scores'] = similarity_scores
    result['predicted_function_calls'] = predicted_calls
    result['ground_truth_function_calls'] = flattened_ground_truth
    
    return result

def main(args):
    """Main function to run BFCL inference with Magma model"""
    print(f"Starting BFCL evaluation with Magma model: {args.model_name_or_path}")
    
    # Map data type
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Configure tokenizer
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    # Load similarity model
    print("Loading similarity model for evaluation...")
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    print("✓ Similarity model loaded successfully")

    # Load BFCL dataset
    print("Loading BFCL dataset...")
    dataset_obj, dataloader = get_bfcl_test_dataloader(
        test_dir=args.dataset_dir,
        batch_size=args.batch_size
    )
    print(f"Created dataloader with {len(dataset_obj)} samples")

    # Print dataset info
    dataset_info = dataset_obj.get_dataset_info()
    print("Dataset info:")
    print(f"  Total conversations: {dataset_info['num_conversations']}")
    print(f"  Total turns: {dataset_info['total_turns']}")
    print(f"  Average turns per conversation: {dataset_info['avg_turns_per_conversation']:.2f}")
    print(f"  Unique tool classes: {dataset_info['num_tool_classes']}")

    # Initialize results
    dataset_results = DatasetResults()
    start_time = time.perf_counter()

    # Generation arguments
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
        ground_truth_functions = batch["ground_truth_functions"]
        user_turns_batch = batch["turns"]
        initial_user_prompts = batch["prompt"]
        batch_size = len(user_turns_batch)

        # Initialize histories for each sample in the batch
        batch_chat_histories = [
            [
                {"role": "system", "content": BFCL_SYSTEM_PROMPT}, 
                {"role": "user", "content": initial_user_prompts[i]}
            ] for i in range(batch_size)
        ]
        # Store all predicted calls for each sample
        batch_all_predicted_calls = [[] for _ in range(batch_size)]

        # Determine the number of turns to process (max in batch)
        num_turns = max(len(s) for s in user_turns_batch)

        for turn_idx in range(num_turns):
            turn_batch_inputs = []
            
            # Prepare inputs for the current turn for all samples in the batch
            for i in range(batch_size):
                # Check if the current sample has this many turns
                if turn_idx < len(user_turns_batch[i]):
                    # Add user messages for the current turn
                    user_messages_for_turn = user_turns_batch[i][turn_idx]
                    batch_chat_histories[i].extend(user_messages_for_turn)
                
                turn_batch_inputs.append(batch_chat_histories[i])

            
            # Tokenize the batch of conversations for the current turn
            input_ids = processor.tokenizer.apply_chat_template(
                turn_batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
                add_generation_prompt=True,
            ).to(model.device)

            # Generate responses for the batch
            with torch.inference_mode():
                generate_ids = model.generate(input_ids=input_ids, **generation_args)
                input_token_len = input_ids.shape[1]
                turn_responses = processor.batch_decode(
                    generate_ids[:, input_token_len:],
                    skip_special_tokens=True,
                )

            # Process outputs and update histories for the next turn
            for i in range(batch_size):
                if turn_idx < len(user_turns_batch[i]):
                    response_text = turn_responses[i].strip()
                    
                    # Parse the generated string into a list of function calls
                    predicted_calls_for_turn = [line.strip() for line in response_text.split('\n') if line.strip()]
                    batch_all_predicted_calls[i].append(predicted_calls_for_turn)
                    
                    # Add the model's output to the history as the assistant's turn
                    if response_text: # Only add assistant message if something was generated
                        batch_chat_histories[i].append({"role": "assistant", "content": response_text})

        # Consolidate all predicted calls for the batch to match metric function input
        final_batch_outputs = []
        final_batch_predicted_calls = []
        for sample_predicted_turns in batch_all_predicted_calls:
            # Flatten the list of lists of calls into a single list
            flattened_calls = [call for turn_calls in sample_predicted_turns for call in turn_calls]

            # [["func1", "func2"], ["func3"]] -> ["func1", "func2", "func3"]
            final_batch_predicted_calls.append(flattened_calls)
            
            # Join into a single string for validation and similarity scoring
            # "func1\nfunc2\nfunc3"
            final_batch_outputs.append("\n".join(flattened_calls))

        # Validate and calculate metrics for the completed batch
        exact_matches, similarity_scores, _, invalid_preds = _validate_outputs_and_calculate_metrics(
            similarity_model, final_batch_outputs, ground_truth_functions
        )

        # Update results
        dataset_results.total_batches = batch_idx + 1
        dataset_results.total_samples += len(user_turns_batch)
        dataset_results.all_exact_matches.extend(exact_matches)
        dataset_results.all_similarity_scores.extend(similarity_scores)
        dataset_results.all_predicted_calls.extend(final_batch_predicted_calls)
        dataset_results.all_ground_truth_calls.extend(ground_truth_functions)
        dataset_results.all_original_outputs.extend(batch_all_predicted_calls)
        dataset_results.total_invalid_predictions += invalid_preds

        # Progress update
        if (batch_idx + 1) % 10 == 0:
            current_accuracy = sum(dataset_results.all_exact_matches) / len(dataset_results.all_exact_matches) if dataset_results.all_exact_matches else 0.0
            current_similarity = sum(dataset_results.all_similarity_scores) / len(dataset_results.all_similarity_scores) if dataset_results.all_similarity_scores else 0.0
            print(f"Progress: {batch_idx + 1} batches processed. Current accuracy: {current_accuracy:.4f}, Current avg similarity: {current_similarity:.4f}")

        # Check for max_samples limit
        if args.max_samples is not None and dataset_results.total_samples >= args.max_samples:
            print(f"Reached maximum sample limit of {args.max_samples}. Stopping evaluation.")
            break

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate final metrics
    final_metrics = _calculate_final_metrics(
        dataset_results.all_exact_matches,
        dataset_results.all_similarity_scores,
        dataset_results.total_invalid_predictions,
        dataset_results.all_predicted_calls,
        dataset_results.all_ground_truth_calls
    )

    # Update dataset results
    dataset_results.exact_match_accuracy = final_metrics["exact_match_accuracy"]
    dataset_results.exact_match_accuracy_without_invalids = final_metrics["exact_match_accuracy_without_invalids"]
    dataset_results.function_level_accuracy = final_metrics["function_level_accuracy"]
    dataset_results.avg_similarity_score = final_metrics["avg_similarity_score"]
    dataset_results.max_similarity_score = final_metrics["max_similarity_score"]
    dataset_results.min_similarity_score = final_metrics["min_similarity_score"]
    dataset_results.similarity_std = final_metrics["similarity_std"]
    dataset_results.high_similarity_percentage = final_metrics["high_similarity_percentage"]
    dataset_results.high_similarity_threshold = final_metrics["high_similarity_threshold"]
    dataset_results.invalid_percentage = final_metrics["invalid_percentage"]
    dataset_results.total_predicted_functions = final_metrics["total_predicted_functions"]
    dataset_results.total_ground_truth_functions = final_metrics["total_ground_truth_functions"]
    dataset_results.avg_predicted_functions_per_sample = final_metrics["avg_predicted_functions_per_sample"]
    dataset_results.avg_ground_truth_functions_per_sample = final_metrics["avg_ground_truth_functions_per_sample"]
    dataset_results.eval_time = time.perf_counter() - start_time

    # Print summary
    print("\n=== BFCL Magma Inference Results Summary ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Device: {model.device}")
    print(f"Total samples: {dataset_results.total_samples}")
    print(f"Exact Match Accuracy: {dataset_results.exact_match_accuracy:.4f}")
    print(f"Exact Match Accuracy (without invalids): {dataset_results.exact_match_accuracy_without_invalids:.4f}")
    print(f"Function-Level Accuracy: {dataset_results.function_level_accuracy:.4f}")
    print(f"Average Similarity Score: {dataset_results.avg_similarity_score:.4f}")
    print(f"Max Similarity Score: {dataset_results.max_similarity_score:.4f}")
    print(f"Min Similarity Score: {dataset_results.min_similarity_score:.4f}")
    print(f"Similarity Std Dev: {dataset_results.similarity_std:.4f}")
    print(f"High Similarity (≥{dataset_results.high_similarity_threshold}): {dataset_results.high_similarity_percentage:.2f}%")
    print(f"Invalid predictions: {dataset_results.total_invalid_predictions} ({dataset_results.invalid_percentage:.2f}%)")
    print(f"Average predicted functions per sample: {dataset_results.avg_predicted_functions_per_sample:.2f}")
    print(f"Average ground truth functions per sample: {dataset_results.avg_ground_truth_functions_per_sample:.2f}")
    print(f"Evaluation time: {dataset_results.eval_time:.2f} seconds")

    # Save results
    output_path = Path(args.output_dir) / args.results_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_results.to_dict(), f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on the BFCL dataset using Magma model.")
    parser.add_argument('--model_name_or_path', type=str, default="microsoft/Magma-8B", help="Model identifier.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory containing the BFCL test dataset.")
    parser.add_argument('--dtype', type=str, default="bf16", choices=['fp16', 'bf16', 'fp32'], help="Model data type.")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for inference.")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument('--max_new_tokens', type=int, default=150, help="Max new tokens for generation.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Generation temperature.")
    parser.add_argument('--do_sample', action='store_true', help="Enable sampling for generation.")
    parser.add_argument('--output_dir', type=str, default="./bfcl_magma_results", help="Directory to save the results file.")
    parser.add_argument('--results_filename', type=str, default="bfcl_results.json", help="Name of the output results file.")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to process (default: all samples)")
    args = parser.parse_args()
    
    # Validate dataset directory
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    main(args)