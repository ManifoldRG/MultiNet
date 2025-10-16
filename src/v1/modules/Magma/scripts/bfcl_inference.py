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
import re
import sys
import time
from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
BFCL_SYSTEM_PROMPT = """You are an AI assistant that can call functions to complete tasks. You will be presented with conversation histories where each turn may require function calls.                                                          
                                                                                                                                                                                                                                                  
For each turn, analyze the conversation history, which may include previous assistant responses in addition to user prompts, and respond with the correct function to call.                                                                       
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
    all_extracted_function_calls: List[List[List[str]]] = field(default_factory=list)
    all_full_responses: List[List[str]] = field(default_factory=list)
    # Accuracy for each turn
    all_turn_exact_matches: List[List[float]] = field(default_factory=list)

    # Within a turn, counts exact matched function until the first time the function 
    # does not equal to the ground truth function
    all_turn_recalls_until_failure: List[List[float]] = field(default_factory=list)
    total_batches: int = 0
    total_samples: int = 0
    eval_time: float = 0
    total_invalid_turns: int = 0
    total_invalid_conversations: int = 0
    
    # Final metrics
    exact_match_accuracy: float = 0
    avg_similarity_score: float = 0
    max_similarity_score: float = 0
    min_similarity_score: float = 0
    similarity_std: float = 0
    high_similarity_percentage: float = 0
    high_similarity_threshold: float = 0.8
    invalid_turn_percentage: float = 0
    invalid_conversation_percentage: float = 0
    total_predicted_functions: int = 0
    total_ground_truth_functions: int = 0
    avg_valid_predicted_functions_per_sample: float = 0
    avg_ground_truth_functions_per_sample: float = 0
    turn_level_accuracy: Dict[str, float] = field(default_factory=dict)

    # Average turn of first failure
    avg_turn_of_first_failure: float = 0
    turn_level_recalls_until_failure: Dict[str, float] = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

def _extract_function_calls(text: str) -> List[str]:
    pattern = r'\b\w+\s*\([^)]*\)'
    return re.findall(pattern, text)

def _validate_outputs_and_calculate_metrics(
    similarity_model: SentenceTransformer,
    predicted_calls_per_sample: List[List[List[str]]],
    ground_truth_per_sample: List[List[List[str]]],
    full_responses_per_sample: List[List[str]]
) -> Tuple[List[float], List[List[float]], List[float], int]:
    """
    Calculates conversation-level and turn-level metrics, and counts empty predictions.

    Args:
        similarity_model: SentenceTransformer model for computing similarity.
        predicted_calls_per_sample: A batch of predictions, structured as [sample][turn][calls].
        ground_truth_per_sample: A batch of ground truths, structured as [sample][turn][calls].

    Returns:
        A tuple containing:
        - all_convo_exact_matches: List of 1.0/0.0 scores for each conversation.
        - all_turn_exact_matches: List of lists with 1.0/0.0 scores for each turn.
        - all_convo_similarity_scores: List of similarity scores for each conversation.
        - total_invalid_turns: Count of turns where model predicted nothing but should have.
    """
    all_convo_exact_matches = []
    all_turn_exact_matches = []
    all_convo_similarity_scores = []
    all_turn_recalls_until_failure = []
    total_invalid_turns = 0 
    total_invalid_conversations = 0

    for i, predicted_turns in enumerate(predicted_calls_per_sample):
        gt_turns = ground_truth_per_sample[i]

        # Conversation-Level Metrics
        flattened_preds = [call for turn in predicted_turns for call in turn]
        flattened_gt = [call for turn in gt_turns for call in turn]
        
        convo_exact_match = 1.0 if flattened_preds == flattened_gt else 0.0
        all_convo_exact_matches.append(convo_exact_match)

        predicted_text = "\n".join(full_responses_per_sample[i])
        gt_text = "\n".join(flattened_gt)
        try:
            emb1 = similarity_model.encode(predicted_text, convert_to_tensor=True)
            emb2 = similarity_model.encode(gt_text, convert_to_tensor=True)
            all_convo_similarity_scores.append(util.cos_sim(emb1, emb2).item())
        except Exception:
            all_convo_similarity_scores.append(0.0)

        # Turn-Level Metrics
        turn_matches = []
        turn_recalls_until_failure = []
        num_turns = max(len(predicted_turns), len(gt_turns))

        invalid_conversation = False
        for turn_idx in range(num_turns):
            pred_calls_for_turn = predicted_turns[turn_idx] if turn_idx < len(predicted_turns) else []
            gt_calls_for_turn = gt_turns[turn_idx] if turn_idx < len(gt_turns) else []
            
            # Prediction is empty but ground truth is not.
            if not pred_calls_for_turn and gt_calls_for_turn:
                total_invalid_turns += 1
                invalid_conversation = True
            
            turn_exact_match = 1.0 if pred_calls_for_turn == gt_calls_for_turn else 0.0
            turn_matches.append(turn_exact_match)

            function_matches = 0
            for pred_call, gt_call in zip(pred_calls_for_turn, gt_calls_for_turn):
                if pred_call == gt_call:
                    function_matches += 1
                else:
                    break
            turn_recalls_until_failure.append(function_matches / len(gt_calls_for_turn))
        
        all_turn_exact_matches.append(turn_matches)
        all_turn_recalls_until_failure.append(turn_recalls_until_failure)
        if invalid_conversation:
            total_invalid_conversations += 1
    return all_convo_exact_matches, all_turn_exact_matches, all_convo_similarity_scores, total_invalid_turns, all_turn_recalls_until_failure, total_invalid_conversations
    
def _calculate_final_metrics(
    exact_matches: List[float], 
    similarity_scores: List[float], 
    predicted_calls: List[List[List[str]]], 
    ground_truth_calls: List[List[List[str]]],
    all_turn_exact_matches: List[List[float]],
    total_invalid_turns: int,
    all_turn_recalls_until_failure: List[List[float]],
    total_invalid_conversations: int
) -> Dict[str, Any]:
    """Calculate comprehensive final metrics for BFCL evaluation."""
    result = {}
    total_samples = len(exact_matches)
    
    # Calculate accuracy metrics
    exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0

    # Calculate function-level metrics
    flattened_predicted_calls = []
    for sample_pred in predicted_calls:
        sample_flattened = []
        for turn_calls in sample_pred:
            sample_flattened.extend(turn_calls)
        flattened_predicted_calls.append(sample_flattened)
    total_predicted_functions = sum(len(calls) for calls in flattened_predicted_calls)

    flattened_ground_truth = []
    for sample_gt in ground_truth_calls:
        sample_flattened = []
        for turn_calls in sample_gt:
            sample_flattened.extend(turn_calls)
        flattened_ground_truth.append(sample_flattened)
    total_ground_truth_functions = sum(len(calls) for calls in flattened_ground_truth)
          
    # Calculate similarity metrics
    avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
    max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
    min_similarity_score = min(similarity_scores) if similarity_scores else 0.0
    similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
    
    # Calculate percentage of high similarity matches (threshold-based)
    high_similarity_threshold = 0.8
    high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
    high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0
    
    # Calculate Turn-by-Turn Accuracy
    turn_accuracies = {}
    if all_turn_exact_matches:
        max_turns = max(len(turns) for turns in all_turn_exact_matches)
        for i in range(max_turns):
            turn_scores = [sample_turns[i] for sample_turns in all_turn_exact_matches if i < len(sample_turns)]
            if turn_scores:
                turn_accuracies[f"turn_{i+1}_accuracy"] = sum(turn_scores) / len(turn_scores)
    result['turn_level_accuracy'] = turn_accuracies

    # Calculate Turn-by-Turn Accuracy Until Failure
    turn_recalls_until_failure = {}
    if all_turn_recalls_until_failure:
        max_turns = max(len(turns) for turns in all_turn_recalls_until_failure)
        for i in range(max_turns):
            turn_scores = [sample_turns[i] for sample_turns in all_turn_recalls_until_failure if i < len(sample_turns)]
            if turn_scores:
                turn_recalls_until_failure[f"turn_{i+1}_recall_until_failure"] = sum(turn_scores) / len(turn_scores)
    result['turn_level_recalls_until_failure'] = turn_recalls_until_failure

    # Calculate Average Turn of First Failure
    first_failure_turns = []
    for sample_turns in all_turn_exact_matches:
        failed_turns = [i + 1 for i, match in enumerate(sample_turns) if match == 0.0]
        if failed_turns:
            first_failure_turns.append(min(failed_turns))
        else:
            # If a sample never fails, its "first failure" is after the last turn
            first_failure_turns.append(len(sample_turns) + 1)

    result['avg_turn_of_first_failure'] = sum(first_failure_turns) / len(first_failure_turns) if first_failure_turns else 0.0

    total_ground_truth_turns = sum(len(gt_turns) for gt_turns in ground_truth_calls)
    result['total_invalid_turns'] = total_invalid_turns
    result['invalid_turn_percentage'] = (total_invalid_turns / total_ground_truth_turns * 100) if total_ground_truth_turns > 0 else 0.0
    result['total_invalid_conversations'] = total_invalid_conversations
    result['invalid_conversation_percentage'] = (total_invalid_conversations / total_samples * 100) if total_samples > 0 else 0.0
    result['exact_match_accuracy'] = exact_match_accuracy
    result['turn_level_recalls_until_failure'] = turn_recalls_until_failure
    result['avg_similarity_score'] = avg_similarity_score
    result['max_similarity_score'] = max_similarity_score
    result['min_similarity_score'] = min_similarity_score
    result['similarity_std'] = similarity_std
    result['high_similarity_percentage'] = high_similarity_percentage
    result['high_similarity_threshold'] = high_similarity_threshold
    result['total_samples'] = total_samples
    result['total_predicted_functions'] = total_predicted_functions
    result['total_ground_truth_functions'] = total_ground_truth_functions
    result['avg_predicted_functions_per_sample'] = total_predicted_functions / total_samples if total_samples > 0 else 0.0
    result['avg_ground_truth_functions_per_sample'] = total_ground_truth_functions / total_samples if total_samples > 0 else 0.0
    result['exact_matches'] = exact_matches
    result['similarity_scores'] = similarity_scores
    result['predicted_function_calls'] = flattened_predicted_calls
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
        "use_cache": True,
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

        # fully flattened prediction strings
        batch_all_full_responses = [[] for _ in range(batch_size)]

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
                    predicted_calls_for_turn = _extract_function_calls(response_text)
                    batch_all_predicted_calls[i].append(predicted_calls_for_turn)
                    batch_all_full_responses[i].append(response_text)
                    
                    # Add the model's output to the history as the assistant's turn
                    if response_text: # Only add assistant message if something was generated
                        batch_chat_histories[i].append({"role": "assistant", "content": response_text})

        # Validate and calculate metrics using the new function and structured data
        convo_exact_matches, turn_exact_matches, convo_similarity_scores, invalid_turns, turn_recalls_until_failure, invalid_conversations = _validate_outputs_and_calculate_metrics(
            similarity_model, batch_all_predicted_calls, ground_truth_functions, batch_all_full_responses
        )

        # Update results
        dataset_results.total_batches = batch_idx + 1
        dataset_results.total_samples += len(user_turns_batch)
        dataset_results.all_exact_matches.extend(convo_exact_matches)
        dataset_results.all_similarity_scores.extend(convo_similarity_scores)
        dataset_results.all_turn_exact_matches.extend(turn_exact_matches)
        dataset_results.all_turn_recalls_until_failure.extend(turn_recalls_until_failure)
        # Store both the structured and a flattened version of predictions
        dataset_results.all_predicted_calls.extend(
            [[call for turn in sample for call in turn] for sample in batch_all_predicted_calls]
        )
        dataset_results.all_ground_truth_calls.extend(ground_truth_functions)
        dataset_results.all_extracted_function_calls.extend(batch_all_predicted_calls)
        dataset_results.all_full_responses.extend(batch_all_full_responses)
        dataset_results.total_invalid_turns += invalid_turns
        dataset_results.total_invalid_conversations += invalid_conversations


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
        dataset_results.all_extracted_function_calls,
        dataset_results.all_ground_truth_calls,
        dataset_results.all_turn_exact_matches,
        dataset_results.total_invalid_turns,
        dataset_results.all_turn_recalls_until_failure,
        dataset_results.total_invalid_conversations
    )

    # Update dataset results
    dataset_results.total_invalid_turns = final_metrics["total_invalid_turns"]
    dataset_results.invalid_turn_percentage = final_metrics["invalid_turn_percentage"]
    dataset_results.total_invalid_conversations = final_metrics["total_invalid_conversations"]
    dataset_results.invalid_conversation_percentage = final_metrics["invalid_conversation_percentage"]
    dataset_results.turn_level_accuracy = final_metrics["turn_level_accuracy"]
    dataset_results.avg_turn_of_first_failure = final_metrics["avg_turn_of_first_failure"]
    dataset_results.turn_level_recalls_until_failure = final_metrics["turn_level_recalls_until_failure"]
    dataset_results.exact_match_accuracy = final_metrics["exact_match_accuracy"]
    dataset_results.avg_similarity_score = final_metrics["avg_similarity_score"]
    dataset_results.max_similarity_score = final_metrics["max_similarity_score"]
    dataset_results.min_similarity_score = final_metrics["min_similarity_score"]
    dataset_results.similarity_std = final_metrics["similarity_std"]
    dataset_results.high_similarity_percentage = final_metrics["high_similarity_percentage"]
    dataset_results.high_similarity_threshold = final_metrics["high_similarity_threshold"]
    dataset_results.total_predicted_functions = final_metrics["total_predicted_functions"]
    dataset_results.total_ground_truth_functions = final_metrics["total_ground_truth_functions"]
    dataset_results.avg_valid_predicted_functions_per_sample = final_metrics["avg_predicted_functions_per_sample"]
    dataset_results.avg_ground_truth_functions_per_sample = final_metrics["avg_ground_truth_functions_per_sample"]
    dataset_results.eval_time = time.perf_counter() - start_time

    # Print summary
    print("\n=== BFCL Magma Inference Results Summary ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Device: {model.device}")
    print(f"Total samples: {dataset_results.total_samples}")
    print(f"Exact Match Accuracy: {dataset_results.exact_match_accuracy:.4f}")
    print(f"Average Similarity Score: {dataset_results.avg_similarity_score:.4f}")
    print(f"Max Similarity Score: {dataset_results.max_similarity_score:.4f}")
    print(f"Min Similarity Score: {dataset_results.min_similarity_score:.4f}")
    print(f"Similarity Std Dev: {dataset_results.similarity_std:.4f}")
    print(f"High Similarity (≥{dataset_results.high_similarity_threshold}): {dataset_results.high_similarity_percentage:.2f}%")
    print(f"Invalid turns: {dataset_results.total_invalid_turns} ({dataset_results.invalid_turn_percentage:.2f}%)")
    print(f"Average predicted functions per sample: {dataset_results.avg_valid_predicted_functions_per_sample:.2f}")
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
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the results file.")
    parser.add_argument('--results_filename', type=str, default="bfcl_magma_results.json", help="Name of the output results file.")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to process (default: all samples)")
    args = parser.parse_args()
    
    # Validate dataset directory
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    main(args)