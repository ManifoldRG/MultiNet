"""
BFCL Metrics Calculator for Multi-Turn Function Calling Evaluation

This module provides metrics calculation for the Berkeley Function Calling Leaderboard (BFCL)
dataset, which evaluates models on multi-turn function calling tasks.

Note: Function call extraction is now handled by model adapters, not this metrics calculator.
Model adapters return structured predictions with both raw_output and extracted_calls.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util


class BFCLMetricsCalculator:
    """
    Calculator for BFCL (Berkeley Function Calling Leaderboard) metrics.
    
    Evaluates multi-turn function calling tasks with conversation-level and turn-level metrics.
    """
    
    def __init__(self, similarity_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model for similarity calculations.
        
        Args:
            similarity_model_name: Name of the sentence transformer model to use
        """
        self.similarity_model = SentenceTransformer(similarity_model_name, device="cpu")
    
    def calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate BFCL metrics for multi-turn function calling predictions."""
        predicted_calls_per_sample = []
        ground_truth_per_sample = []
        full_responses_per_sample = []
        
        for pred_dict, gt_dict in zip(predictions, ground_truths):
            turn_predictions_calls = []
            turn_responses_raw = []
            
            for turn_pred in pred_dict['predictions']:
                if isinstance(turn_pred, dict):
                    extracted_calls = turn_pred.get('extracted_calls', [])
                    turn_predictions_calls.append(extracted_calls)
                    raw_output = turn_pred.get('raw_output', '')
                    turn_responses_raw.append(raw_output)
                else:
                    response_str = str(turn_pred) if turn_pred is not None else ""
                    turn_responses_raw.append(response_str)
                    turn_predictions_calls.append([])
            
            predicted_calls_per_sample.append(turn_predictions_calls)
            ground_truth_per_sample.append(gt_dict['ground_truth'])
            full_responses_per_sample.append(turn_responses_raw)
        
        # Calculate metrics
        (
            all_convo_exact_matches,
            all_turn_exact_matches,
            all_convo_similarity_scores,
            total_invalid_turns,
            all_turn_recalls_until_failure,
            total_invalid_conversations
        ) = self._validate_outputs_and_calculate_metrics(
            predicted_calls_per_sample,
            ground_truth_per_sample,
            full_responses_per_sample
        )
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(
            all_convo_exact_matches,
            all_convo_similarity_scores,
            predicted_calls_per_sample,
            ground_truth_per_sample,
            all_turn_exact_matches,
            total_invalid_turns,
            all_turn_recalls_until_failure,
            total_invalid_conversations
        )
        
        return final_metrics
    
    def _validate_outputs_and_calculate_metrics(
        self,
        predicted_calls_per_sample: List[List[List[str]]],
        ground_truth_per_sample: List[List[List[str]]],
        full_responses_per_sample: List[List[str]]
    ) -> Tuple[List[float], List[List[float]], List[float], int, List[List[float]], int]:
        """
        Calculate conversation-level and turn-level metrics, and count empty predictions.
        
        Args:
            predicted_calls_per_sample: Batch of predictions, structured as [sample][turn][calls]
            ground_truth_per_sample: Batch of ground truths, structured as [sample][turn][calls]
            full_responses_per_sample: Full text responses for similarity calculation
            
        Returns:
            Tuple containing:
            - all_convo_exact_matches: List of 1.0/0.0 scores for each conversation
            - all_turn_exact_matches: List of lists with 1.0/0.0 scores for each turn
            - all_convo_similarity_scores: List of similarity scores for each conversation
            - total_invalid_turns: Count of turns where model predicted nothing but should have
            - all_turn_recalls_until_failure: Recall scores until first failure per turn
            - total_invalid_conversations: Count of conversations with at least one invalid turn
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
            
            # Calculate similarity
            predicted_text = "\n".join(full_responses_per_sample[i])
            gt_text = "\n".join(flattened_gt)
            try:
                emb1 = self.similarity_model.encode(predicted_text, convert_to_tensor=True)
                emb2 = self.similarity_model.encode(gt_text, convert_to_tensor=True)
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
                
                # Prediction is empty but ground truth is not
                if not pred_calls_for_turn and gt_calls_for_turn:
                    total_invalid_turns += 1
                    invalid_conversation = True
                
                turn_exact_match = 1.0 if pred_calls_for_turn == gt_calls_for_turn else 0.0
                turn_matches.append(turn_exact_match)
                
                # Calculate recall until failure
                function_matches = 0
                for pred_call, gt_call in zip(pred_calls_for_turn, gt_calls_for_turn):
                    if pred_call == gt_call:
                        function_matches += 1
                    else:
                        break
                
                recall = function_matches / len(gt_calls_for_turn) if gt_calls_for_turn else 1.0
                turn_recalls_until_failure.append(recall)
            
            all_turn_exact_matches.append(turn_matches)
            all_turn_recalls_until_failure.append(turn_recalls_until_failure)
            
            if invalid_conversation:
                total_invalid_conversations += 1
        
        return (
            all_convo_exact_matches,
            all_turn_exact_matches,
            all_convo_similarity_scores,
            total_invalid_turns,
            all_turn_recalls_until_failure,
            total_invalid_conversations
        )
    
    def _calculate_final_metrics(
        self,
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
        
        # Calculate conversation-level accuracy metrics
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
        
        # Calculate percentage of high similarity matches
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
        
        # Calculate Turn-by-Turn Recall Until Failure
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
        
        # Invalid turn statistics
        total_ground_truth_turns = sum(len(gt_turns) for gt_turns in ground_truth_calls)
        result['total_invalid_turns'] = total_invalid_turns
        result['invalid_turn_percentage'] = (total_invalid_turns / total_ground_truth_turns * 100) if total_ground_truth_turns > 0 else 0.0
        result['total_invalid_conversations'] = total_invalid_conversations
        result['invalid_conversation_percentage'] = (total_invalid_conversations / total_samples * 100) if total_samples > 0 else 0.0
        
        # Main metrics
        result['exact_match_accuracy'] = exact_match_accuracy
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
        
        return result

