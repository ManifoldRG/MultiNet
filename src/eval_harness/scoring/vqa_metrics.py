from typing import Dict, Any, List
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util


def _validate_text_output(output: Any) -> bool:
    """Validate that output is a valid text string."""
    if output is None:
        return False
    if isinstance(output, str) and len(output.strip()) > 0:
        return True
    return False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing punctuation and extra spaces."""
    if not isinstance(text, str):
        return ""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


class VQAMetricsCalculator:
    """
    Calculator for VQA metrics.
    
    Takes text predictions and ground truth answers, calculates similarity and accuracy metrics.
    """
    
    def __init__(self, similarity_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model for similarity calculations.
        
        Args:
            similarity_model_name: Name of the sentence transformer model to use
        """
        self.similarity_model = SentenceTransformer(similarity_model_name)
    
    def calculate_metrics(self, predictions: List[str], ground_truth_answers: List[str]) -> Dict[str, Any]:
        """
        Calculate metrics for text predictions vs ground truth answers.
        
        Args:
            predictions: List of model text predictions
            ground_truth_answers: List of ground truth text answers
            
        Returns:
            Dictionary containing calculated metrics
        """
        exact_matches = []
        similarity_scores = []
        total_invalid_preds = 0
        
        for i, pred in enumerate(predictions):
            if _validate_text_output(pred):
                # Normalize both prediction and ground truth for fair comparison
                normalized_pred = _normalize_text(pred)
                normalized_gt = _normalize_text(ground_truth_answers[i])
                
                # Calculate exact match
                exact_match = 1.0 if normalized_pred == normalized_gt else 0.0
                exact_matches.append(exact_match)
                
                # Calculate similarity score using sentence embeddings
                try:
                    emb1 = self.similarity_model.encode(pred, convert_to_tensor=True)
                    emb2 = self.similarity_model.encode(ground_truth_answers[i], convert_to_tensor=True)
                    similarity = util.cos_sim(emb1, emb2).item()
                    similarity_scores.append(similarity)
                except Exception as e:
                    # If similarity calculation fails, assign 0.0
                    print(f"Warning: Similarity calculation failed for prediction {i}: {e}")
                    similarity_scores.append(0.0)
            else:
                # Invalid output - assign worst possible scores
                exact_matches.append(0.0)
                similarity_scores.append(0.0)
                total_invalid_preds += 1
        
        return self._calculate_final_metrics(exact_matches, similarity_scores, total_invalid_preds)
    
    def _calculate_final_metrics(self, exact_matches: List[float], similarity_scores: List[float], total_invalid_preds: int) -> Dict[str, Any]:
        """Calculate comprehensive final metrics for QA evaluation."""
        result = {}
        
        # Calculate accuracy metrics
        total_samples = len(exact_matches)
        exact_match_accuracy = sum(exact_matches) / total_samples if total_samples > 0 else 0.0
        
        # Calculate similarity metrics
        avg_similarity_score = sum(similarity_scores) / total_samples if total_samples > 0 else 0.0
        max_similarity_score = max(similarity_scores) if similarity_scores else 0.0
        min_similarity_score = min(similarity_scores) if similarity_scores else 0.0
        
        # Calculate additional statistics
        similarity_std = np.std(similarity_scores) if similarity_scores else 0.0
        
        # Calculate percentage of high similarity matches (threshold-based)
        high_similarity_threshold = 0.8
        high_similarity_count = sum(1 for score in similarity_scores if score >= high_similarity_threshold)
        high_similarity_percentage = (high_similarity_count / total_samples * 100) if total_samples > 0 else 0.0
        
        # Calculate invalid prediction percentage
        invalid_percentage = (total_invalid_preds / total_samples * 100) if total_samples > 0 else 0.0
        
        result.update({
            'exact_match_accuracy': exact_match_accuracy,
            'avg_similarity_score': avg_similarity_score,
            'max_similarity_score': max_similarity_score,
            'min_similarity_score': min_similarity_score,
            'similarity_std': similarity_std,
            'high_similarity_percentage': high_similarity_percentage,
            'high_similarity_threshold': high_similarity_threshold,
            'total_samples': total_samples,
            'total_invalid_preds': total_invalid_preds,
            'invalid_percentage': invalid_percentage,
        })
        
        return result