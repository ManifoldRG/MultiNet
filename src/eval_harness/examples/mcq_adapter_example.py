"""
MCQ Model Adapter Example

This example demonstrates how to implement the ModelAdapter interface
for Multiple Choice Question (MCQ) tasks like PIQA where each question
has a fixed number of choices that are specific to that question.
"""
import os, sys
# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import numpy as np
from typing import Dict, Any, List, Optional, Union
from src.eval_harness.model_adapter import MultipleChoiceAdapter


class SimpleMCQAdapter(MultipleChoiceAdapter):
    """
    Example adapter for Multiple Choice Question tasks.
    
    This is designed for MCQ tasks like PIQA where:
    - Each question has a fixed number of choices (e.g., 2 for binary choice)
    - The choices are question-specific (different for every question)
    - The task is to select the correct choice for each question
    
    For visual classification with consistent categories (like ODinW),
    use ClassificationAdapter instead.
    """
    
    def __init__(self, num_choices: int = 2):
        super().__init__(
            model_name="SimpleMCQModel",
            supported_datasets=["piqa"],
            num_choices=num_choices
        )
        self.model = None
        
    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the MCQ model."""
        print(f"Initializing {self.model_name} with {self.num_choices} choices per question")
        
        # Mock model initialization
        self.model = MockMCQModel(num_choices=self.num_choices)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("MCQ model initialized successfully!")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        return_probabilities: bool = False,
        **kwargs
    ) -> Union[int, Dict[str, Any]]:
        """Predict choice for MCQ task."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract question
        question = observation.get('question', None)
        
        if question is None:
            raise ValueError("No question found in observation. Expected 'question' key.")
        
        # Run inference
        choice_idx, probabilities = self.model.predict_choice(question, self.num_choices)
        
        # Validate choice is in valid range
        if not (0 <= choice_idx < self.num_choices):
            print(f"Warning: Model predicted invalid choice {choice_idx}, using 0")
            choice_idx = 0
        
        if return_probabilities:
            return {
                'choice': choice_idx,
                'probabilities': probabilities
            }
        
        return choice_idx
        
    def batch_predict_actions(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> List[int]:
        """Predict choices for a batch of MCQ observations."""
        
        # Extract batch components
        questions = batch.get('question', [])
        
        batch_size = len(questions)
        
        if batch_size == 0:
            return []
        
        # For this example, process sequentially
        # Real implementation could use batch processing
        results = []
        for i in range(batch_size):
            try:
                # Build observation for this item
                observation = {'question': questions[i]}
                
                # Predict choice
                choice = self.predict_action(observation, **kwargs)
                results.append(choice)
                
            except Exception as e:
                # For failed predictions, return 0 (first choice)
                print(f"Warning: Prediction failed for item {i}: {e}")
                results.append(0)
            
        return results
        
    def reset(self):
        """Reset MCQ model state."""
        if self.model and hasattr(self.model, 'reset'):
            self.model.reset()
        print("MCQ model state reset")


class MockMCQModel:
    """Mock MCQ model for testing."""
    
    def __init__(self, num_choices: int = 2):
        self.num_choices = num_choices
        
    def predict_choice(self, question: str, num_choices: int) -> tuple:
        """Mock MCQ prediction."""
        
        if not question:
            # Default to first choice if no question
            probabilities = [1.0] + [0.0] * (num_choices - 1)
            return 0, probabilities
        
        # Simple heuristic-based prediction
        # In a real model, this would use actual inference
        question_lower = question.lower()
        
        # Base scores for each choice
        choice_scores = np.ones(num_choices) * 0.5
        
        # Simple heuristics (just for demonstration)
        # Real models would use actual NLP/reasoning
        if "better" in question_lower or "correct" in question_lower:
            # Slightly favor choice 0 for "which is better" questions
            choice_scores[0] += 0.2
        
        # Add some randomness
        choice_scores += np.random.random(num_choices) * 0.3
        
        # Normalize to probabilities
        probabilities = choice_scores / choice_scores.sum()
        
        # Pick the highest scoring choice
        choice_idx = int(np.argmax(choice_scores))
        
        return choice_idx, probabilities.tolist()
    
    def reset(self):
        """Reset model state if needed."""
        pass


def test_mcq_adapter():
    """Test the MCQ adapter implementation."""
    
    print("=== Testing SimpleMCQAdapter ===\n")
    
    # Create adapter for binary choice (like PIQA)
    adapter = SimpleMCQAdapter(num_choices=2)
    
    # Initialize
    adapter.initialize()
    
    # Test model info
    info = adapter.get_model_info()
    print(f"Model info: {info}\n")
    
    # Test PIQA-style observation
    print("--- Testing PIQA (binary multiple choice) ---")
    
    piqa_observation = {
        'question': 'Goal: To remove rust from a knife.\nSolution 0: Soak the knife in lemon juice.\nSolution 1: Soak the knife in sugar water.\nWhich solution is better for the given goal? Output 0 or 1 only.'
    }
    
    choice = adapter.predict_action(
        piqa_observation,
        dataset_name="piqa"
    )
    print(f"Question: {piqa_observation['question'][:100]}...")
    print(f"Predicted choice: {choice}\n")
    
    # Test with return_probabilities
    result = adapter.predict_action(
        piqa_observation,
        dataset_name="piqa",
        return_probabilities=True
    )
    print(f"With probabilities:")
    print(f"  Choice: {result['choice']}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}\n")
    
    # Test another example
    piqa_observation_2 = {
        'question': 'Goal: Clean a computer keyboard.\nSolution 0: Use compressed air between the keys.\nSolution 1: Pour water over the keyboard.\nWhich solution is better? Output 0 or 1 only.'
    }
    
    choice = adapter.predict_action(
        piqa_observation_2,
        dataset_name="piqa"
    )
    print(f"Question: {piqa_observation_2['question'][:80]}...")
    print(f"Predicted choice: {choice}\n")
    
    # Test batch processing
    print("--- Testing batch MCQ ---")
    batch = {
        'question': [
            'Goal: Make a bookmark. Solution 0: Use a paperclip. Solution 1: Use a stapler. Which is better? Output 0 or 1.',
            'Goal: Preserve flowers. Solution 0: Press them in a book. Solution 1: Put them in water. Which is better? Output 0 or 1.',
            'Goal: Open a jar. Solution 0: Use a jar opener. Solution 1: Hit it with a hammer. Which is better? Output 0 or 1.'
        ]
    }
    
    batch_results = adapter.batch_predict_actions(batch, dataset_name="piqa")
    print(f"Batch results: {len(batch_results)} predictions")
    for i, choice in enumerate(batch_results):
        print(f"  Question {i+1}: choice={choice}")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Test missing question
        bad_obs = {}
        adapter.predict_action(bad_obs, dataset_name="piqa")
    except Exception as e:
        print(f"Expected error for missing question: {e}")
    
    # Test multiple predictions to show variation
    print("\n--- Testing prediction variation ---")
    for i in range(3):
        choice = adapter.predict_action(piqa_observation, dataset_name="piqa")
        print(f"Prediction {i+1}: choice={choice}")
        
    print("\n=== MCQ adapter test completed! ===")


if __name__ == "__main__":
    test_mcq_adapter()

