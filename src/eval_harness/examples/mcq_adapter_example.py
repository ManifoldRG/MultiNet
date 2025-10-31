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

import re
import numpy as np
from typing import Dict, Any, List, Optional, Union
from src.eval_harness.model_adapter import ModelAdapter


class SimpleMCQAdapter(ModelAdapter):
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
        super().__init__()
        self.model_name = "SimpleMCQModel"
        self.model_type = "multiple_choice"
        self.num_choices = num_choices
        self.model = None

    @property
    def supported_datasets(self) -> List[str]:
        return ["piqa"]

    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the MCQ model."""
        print(f"Initializing {self.model_name} with {self.num_choices} choices per question")
        
        # Mock model initialization
        self.model = MockMCQModel(num_choices=self.num_choices)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("MCQ model initialized")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict choice for MCQ task."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # For PIQA and other text-only MCQ tasks, observation may be None
        # Question comes from instruction parameter
        if instruction is None:
            raise ValueError("No instruction provided. MCQ tasks require an instruction (the question).")
        
        # Run inference
        raw_output, choice_output = self.model.predict_choice(instruction, self.num_choices)
        
        # Adapter handles ALL extraction and validation
        final_choice = self._extract_and_validate_choice(choice_output, self.num_choices)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": final_choice  # Always int in [0, num_choices-1] or -1
        }
    
    def _extract_and_validate_choice(self, output: Any, num_choices: int) -> int:
        """
        Extract and validate choice from model output.
        Returns -1 if invalid (out of range, wrong type, unparseable).
        """
        if output is None:
            return -1
        
        # Handle string outputs - parse numbers
        if isinstance(output, str):
            numbers = re.findall(r'\d+', output.strip())
            if numbers:
                choice = int(numbers[0])
                return choice if 0 <= choice < num_choices else -1
            return -1
        
        # Handle integer outputs
        if isinstance(output, (int, np.integer)):
            choice = int(output)
            return choice if 0 <= choice < num_choices else -1
        
        # Handle float outputs - round to nearest int
        if isinstance(output, (float, np.floating)):
            if not np.isfinite(output):
                return -1
            choice = int(np.round(output))
            return choice if 0 <= choice < num_choices else -1
        
        # Unknown type
        return -1
        
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Predict choices for a batch of MCQ observations."""
        
        batch_size = len(observations)
        
        if batch_size == 0:
            return []
        
        # For this example, process sequentially
        # Real implementation could use batch processing
        results = []
        for i in range(batch_size):
            try:
                # Get observation and instruction for this item
                # For PIQA, observation is None, instruction contains the question
                observation = observations[i]
                instruction = instructions[i] if instructions else None
                
                # Predict choice
                result = self.predict_action(observation, instruction, dataset_name, **kwargs)
                results.append(result)
                
            except Exception as e:
                # For failed predictions, return -1 (invalid)
                print(f"Warning: Prediction failed for item {i}: {e}")
                results.append({
                    "raw_output": f"Error: {str(e)}",
                    "extracted_outputs": -1
                })
            
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
        
    def predict_choice(self, question: str, num_choices: int) -> tuple[str, Any]:
        """
        Mock MCQ prediction.
        Returns (raw_output: str, choice: Any) where choice can be int, float, or str.
        """
        
        if not question:
            # Default to first choice if no question
            raw_output = "Choosing first option (no question provided)"
            return raw_output, 0
        
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
        
        # Pick the highest scoring choice
        choice_idx = int(np.argmax(choice_scores))
        
        # Generate raw output text
        raw_output = f"Question: {question[:50]}... Best choice: {choice_idx}"
        
        # Return the choice (adapter will handle extraction/validation)
        return raw_output, choice_idx
    
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
    
    piqa_instruction = 'Goal: To remove rust from a knife.\nSolution 0: Soak the knife in lemon juice.\nSolution 1: Soak the knife in sugar water.\nWhich solution is better for the given goal? Output 0 or 1 only.'
    
    result = adapter.predict_action(
        observation=None,  # PIQA has no observation
        instruction=piqa_instruction,
        dataset_name="piqa"
    )
    print(f"Question: {piqa_instruction[:100]}...")
    print(f"Structured result:")
    print(f"  raw_output: {result['raw_output']}")
    print(f"  extracted_outputs: {result['extracted_outputs']}\n")
    
    # Test another example
    piqa_instruction_2 = 'Goal: Clean a computer keyboard.\nSolution 0: Use compressed air between the keys.\nSolution 1: Pour water over the keyboard.\nWhich solution is better? Output 0 or 1 only.'
    
    result = adapter.predict_action(
        observation=None,
        instruction=piqa_instruction_2,
        dataset_name="piqa"
    )
    print(f"Question: {piqa_instruction_2[:80]}...")
    print(f"Structured result:")
    print(f"  raw_output: {result['raw_output']}")
    print(f"  extracted_outputs: {result['extracted_outputs']}\n")
    
    # Test batch processing
    print("--- Testing batch MCQ ---")
    batch_instructions = [
        'Goal: Make a bookmark. Solution 0: Use a paperclip. Solution 1: Use a stapler. Which is better? Output 0 or 1.',
        'Goal: Preserve flowers. Solution 0: Press them in a book. Solution 1: Put them in water. Which is better? Output 0 or 1.',
        'Goal: Open a jar. Solution 0: Use a jar opener. Solution 1: Hit it with a hammer. Which is better? Output 0 or 1.'
    ]
    batch_observations = [None, None, None]  # PIQA has no observations
    
    batch_results = adapter.batch_predict_actions(
        observations=batch_observations,
        instructions=batch_instructions,
        dataset_name="piqa"
    )
    print(f"Batch results: {len(batch_results)} predictions")
    for i, result in enumerate(batch_results):
        print(f"  Question {i+1}: extracted_outputs={result['extracted_outputs']}, raw_output={result['raw_output'][:50]}...")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Test missing instruction
        adapter.predict_action(observation=None, instruction=None, dataset_name="piqa")
    except Exception as e:
        print(f"Expected error for missing instruction: {e}")
    
    # Test multiple predictions to show variation
    print("\n--- Testing prediction variation ---")
    for i in range(3):
        result = adapter.predict_action(observation=None, instruction=piqa_instruction, dataset_name="piqa")
        print(f"Prediction {i+1}: extracted_outputs={result['extracted_outputs']}")
        
    print("\n=== MCQ adapter test completed! ===")


if __name__ == "__main__":
    test_mcq_adapter()

