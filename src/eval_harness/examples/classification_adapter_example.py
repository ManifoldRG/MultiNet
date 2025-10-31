"""
Classification Model Adapter Example

This example demonstrates how to implement the ModelAdapter interface
for visual classification tasks like ODinW
"""
import os, sys
# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import re
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union
from src.eval_harness.model_adapter import ModelAdapter


class SimpleClassificationAdapter(ModelAdapter):
    """
    Example adapter for multiple choice classification taks
    
    This shows how to implement a model for:
    - ODinW: Visual classification (image -> category selection)
    
    The model selects one option from multiple choices.
    """
    
    def __init__(self, num_choices: int = 4):
        super().__init__()
        self.model_name = "SimpleClassificationModel"
        self.model_type = "multiple_choice"
        self.num_choices = num_choices
        self.model = None

    @property
    def supported_datasets(self) -> List[str]:
        return ["odinw"]

    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the classification model."""
        print(f"Initializing {self.model_name} with {self.num_choices} classes")
        
        # Mock model initialization
        self.model = MockClassificationModel(num_classes=self.num_choices)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("Classification model initialized")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict class/choice for classification task."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract image from standardized observation
        image = observation.get('image_observation', None)
        
        # Question comes from instruction parameter
        if instruction is None:
            raise ValueError("No instruction provided. Classification tasks require an instruction (the question).")
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation, dataset_name or "odinw")
        
        # Run inference
        raw_output, class_output = self.model.predict_class(
            processed_obs.get('image_observation'),
            instruction,
            self.num_choices
        )
        
        # Adapter handles ALL extraction and validation
        final_class = self._extract_and_validate_class(class_output, self.num_choices)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": final_class  # Always int in [0, num_classes-1] or -1
        }
    
    def _extract_and_validate_class(self, output: Any, num_classes: int) -> int:
        """
        Extract and validate class index from model output.
        Returns -1 if invalid (out of range, wrong type, unparseable).
        """
        FLOAT_NUM_TOLERANCE = 0.01
        
        if output is None:
            return -1
        
        # Handle string outputs - parse numbers
        if isinstance(output, str):
            numbers = re.findall(r'\d+', output.strip())
            if numbers:
                class_idx = int(numbers[0])
                return class_idx if 0 <= class_idx < num_classes else -1
            return -1
        
        # Handle integer outputs
        if isinstance(output, (int, np.integer)):
            class_idx = int(output)
            return class_idx if 0 <= class_idx < num_classes else -1
        
        # Handle float outputs with tolerance
        if isinstance(output, (float, np.floating)):
            if not np.isfinite(output):
                return -1
            # Check if close enough to an integer
            rounded = np.round(output)
            if abs(output - rounded) > FLOAT_NUM_TOLERANCE:
                return -1
            class_idx = int(rounded)
            return class_idx if 0 <= class_idx < num_classes else -1
        
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
        """Predict classes/choices for a batch of observations."""
        
        batch_size = len(observations)
        
        if batch_size == 0:
            return []
        
        # For this example, process sequentially
        # Real implementation could use batch processing
        results = []
        for i in range(batch_size):
            try:
                # Get observation and instruction for this item
                observation = observations[i]
                instruction = instructions[i] if instructions else None
                
                # Predict class
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
        
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess observation for classification."""
        
        processed_obs = observation.copy()
        
        # Preprocess image from standardized key
        if 'image_observation' in processed_obs and processed_obs['image_observation'] is not None:
            image = processed_obs['image_observation']
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Resize based on dataset requirements
            target_size = self._get_target_size(dataset_name)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
            processed_obs['image_observation'] = image
                        
        return processed_obs
        
    def _get_target_size(self, dataset_name: str) -> tuple:
        """Get target image size for different datasets."""
        size_map = {
            "odinw": (224, 224),
        }
        return size_map.get(dataset_name, (224, 224))
        
    def reset(self):
        """Reset classification model state."""
        if self.model and hasattr(self.model, 'reset'):
            self.model.reset()
        print("Classification model state reset")


class MockClassificationModel:
    """Mock classification model for testing."""
    
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        
    def predict_class(
        self, 
        image: Optional[Image.Image], 
        question: str,
        num_classes: int
    ) -> tuple[str, Any]:
        """
        Mock classification prediction.
        Returns (raw_output: str, class_idx: Any) where class_idx can be int, float, or str.
        """
        
        if not question:
            # Default to first class if no question
            raw_output = "Choosing first class (no question provided)"
            return raw_output, 0
        
        # Simple heuristic-based prediction
        question_lower = question.lower()
        
        # Use question text to bias certain choices
        # This is just for demonstration - real models would use actual inference
        class_scores = np.ones(num_classes) * 0.1  # Base score
        
        # Add some randomness
        class_scores += np.random.random(num_classes) * 0.3
        
        # Pick the highest scoring class
        class_idx = int(np.argmax(class_scores))
        
        # Generate raw output text
        raw_output = f"Question: {question[:50]}... Predicted class: {class_idx}"
        
        # Return the class (adapter will handle extraction/validation)
        return raw_output, class_idx
    
    def reset(self):
        """Reset model state if needed."""
        pass


def test_classification_adapter():
    """Test the classification adapter implementation."""
    
    print("=== Testing SimpleClassificationAdapter ===\n")
    
    # Create adapter
    adapter = SimpleClassificationAdapter(num_choices=4)
    
    # Initialize
    adapter.initialize()
    
    # Test model info
    info = adapter.get_model_info()
    print(f"Model info: {info}\n")
    
    # Test ODinW style observation (visual classification with image)
    print("--- Testing ODinW (visual classification) ---")
    bbox_image = Image.new('RGB', (128, 128), color='blue')
    
    odinw_instruction = 'What object is shown in this image from the AerialMaritimeDrone dataset?\nOption 0: boat\nOption 1: dock\nOption 2: jetski\nOption 3: lift\nOutput the number (0-3) of the correct option only.'
    options = ['boat', 'dock', 'jetski', 'lift']
    
    odinw_observation = {
        'image_observation': bbox_image
    }
    
    result = adapter.predict_action(
        observation=odinw_observation,
        instruction=odinw_instruction,
        dataset_name="odinw"
    )
    print(f"Question: {odinw_instruction[:80]}...")
    print(f"Options: {options}")
    print(f"Structured result:")
    print(f"  raw_output: {result['raw_output']}")
    print(f"  extracted_outputs: {result['extracted_outputs']} ({options[result['extracted_outputs']]})\n")
    
    # Test batch processing
    print("--- Testing batch classification ---")
    batch_observations = [
        {'image_observation': bbox_image},
        {'image_observation': bbox_image},
        {'image_observation': bbox_image}
    ]
    batch_instructions = [
        'What object is shown? Option 0: boat Option 1: dock Option 2: jetski Option 3: lift',
        'What object is shown? Option 0: boat Option 1: dock Option 2: jetski Option 3: lift',
        'What object is shown? Option 0: boat Option 1: dock Option 2: jetski Option 3: lift'
    ]
    
    batch_results = adapter.batch_predict_actions(
        observations=batch_observations,
        instructions=batch_instructions,
        dataset_name="odinw"
    )
    print(f"Batch results: {len(batch_results)} predictions")
    for i, result in enumerate(batch_results):
        print(f"  Sample {i+1}: extracted_outputs={result['extracted_outputs']} ({options[result['extracted_outputs']]})")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Test missing instruction
        adapter.predict_action(observation=odinw_observation, instruction=None, dataset_name="odinw")
    except Exception as e:
        print(f"Expected error for missing instruction: {e}")
    
    # Test multiple predictions to show variation
    print("\n--- Testing prediction variation ---")
    for i in range(3):
        result = adapter.predict_action(observation=odinw_observation, instruction=odinw_instruction, dataset_name="odinw")
        print(f"Prediction {i+1}: extracted_outputs={result['extracted_outputs']} ({options[result['extracted_outputs']]})")
        
    print("\n=== Classification adapter test completed! ===")


if __name__ == "__main__":
    test_classification_adapter()

