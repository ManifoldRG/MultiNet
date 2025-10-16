"""
Classification Model Adapter Example

This example demonstrates how to implement the ModelAdapter interface
for visual classification tasks like ODinW and multiple choice tasks like PIQA.
"""
import os, sys
# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union
from src.eval_harness.model_adapter import MultipleChoiceAdapter


class SimpleClassificationAdapter(MultipleChoiceAdapter):
    """
    Example adapter for classification and multiple choice tasks.
    
    This shows how to implement a model for:
    - ODinW: Visual classification (image -> category selection)
    - PIQA: Text-based multiple choice reasoning (text -> choice selection)
    
    The model selects one option from multiple choices.
    """
    
    def __init__(self, num_choices: int = 4):
        super().__init__(
            model_name="SimpleClassificationModel",
            supported_datasets=["odinw", "piqa"],
            num_choices=num_choices
        )
        self.model = None
        
    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the classification model."""
        print(f"Initializing {self.model_name} with {self.num_choices} classes")
        
        # Mock model initialization
        self.model = MockClassificationModel(num_classes=self.num_choices)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("Classification model initialized successfully!")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        return_probabilities: bool = False,
        **kwargs
    ) -> Union[int, Dict[str, Any]]:
        """Predict class/choice for classification task."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract components from observation
        image = observation.get('image', None)
        question = observation.get('question', None)
        options = observation.get('options', None)
        
        if question is None:
            raise ValueError("No question found in observation. Expected 'question' key.")
        
        # Determine number of choices from options if available
        num_classes = len(options) if options else self.num_choices
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation, dataset_name or "odinw")
        
        # Run inference
        class_idx, probabilities = self.model.predict_class(
            processed_obs.get('image'),
            question,
            num_classes,
            options
        )
        
        # Validate class is in valid range
        if not (0 <= class_idx < num_classes):
            print(f"Warning: Model predicted invalid class {class_idx}, using 0")
            class_idx = 0
        
        if return_probabilities:
            return {
                'choice': class_idx,
                'probabilities': probabilities
            }
        
        return class_idx
        
    def batch_predict_actions(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> List[int]:
        """Predict classes/choices for a batch of observations."""
        
        # Extract batch components
        # ODinW uses: image, question, options
        # PIQA uses: question (no image)
        images = batch.get('image', [])
        questions = batch.get('question', [])
        options_list = batch.get('options', [])
        
        batch_size = len(questions) if questions else 0
        
        if batch_size == 0:
            return []
        
        # For this example, process sequentially
        # Real implementation could use batch processing
        results = []
        for i in range(batch_size):
            try:
                # Build observation for this item
                observation = {}
                
                if images and i < len(images) and images[i] is not None:
                    observation['image'] = images[i]
                
                if questions and i < len(questions):
                    observation['question'] = questions[i]
                
                if options_list and i < len(options_list):
                    observation['options'] = options_list[i]
                
                # Predict class
                class_idx = self.predict_action(observation, **kwargs)
                results.append(class_idx)
                
            except Exception as e:
                # For failed predictions, return 0 (first class)
                print(f"Warning: Prediction failed for item {i}: {e}")
                results.append(0)
            
        return results
        
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess observation for classification."""
        
        processed_obs = observation.copy()
        
        # Preprocess image if present (ODinW has images, PIQA doesn't)
        if 'image' in processed_obs and processed_obs['image'] is not None:
            image = processed_obs['image']
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Resize based on dataset requirements
            target_size = self._get_target_size(dataset_name)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
                
            processed_obs['image'] = image
        
        # Normalize question text
        if 'question' in processed_obs:
            question = processed_obs['question']
            # Strip whitespace
            if isinstance(question, str):
                question = question.strip()
                processed_obs['question'] = question
                        
        return processed_obs
        
    def _get_target_size(self, dataset_name: str) -> tuple:
        """Get target image size for different datasets."""
        size_map = {
            "odinw": (224, 224),
            "piqa": (224, 224)  # PIQA doesn't use images, but just in case
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
        num_classes: int,
        options: Optional[List[str]] = None
    ) -> tuple[int, List[float]]:
        """Mock classification prediction."""
        
        if not question:
            # Default to first class if no question
            probabilities = [1.0] + [0.0] * (num_classes - 1)
            return 0, probabilities
        
        # Simple heuristic-based prediction
        question_lower = question.lower()
        
        # Use question text to bias certain choices
        # This is just for demonstration - real models would use actual inference
        class_scores = np.ones(num_classes) * 0.1  # Base score
        
        # If we have options, use them to inform the choice
        if options:
            for i, option in enumerate(options):
                option_lower = option.lower()
                # Simple keyword matching
                if any(word in question_lower for word in option_lower.split()):
                    class_scores[i] += 0.5
        
        # Add some randomness
        class_scores += np.random.random(num_classes) * 0.3
        
        # Normalize to probabilities
        probabilities = class_scores / class_scores.sum()
        
        # Pick the highest scoring class
        class_idx = int(np.argmax(class_scores))
        
        return class_idx, probabilities.tolist()
    
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
    
    odinw_observation = {
        'image': bbox_image,
        'question': 'What object is shown in this image from the AerialMaritimeDrone dataset?\nOption 0: boat\nOption 1: dock\nOption 2: jetski\nOption 3: lift\nOutput the number (0-3) of the correct option only.',
        'options': ['boat', 'dock', 'jetski', 'lift']
    }
    
    class_idx = adapter.predict_action(
        odinw_observation,
        dataset_name="odinw"
    )
    print(f"Question: {odinw_observation['question'][:80]}...")
    print(f"Options: {odinw_observation['options']}")
    print(f"Predicted class: {class_idx} ({odinw_observation['options'][class_idx]})\n")
    
    # Test with return_probabilities
    result = adapter.predict_action(
        odinw_observation,
        dataset_name="odinw",
        return_probabilities=True
    )
    print(f"With probabilities:")
    print(f"  Class: {result['choice']}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}\n")
    
    # Test PIQA style observation (text only, binary choice/MCQ)
    print("--- Testing PIQA (text-based multiple choice) ---")
    adapter_binary = SimpleClassificationAdapter(num_choices=2)
    adapter_binary.initialize()
    
    piqa_observation = {
        'question': 'Goal: To remove rust from a knife.\nSolution 0: Soak the knife in lemon juice.\nSolution 1: Soak the knife in sugar water.\nWhich solution is better for the given goal? Output 0 or 1 only.'
    }
    
    choice = adapter_binary.predict_action(
        piqa_observation,
        dataset_name="piqa"
    )
    print(f"Question: {piqa_observation['question'][:100]}...")
    print(f"Predicted choice: {choice}\n")
    
    # Test batch processing
    print("--- Testing batch classification ---")
    batch = {
        'image': [bbox_image, bbox_image, bbox_image],
        'question': [
            'What object is shown? Option 0: boat Option 1: dock Option 2: jetski Option 3: lift',
            'What object is shown? Option 0: boat Option 1: dock Option 2: jetski Option 3: lift',
            'What object is shown? Option 0: boat Option 1: dock Option 2: jetski Option 3: lift'
        ],
        'options': [
            ['boat', 'dock', 'jetski', 'lift'],
            ['boat', 'dock', 'jetski', 'lift'],
            ['boat', 'dock', 'jetski', 'lift']
        ]
    }
    
    batch_results = adapter.batch_predict_actions(batch, dataset_name="odinw")
    print(f"Batch results: {len(batch_results)} predictions")
    for i, class_idx in enumerate(batch_results):
        print(f"  Sample {i+1}: class={class_idx} ({batch['options'][i][class_idx]})")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Test missing question
        bad_obs = {'image': bbox_image}
        adapter.predict_action(bad_obs, dataset_name="odinw")
    except Exception as e:
        print(f"Expected error for missing question: {e}")
    
    # Test multiple predictions to show variation
    print("\n--- Testing prediction variation ---")
    for i in range(3):
        class_idx = adapter.predict_action(odinw_observation, dataset_name="odinw")
        print(f"Prediction {i+1}: class={class_idx} ({odinw_observation['options'][class_idx]})")
        
    print("\n=== Classification adapter test completed! ===")


if __name__ == "__main__":
    test_classification_adapter()

