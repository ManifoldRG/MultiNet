"""
VQA Model Adapter Example

This example demonstrates how to implement the ModelAdapter interface
for Visual Question Answering tasks like robot_vqa and sqa3d.
"""
import os, sys
# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from src.eval_harness.model_adapter import TextGenerationAdapter


class SimpleVQAAdapter(TextGenerationAdapter):
    """
    Example adapter for Visual Question Answering tasks.
    
    This shows how to implement a model for VQA tasks like robot_vqa and sqa3d
    where the model takes an image and question as input and generates a text answer.
    """
    
    def __init__(self, max_answer_length: int = 100):
        super().__init__(
            model_name="SimpleVQAModel",
            supported_datasets=["robot_vqa", "sqa3d"],
            max_answer_length=max_answer_length
        )
        self.model = None
        
    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the VQA model."""
        print(f"Initializing {self.model_name}")
        
        # Mock model initialization
        self.model = MockVQAModel(max_length=self.max_answer_length)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("VQA model initialized successfully!")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text answer for a visual question."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract components from observation
        # Handle different observation formats (robot_vqa vs sqa3d)
        image = observation.get('image', observation.get('scene_image', None))
        question = observation.get('question', observation.get('text_observation', None))
            
        if question is None:
            raise ValueError("No question found in observation. Expected 'question' or 'text_observation' key.")
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation, dataset_name or "robot_vqa")
        
        # Run inference
        answer = self.model.answer_question(
            processed_obs.get('image'),
            question
        )
        
        # Validate answer is a string
        if not isinstance(answer, str):
            print(f"Warning: Model returned non-string answer, converting to string")
            answer = str(answer)
        
        # Truncate if needed
        if len(answer) > self.max_answer_length:
            answer = answer[:self.max_answer_length]
        
        return answer
        
    def batch_predict_actions(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> List[str]:
        """Predict answers for a batch of VQA observations."""
        
        # Extract batch components - handle different dataset formats
        # robot_vqa uses: image_observation, text_observation
        # sqa3d uses: scene_image, question
        images = batch.get('image_observation', batch.get('scene_image', []))
        questions = batch.get('text_observation', batch.get('question', []))
        
        batch_size = len(questions) if questions else len(images)
        
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
                
                # Predict answer
                answer = self.predict_action(observation, **kwargs)
                results.append(answer)
                
            except Exception as e:
                # For failed predictions, return empty string
                print(f"Warning: Prediction failed for item {i}: {e}")
                results.append("")
            
        return results
        
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess observation for VQA."""
        
        processed_obs = observation.copy()
        
        # Preprocess image - handle both 'image' and 'scene_image' keys
        image_key = None
        if 'image' in processed_obs:
            image_key = 'image'
        elif 'scene_image' in processed_obs:
            image_key = 'scene_image'
            
        if image_key:
            image = processed_obs[image_key]
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Resize based on dataset requirements
            target_size = self._get_target_size(dataset_name)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Store as 'image' for consistency
            processed_obs['image'] = image
        
        # Normalize question text
        question_key = 'question' if 'question' in processed_obs else 'text_observation'
        if question_key in processed_obs:
            question = processed_obs[question_key]
            # Strip whitespace
            if isinstance(question, str):
                question = question.strip()
                processed_obs[question_key] = question
                        
        return processed_obs
        
    def _get_target_size(self, dataset_name: str) -> tuple:
        """Get target image size for different datasets."""
        size_map = {
            "robot_vqa": (224, 224),
            "sqa3d": (224, 224)
        }
        return size_map.get(dataset_name, (224, 224))
        
    def reset(self):
        """Reset VQA model state."""
        if self.model and hasattr(self.model, 'reset'):
            self.model.reset()
        print("VQA model state reset")


class MockVQAModel:
    """Mock VQA model for testing."""
    
    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        # Simple answer templates for common question types
        self.answer_templates = {
            "what": ["object", "item", "thing", "robot arm", "gripper"],
            "where": ["on the table", "in the center", "to the left", "to the right"],
            "how many": ["one", "two", "three", "several"],
            "is": ["yes", "no"],
            "can": ["yes", "no"],
            "does": ["yes", "no"],
            "color": ["red", "blue", "green", "yellow", "black", "white"],
            "default": ["I see that in the image", "Based on the observation", "Yes"]
        }
        
    def answer_question(
        self, 
        image: Optional[Image.Image], 
        question: str
    ) -> str:
        """Mock VQA answer generation."""
        
        if not question:
            return "No question provided"
        
        # Simple rule-based answer generation
        question_lower = question.lower().strip()
        
        # Remove question marks and normalize
        question_lower = question_lower.rstrip('?').strip()
        
        # Determine question type and generate appropriate answer
        if question_lower.startswith("what"):
            answers = self.answer_templates["what"]
            answer = np.random.choice(answers)
        elif question_lower.startswith("where"):
            answers = self.answer_templates["where"]
            answer = np.random.choice(answers)
        elif question_lower.startswith("how many"):
            answers = self.answer_templates["how many"]
            answer = np.random.choice(answers)
        elif question_lower.startswith("is ") or question_lower.startswith("are "):
            answers = self.answer_templates["is"]
            answer = np.random.choice(answers)
        elif question_lower.startswith("can "):
            answers = self.answer_templates["can"]
            answer = np.random.choice(answers)
        elif question_lower.startswith("does ") or question_lower.startswith("do "):
            answers = self.answer_templates["does"]
            answer = np.random.choice(answers)
        elif "color" in question_lower or "colour" in question_lower:
            answers = self.answer_templates["color"]
            answer = np.random.choice(answers)
        else:
            answers = self.answer_templates["default"]
            answer = np.random.choice(answers)
        
        return answer
    
    def reset(self):
        """Reset model state if needed."""
        pass


def test_vqa_adapter():
    """Test the VQA adapter implementation."""
    
    print("=== Testing SimpleVQAAdapter ===\n")
    
    # Create adapter
    adapter = SimpleVQAAdapter(max_answer_length=100)
    
    # Initialize
    adapter.initialize()
    
    # Test model info
    info = adapter.get_model_info()
    print(f"Model info: {info}\n")
    
    # Test robot_vqa style observation
    print("--- Testing robot_vqa ---")
    robot_image = Image.new('RGB', (256, 256), color='blue')
    
    robot_vqa_observation = {
        'image': robot_image,
        'text_observation': 'What object is the robot arm holding?'
    }
    
    answer = adapter.predict_action(
        robot_vqa_observation,
        dataset_name="robot_vqa"
    )
    print(f"Question: {robot_vqa_observation['text_observation']}")
    print(f"Answer: {answer}\n")
    
    # Test sqa3d style observation
    print("--- Testing sqa3d ---")
    scene_image = Image.new('RGB', (480, 640), color='red')
    
    sqa3d_observation = {
        'image': scene_image,
        'question': 'How many chairs are in the room?'
    }
    
    answer = adapter.predict_action(
        sqa3d_observation,
        dataset_name="sqa3d"
    )
    print(f"Question: {sqa3d_observation['question']}")
    print(f"Answer: {answer}\n")
    
    # Test batch processing
    print("--- Testing batch VQA ---")
    batch = {
        'image_observation': [robot_image, scene_image, robot_image],
        'text_observation': [
            'Is the robot arm extended?',
            'Where is the cup located?',
            'What color is the gripper?'
        ]
    }
    
    batch_results = adapter.batch_predict_actions(batch, dataset_name="robot_vqa")
    print(f"Batch results: {len(batch_results)} predictions")
    for i, (question, answer) in enumerate(zip(batch['text_observation'], batch_results)):
        print(f"  Q{i+1}: {question}")
        print(f"  A{i+1}: {answer}")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Test missing question
        bad_obs = {'image': robot_image}
        adapter.predict_action(bad_obs, dataset_name="robot_vqa")
    except Exception as e:
        print(f"Expected error for missing question: {e}")
    
    # Test multiple predictions to show variation
    print("\n--- Testing prediction variation ---")
    test_question = "What is visible in the image?"
    test_obs = {'image': robot_image, 'text_observation': test_question}
    
    for i in range(3):
        answer = adapter.predict_action(test_obs, dataset_name="robot_vqa")
        print(f"Prediction {i+1}: {answer}")
        
    print("\n=== VQA adapter test completed! ===")


if __name__ == "__main__":
    test_vqa_adapter()

