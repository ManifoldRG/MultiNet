"""
VQA Model Adapter Example

This example demonstrates how to implement the ModelAdapter interface
for Visual Question Answering tasks like robot_vqa and sqa3d.
"""
import os, sys
# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import re
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from src.eval_harness.model_adapter import ModelAdapter


class SimpleVQAAdapter(ModelAdapter):
    """
    Example adapter for Visual Question Answering tasks.
    
    This shows how to implement a model for VQA tasks like robot_vqa and sqa3d
    where the model takes an image and question as input and generates a text answer.
    """
    
    def __init__(self, max_answer_length: int = 100):
        super().__init__()
        self.model_name = "SimpleVQAModel"
        self.model_type = "text_generation"
        self.max_answer_length = max_answer_length
        self.model = None

    @property
    def supported_datasets(self) -> List[str]:
        return ["robot_vqa", "sqa3d"]

    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the VQA model."""
        print(f"Initializing {self.model_name}")
        
        # Mock model initialization
        self.model = MockVQAModel(max_length=self.max_answer_length)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("VQA model initialized")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text answer for a visual question."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract image from standardized observation
        image = observation.get('image_observation', None)
        
        # Question comes from instruction parameter
        if instruction is None:
            raise ValueError("No instruction provided. VQA tasks require an instruction (the question).")
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation, dataset_name or "robot_vqa")
        
        # Run inference
        answer = self.model.answer_question(
            processed_obs.get('image_observation'),
            instruction
        )
        
        # Validate answer is a string
        if not isinstance(answer, str):
            print(f"Warning: Model returned non-string answer, converting to string")
            answer = str(answer)
        
        # Truncate if needed
        if len(answer) > self.max_answer_length:
            answer = answer[:self.max_answer_length]
        
        # Normalize the answer (adapter's responsibility)
        normalized_answer = self._normalize_text(answer)
        
        return {
            "raw_output": answer,  # Keep original for debugging
            "extracted_outputs": normalized_answer  # Normalized for fair comparison
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing punctuation and extra spaces."""
        if not isinstance(text, str):
            return ""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Predict answers for a batch of VQA observations."""
        
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
                
                # Predict answer
                result = self.predict_action(observation, instruction, dataset_name, **kwargs)
                results.append(result)
                
            except Exception as e:
                # For failed predictions, return empty string
                print(f"Warning: Prediction failed for item {i}: {e}")
                results.append({
                    "raw_output": f"Error: {str(e)}",
                    "extracted_outputs": ""
                })
            
        return results
        
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess observation for VQA."""
        
        processed_obs = observation.copy()
        
        # Preprocess image from standardized key
        if 'image_observation' in processed_obs:
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
        instruction=robot_vqa_observation['text_observation'],
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
        instruction=sqa3d_observation['question'],
        dataset_name="sqa3d"
    )
    print(f"Question: {sqa3d_observation['question']}")
    print(f"Answer: {answer}\n")
    
    # Test batch processing
    print("--- Testing batch VQA ---")
    batch_observations = [
        {'image_observation': robot_image},
        {'image_observation': scene_image},
        {'image_observation': robot_image}
    ]
    batch_instructions = [
        'Is the robot arm extended?',
        'Where is the cup located?',
        'What color is the gripper?'
    ]
    
    batch_results = adapter.batch_predict_actions(
        batch_observations, 
        instructions=batch_instructions,
        dataset_name="robot_vqa"
    )
    print(f"Batch results: {len(batch_results)} predictions")
    for i, (question, answer) in enumerate(zip(batch_instructions, batch_results)):
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
        answer = adapter.predict_action(
            test_obs, 
            instruction=test_question,
            dataset_name="robot_vqa"
        )
        print(f"Prediction {i+1}: {answer}")
        
    print("\n=== VQA adapter test completed! ===")


if __name__ == "__main__":
    test_vqa_adapter()

