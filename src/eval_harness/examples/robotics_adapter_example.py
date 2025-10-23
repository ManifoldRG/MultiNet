"""
Robotics Model Adapter Example

This example demonstrates how to implement the ModelAdapter interface
for continuous action robotics tasks like OpenX, Locomujoco, Agibot.
"""
import os, sys
# Adding the root directory to the system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union
from src.eval_harness.model_adapter import ContinuousActionAdapter


class SimpleRoboticsAdapter(ContinuousActionAdapter):
    """
    Example adapter for robotics tasks with continuous action spaces.
    
    This shows how to implement a model for robotics tasks with continuous actions
    like OpenX environments.
    """
    
    def __init__(self, action_dim: int = 7):
        super().__init__(
            model_name="SimpleRoboticsModel",
            supported_datasets=["openx"],
            action_dim=action_dim,
        )
        self.model = None
        
    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the robotics model."""
        print(f"Initializing {self.model_name} with {self.action_dim}D action space")
        
        # Mock model initialization
        self.model = MockRoboticsModel(action_dim=self.action_dim)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        print("Robotics model initialized")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict continuous action for robotics."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract components from standardized observation
        image = observation.get('image_observation', None)
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation, dataset_name or "openx")
        
        # Run inference
        if instruction:
            # Instruction-following robotics
            raw_output, action_vector = self.model.predict_with_instruction(
                processed_obs['image_observation'], instruction
            )
        else:
            # No instructions provided
            raw_output, action_vector = self.model.predict_action(
                processed_obs['image_observation']
            )
        
        # Validate action is finite
        if not np.isfinite(action_vector).all():
            print(f"Warning: Model predicted non-finite action, using zeros")
            action_vector = np.zeros(self.action_dim)
        
        return {
            "raw_output": raw_output,
            "extracted_outputs": action_vector
        }
        
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        histories: Optional[List[List[Dict[str, str]]]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Predict actions for a batch of robotics observations."""
        
        batch_size = len(observations)
        
        # For this example, process sequentially
        # Real implementation could use batch processing
        results = []
        for i in range(batch_size):
            try:
                # Get observation for this item
                observation = observations[i]
                
                # Get instruction if available
                instruction = instructions[i] if instructions else None
                
                # Predict action
                result = self.predict_action(observation, instruction, **kwargs)
                results.append(result)
                
            except Exception as e:
                # For failed predictions, return zero action
                print(f"Warning: Prediction failed: {e}")
                results.append({
                    "raw_output": f"Error: {str(e)}",
                    "extracted_outputs": np.zeros(self.action_dim)
                })
            
        return results
        
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess observation for robotics."""
        
        processed_obs = observation.copy()
        
        # Preprocess image
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
        
    def _get_target_size(self, dataset_name: str) -> tuple[int, int]:
        """Get target image size for different datasets."""
        size_map = {
            "openx": (256, 256),
            "locomujoco": (64, 64), 
            "agibot": (224, 224)
        }
        return size_map.get(dataset_name, (256, 256))
        
    def reset(self):
        """Reset robotics state."""
        if self.model and hasattr(self.model, 'reset'):
            self.model.reset()
        print("Robotics model state reset")


class MockRoboticsModel:
    """Mock robotics model for testing."""
    
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim
        
    def predict_action(
        self, 
        image: Optional[Image.Image]
    ) -> tuple[str, np.ndarray]:
        """Mock action prediction."""
        
        # Generate random action vector
        action_vector = np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        
        # Clip to reasonable range
        action_vector = np.clip(action_vector, -2.0, 2.0)
        
        # Generate raw output text
        raw_output = f"Action: {action_vector.tolist()}"
        
        return raw_output, action_vector
        
    def predict_with_instruction(
        self, 
        image: Optional[Image.Image], 
        instruction: str
    ) -> tuple[str, np.ndarray]:
        """Mock instruction-following prediction."""
        
        # Generate base action vector
        _, action_vector = self.predict_action(image)
        
        # Simple instruction influence (in practice would use NLP)
        if "move" in instruction.lower():
            # Boost movement actions (first few dimensions)
            action_vector[:3] += 0.5
        if "grasp" in instruction.lower() or "pick" in instruction.lower():
            # Boost grasping actions (middle dimensions)
            action_vector[3:5] += 0.5
        if "rotate" in instruction.lower():
            # Boost rotation actions (last dimensions)
            action_vector[-2:] += 0.5
            
        # Clip to reasonable range
        action_vector = np.clip(action_vector, -2.0, 2.0)
        
        # Generate raw output text
        raw_output = f"Following instruction '{instruction}': Action {action_vector.tolist()}"
        
        return raw_output, action_vector


def test_robotics_adapter():
    """Test the robotics adapter implementation."""
    
    print("=== Testing SimpleRoboticsAdapter ===\n")
    
    # Create adapter
    adapter = SimpleRoboticsAdapter(action_dim=7)
    
    # Initialize
    adapter.initialize()
    
    # Test model info
    info = adapter.get_model_info()
    print(f"Model info: {info}\n")
    
    # Test standard robotics (OpenX)
    print("--- Testing OpenX robotics ---")
    robot_image = Image.new('RGB', (256, 256), color='green')
    
    openx_observation = {
        'image_observation': robot_image
    }
    
    result = adapter.predict_action(
        openx_observation,
        dataset_name="openx"
    )
    print(f"Raw output: {result['raw_output']}")
    print(f"Action: {result['extracted_outputs']}")
    print(f"Action shape: {result['extracted_outputs'].shape}")
    
    # Test with instruction
    result_with_instruction = adapter.predict_action(
        openx_observation,
        instruction="Move the robot arm to grasp the object",
        dataset_name="openx"
    )
    print(f"Raw output: {result_with_instruction['raw_output']}")
    print(f"Action: {result_with_instruction['extracted_outputs']}")
    
    # Test batch processing
    print("\n--- Testing batch robotics ---")
    observations = [
        {'image_observation': robot_image},
        {'image_observation': robot_image}
    ]
    instructions = [None, "Rotate the end effector"]
    
    batch_results = adapter.batch_predict_actions(
        observations=observations,
        instructions=instructions,
        dataset_name="openx"
    )
    print(f"Batch results: {len(batch_results)} predictions")
    for i, result in enumerate(batch_results):
        print(f"  Prediction {i+1}:")
        print(f"    Raw output: {result['raw_output']}")
        print(f"    Action shape: {result['extracted_outputs'].shape}")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Test missing observation
        bad_obs = {}
        adapter.predict_action(bad_obs, dataset_name="openx")
    except Exception as e:
        print(f"Expected error for missing observation: {e}")
        
    # Test multiple predictions to show consistency
    print("\n--- Testing prediction consistency ---")
    for i in range(3):
        result = adapter.predict_action(openx_observation, dataset_name="openx")
        action = result['extracted_outputs']
        print(f"Prediction {i+1}: mean={np.mean(action):.3f}, std={np.std(action):.3f}")
        
    print("\n=== Robotics adapter test completed! ===")


if __name__ == "__main__":
    test_robotics_adapter()
