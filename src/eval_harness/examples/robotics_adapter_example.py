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
            supported_datasets=["openx", "locomujoco", "agibot"],
            action_dim=action_dim,
        )
        self.model = None
        
    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the robotics model."""
        print(f"Initializing {self.model_name} with {self.action_dim}D action space")
        
        # Mock model initialization
        self.model = MockRoboticsModel(action_dim=self.action_dim)
        self.device = device
        self._is_initialized = True
        self.seed = seed
        print("Robotics model initialized successfully!")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Predict continuous action for robotics."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        # Validate dataset
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        # Extract components from observation
        image = observation.get('image', None)
        continuous_obs = observation.get('continuous_observation', None)
        
        # Preprocess observation
        processed_obs = self.preprocess_observation(observation, dataset_name or "openx")
        
        # Run inference
        if instruction:
            # Instruction-following robotics
            action_vector = self.model.predict_with_instruction(
                processed_obs['image'], instruction, continuous_obs
            )
        else:
            # No instructions provided
            action_vector = self.model.predict_action(
                processed_obs['image'], continuous_obs
            )
        
        # Validate action is finite
        if not np.isfinite(action_vector).all():
            print(f"Warning: Model predicted non-finite action, using zeros")
            action_vector = np.zeros(self.action_dim)
        
        return action_vector
        
    def batch_predict_actions(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> List[np.ndarray]:
        """Predict actions for a batch of robotics observations."""
        
        # Extract batch components
        images = batch.get('image_observation', [])
        continuous_obs = batch.get('continuous_observation', [])
        text_obs = batch.get('text_observation', [])
        
        batch_size = len(images) if images else len(continuous_obs) if continuous_obs else 1
        
        # For this example, process sequentially
        # Real implementation could use true batch processing for efficiency
        results = []
        for i in range(batch_size):
            try:
                # Build observation for this item
                observation = {}
                if images and i < len(images):
                    observation['image'] = images[i]
                if continuous_obs and i < len(continuous_obs):
                    observation['continuous_observation'] = continuous_obs[i]
                
                # Get instruction if available
                instruction = None
                if text_obs and i < len(text_obs):
                    instruction = text_obs[i]
                
                # Predict action
                action = self.predict_action(observation, instruction, **kwargs)
                results.append(action)
                
            except Exception as e:
                # For failed predictions, return zero action
                print(f"Warning: Prediction failed: {e}")
                results.append(np.zeros(self.action_dim))
            
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
        if 'image' in processed_obs:
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
        
        # Preprocess continuous observations
        if 'continuous_observation' in processed_obs:
            continuous_obs = processed_obs['continuous_observation']
            if isinstance(continuous_obs, (list, tuple)):
                continuous_obs = np.array(continuous_obs, dtype=np.float32)
            elif isinstance(continuous_obs, np.ndarray):
                continuous_obs = continuous_obs.astype(np.float32)
            else:
                continuous_obs = np.array([float(continuous_obs)], dtype=np.float32)
            
            # Ensure it's 1D
            if continuous_obs.ndim > 1:
                continuous_obs = continuous_obs.flatten()
                
            processed_obs['continuous_observation'] = continuous_obs
                        
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
        image: Optional[Image.Image], 
        continuous_obs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Mock action prediction."""
        
        # Generate random action vector
        action_vector = np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
        
        # Incorporate continuous observation if available
        if continuous_obs is not None:
            # Simple heuristic: add some influence from continuous observation
            obs_influence = np.mean(continuous_obs) * 0.1
            action_vector += obs_influence
            
        # Clip to reasonable range
        action_vector = np.clip(action_vector, -2.0, 2.0)
        
        return action_vector
        
    def predict_with_instruction(
        self, 
        image: Optional[Image.Image], 
        instruction: str,
        continuous_obs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Mock instruction-following prediction."""
        
        # Generate base action vector
        action_vector = self.predict_action(image, continuous_obs)
        
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
        
        return action_vector


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
    continuous_obs = np.array([0.1, -0.5, 0.8, 0.2, -0.3, 0.6, 0.4, -0.1, 0.9, 0.0, -0.7, 0.3], dtype=np.float32)
    
    openx_observation = {
        'image': robot_image,
        'continuous_observation': continuous_obs
    }
    
    action = adapter.predict_action(
        openx_observation,
        dataset_name="openx"
    )
    print(f"OpenX action: {action}")
    print(f"Action shape: {action.shape}")
    
    # Test with instruction
    action_with_instruction = adapter.predict_action(
        openx_observation,
        instruction="Move the robot arm to grasp the object",
        dataset_name="openx"
    )
    print(f"With instruction: {action_with_instruction}")
    
    # Test batch processing
    print("\n--- Testing batch robotics ---")
    batch = {
        'image_observation': [robot_image, robot_image],
        'continuous_observation': [continuous_obs, continuous_obs * 0.5],
        'text_observation': [None, "Rotate the end effector"]
    }
    
    batch_results = adapter.batch_predict_actions(batch, dataset_name="openx")
    print(f"Batch results: {len(batch_results)} predictions")
    for i, result in enumerate(batch_results):
        print(f"  Prediction {i+1}: action_shape={result.shape}")
    
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
        action = adapter.predict_action(openx_observation, dataset_name="openx")
        print(f"Prediction {i+1}: mean={np.mean(action):.3f}, std={np.std(action):.3f}")
        
    print("\n=== Robotics adapter test completed! ===")


if __name__ == "__main__":
    test_robotics_adapter()
