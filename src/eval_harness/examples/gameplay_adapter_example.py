"""
Gameplay Model Adapter Example

Demonstrates how to implement the DiscreteActionAdapter interface
for discrete action gameplay tasks like Procgen, OvercookedAI.
"""
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(ROOT_DIR)

import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union
from src.eval_harness.model_adapter import DiscreteActionAdapter


class SimpleGameplayAdapter(DiscreteActionAdapter):
    """Example adapter for gameplay tasks with discrete action spaces."""
    
    def __init__(self, action_space_size: int = 15):
        super().__init__(
            model_name="SimpleGameplayModel",
            supported_datasets=["procgen", "overcooked_ai"],
            action_space_size=action_space_size,
        )
        self.model = None
        
    def initialize(self, model_path: Optional[str] = None, device: str = "cpu", seed: int = 42, **kwargs):
        """Initialize the gameplay model."""
        print(f"Initializing {self.model_name} with {self.action_space_size} actions")
        
        self.model = MockGameplayModel(action_space_size=self.action_space_size)
        self.device = device
        self.set_seed(seed)
        self._is_initialized = True
        
        print("Gameplay model initialized")
        
    def predict_action(
        self,
        observation: Dict[str, Any],
        instruction: Optional[str] = None,
        dataset_name: Optional[str] = None,
        return_probabilities: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Predict discrete action for gameplay."""
        
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        if dataset_name and not self.is_dataset_supported(dataset_name):
            raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
        image = observation.get('image_observation', None)
        if image is None:
            raise ValueError("Gameplay task requires 'image_observation' in observation")
            
        processed_obs = self.preprocess_observation(observation, dataset_name or "overcooked")
        
        if dataset_name == "overcooked" and instruction:
            raw_output, action_idx, probabilities = self.model.predict_with_instruction(
                processed_obs['image_observation'], instruction
            )
        else:
            raw_output, action_idx, probabilities = self.model.predict_action(
                processed_obs['image_observation']
            )
        
        if action_idx >= self.action_space_size:
            print(f"Warning: Model predicted action {action_idx} >= {self.action_space_size}, clipping")
            action_idx = self.action_space_size - 1
        
        result = {
            "raw_output": raw_output,
            "extracted_outputs": action_idx
        }
        
        if return_probabilities:
            result["probabilities"] = probabilities
        
        return result
        
    def batch_predict_actions(
        self,
        observations: List[Dict[str, Any]],
        instructions: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        return_probabilities: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Predict actions for a batch of gameplay observations."""
        
        if instructions is None:
            instructions = [None] * len(observations)
            
        results = []
        for obs, instruction in zip(observations, instructions):
            result = self.predict_action(obs, instruction, dataset_name, return_probabilities, **kwargs)
            results.append(result)
            
        return results
        
    def preprocess_observation(
        self,
        observation: Dict[str, Any],
        dataset_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Preprocess observation for gameplay based on model requirements."""
        
        processed_obs = observation.copy()
        
        if 'image_observation' in processed_obs:
            image = processed_obs['image_observation']
            
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
                
            processed_obs['image_observation'] = image
                        
        return processed_obs
        
    def reset(self):
        """Reset gameplay state."""
        if self.model and hasattr(self.model, 'reset'):
            self.model.reset()
        print("Gameplay model state reset")


class MockGameplayModel:
    """Mock gameplay model for testing."""
    
    def __init__(self, action_space_size: int = 15):
        self.action_space_size = action_space_size
        
    def predict_action(
        self, 
        image: Image.Image, 
        state: Optional[np.ndarray] = None
    ) -> tuple[str, int, np.ndarray]:
        """Mock action prediction."""
        
        logits = np.random.uniform(-1, 1, self.action_space_size)
        if state is not None:
            logits += np.sum(state) * 0.1
            
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        action_idx = np.argmax(probabilities)
        
        raw_output = f"Action {action_idx} selected with probability {probabilities[action_idx]:.3f}"

        return raw_output, action_idx, probabilities
        
    def predict_with_instruction(
        self, 
        image: Image.Image, 
        instruction: str,
        state: Optional[np.ndarray] = None
    ) -> tuple[str, int, np.ndarray]:
        """Mock instruction-following prediction."""
        
        logits = np.random.uniform(-1, 1, self.action_space_size)
        
        if "move" in instruction.lower():
            logits[0:4] += 1.0
        if "pick" in instruction.lower() or "take" in instruction.lower():
            logits[4:8] += 1.0
            
        if state is not None:
            logits += np.sum(state) * 0.1
            
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        action_idx = np.argmax(probabilities)
        
        raw_output = f"Following instruction '{instruction}': Action {action_idx} selected"

        return raw_output, action_idx, probabilities


def test_gameplay_adapter():
    """Test the gameplay adapter implementation."""
    
    print("=== Testing SimpleGameplayAdapter for OvercookedAI ===\n")
    
    adapter = SimpleGameplayAdapter(action_space_size=6)
    adapter.initialize()
    
    info = adapter.get_model_info()
    print(f"Model info: {info}\n")
    
    print("--- Testing OvercookedAI gameplay ---")
    game_image = Image.new('RGB', (675, 375), color=(255, 255, 255))

    overcooked_ai_observation = {
        'image_observation': game_image,
        'text_observation': "fulfill the order",
        'time_left': 100
    }
    
    action = adapter.predict_action(
        overcooked_ai_observation,
        dataset_name="overcooked_ai",
        return_probabilities=False
    )
    print(f"OvercookedAI action: {action}")
    
    result = adapter.predict_action(
        overcooked_ai_observation,
        dataset_name="overcooked_ai",
        return_probabilities=True
    )
    print(f"With probabilities: action={result['action']}, top_prob={result['probabilities'].max():.3f}")

    print("\n--- Testing batch gameplay ---")
    batch_observations = [overcooked_ai_observation, overcooked_ai_observation]
    batch_instructions = [None, overcooked_ai_observation['text_observation']]
    
    batch_results = adapter.batch_predict_actions(
        batch_observations,
        batch_instructions,
        dataset_name="overcooked_ai"
    )
    print(f"Batch results: {batch_results}")
    
    print("--- Testing error handling ---")
    
    try:
        bad_obs = {'state': np.array([1, 2, 3])}
        adapter.predict_action(bad_obs, dataset_name="overcooked_ai")
    except ValueError as e:
        print(f"Expected error for missing image: {e}")
        
    print("\n--- Testing prediction consistency ---")
    for i in range(3):
        action = adapter.predict_action(overcooked_ai_observation, dataset_name="overcooked_ai")
        print(f"Prediction {i+1}: {action}")
        
    print("\n=== Gameplay adapter test completed! ===")


if __name__ == "__main__":
    test_gameplay_adapter()
