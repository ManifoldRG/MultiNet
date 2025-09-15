#!/usr/bin/env python3
"""
Test script for Overcooked Pi0 inference pipeline.
This script validates all components work correctly before running on production data.
"""

import os
import sys
import tempfile
import pickle
import base64
import io
import numpy as np
from PIL import Image
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from src.eval.profiling.openpi.scripts.overcooked_inference import OvercookedInference, parse_args
from src.eval.profiling.openpi.src.openpi.models import pi0
from src.eval.profiling.openpi.src.openpi.models import model as _model
from src.eval.profiling.openpi.src.openpi.models.tokenizer import PaligemmaTokenizer
from src.eval.profiling.openpi.src.openpi.shared import download
from src.data_utils.overcooked_dataloader import get_overcooked_dataloader
import jax


def create_dummy_image_base64():
    """Create a dummy RGB image encoded as base64 string."""
    # Create a simple 64x64 RGB image
    image = Image.new('RGB', (64, 64), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    
    return base64_string


def create_test_data():
    """Create test data in the same format as the real Overcooked dataset."""
    test_data = []
    
    # Create 10 timesteps across 2 episodes
    for episode_id in range(2):
        for timestep in range(5):
            # Create dummy timestep data
            # Mix different action types for better testing
            if timestep == 0:
                joint_action = str(['interact', [1, 0]])  # Action 28: INTERACT-EAST
            elif timestep == 1:
                joint_action = str([[0, -1], 'interact'])  # Action 30: NORTH-INTERACT  
            elif timestep == 2:
                joint_action = str(['interact', 'interact'])  # Action 35: INTERACT-INTERACT
            elif timestep == 3:
                joint_action = str([[0, 0], [0, 0]])  # Action 24: STAY-STAY
            else:
                joint_action = str([[0, -1], [1, 0]])  # Action 2: NORTH-EAST
                
            timestep_data = {
                'trial_id': f'test_episode_{episode_id}',
                'state': create_dummy_image_base64(),
                'joint_action': joint_action,
                'reward': 0.0,
                'score': float(timestep * 10),
                'time_left': float(300 - timestep * 10),
                'time_elapsed': float(timestep * 10),
                'layout_name': 'cramped_room'
            }
            test_data.append(timestep_data)
    
    return test_data


def create_test_pickle_file():
    """Create a temporary pickle file with test data."""
    test_data = create_test_data()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(test_data, f)
        return f.name


def test_discrete_action_mapping():
    """Test the discrete action mapping in the dataloader."""
    print("Testing discrete action mapping...")
    
    # Create test data file to initialize dataset
    test_file = create_test_pickle_file()
    
    try:
        # Create dataset to access action mappings
        dataset, _ = get_overcooked_dataloader(test_file, batch_size=1, by_episode=False)
        
        # Test key action mappings
        test_cases = [
            # Movement combinations
            (((0, 0), (0, 0)), 24, "STAY-STAY"),           # Both STAY 
            (((0, -1), (0, 1)), 1, "NORTH-SOUTH"),         # Different movements
            (((1, 0), (-1, 0)), 13, "EAST-WEST"),          # Opposite movements
            
            # Interact combinations  
            (('interact', (0, 0)), 29, "INTERACT-STAY"),   # Player 0 interact
            (((0, 0), 'interact'), 34, "STAY-INTERACT"),   # Player 1 interact
            (('interact', 'interact'), 35, "INTERACT-INTERACT"), # Both interact
        ]
        
        for joint_action, expected_idx, description in test_cases:
            result = dataset.joint_to_discrete[joint_action]
            reverse = dataset.discrete_to_joint[expected_idx]
            print(f"{description}: {joint_action} -> {result} (expected {expected_idx})")
            print(f"  Reverse: {expected_idx} -> {reverse}")
            assert result == expected_idx, f"Expected {expected_idx}, got {result}"
            assert reverse == joint_action, f"Reverse mapping failed: {reverse} != {joint_action}"
        
        print(f"Action mapping tests passed! Total actions: {dataset.num_discrete_actions}")
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_dataloader():
    """Test the Overcooked dataloader with dummy data."""
    print("\nTesting dataloader...")
    
    # Create test data file
    test_file = create_test_pickle_file()
    
    try:
        # Create dataloader
        dataset, dataloader = get_overcooked_dataloader(test_file, batch_size=2, by_episode=False)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Action stats: {dataset.action_stats}")
        
        # Test a few batches
        batch_count = 0
        for batch in dataloader:
            print(f"\nBatch {batch_count}:")
            print(f"  Image observations: {len(batch['image_observation'])}")
            print(f"  Text observations: {batch['text_observation']}")
            print(f"  Actions: {batch['action']}")
            print(f"  Rewards: {batch['reward']}")
            
            batch_count += 1
            if batch_count >= 2:  # Only test first 2 batches
                break
        
        print("Dataloader tests passed!")
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_observation_preparation():
    """Test observation preparation for Pi0 model."""
    print("\nTesting observation preparation...")
    
    # Create test data
    test_file = create_test_pickle_file()
    
    try:
        # Initialize inference object
        config = pi0.Pi0Config(action_horizon=1)
        tokenizer = PaligemmaTokenizer()
        inference = OvercookedInference(None, tokenizer, config)
        
        # Create dataloader
        dataset, dataloader = get_overcooked_dataloader(test_file, batch_size=2, by_episode=False)
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Test observation preparation
        obs = {
            'image_observation': batch['image_observation'],
            'text_observation': batch['text_observation']
        }
        
        transformed_dict = inference.prepare_observation(obs, batch_size=len(batch['image_observation']))
        
        print("Transformed observation keys:", list(transformed_dict.keys()))
        print("Image shapes:", {k: v.shape for k, v in transformed_dict['image'].items()})
        print("Image mask shapes:", {k: v.shape for k, v in transformed_dict['image_mask'].items()})
        print("State shape:", transformed_dict['state'].shape)
        print("Tokenized prompt shape:", transformed_dict['tokenized_prompt'].shape)
        
        # Verify expected shapes
        batch_size = len(batch['image_observation'])
        assert transformed_dict['state'].shape == (batch_size, 32), f"Expected state shape ({batch_size}, 32)"
        assert transformed_dict['tokenized_prompt'].shape[0] == batch_size, "Batch size mismatch in tokenized prompt"
        
        print("✓ Observation preparation tests passed!")
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_normalization_pipeline():
    """Test the normalization statistics and unnormalization pipeline."""
    print("\nTesting normalization pipeline...")
    
    test_file = create_test_pickle_file()
    
    try:
        # Initialize components
        config = pi0.Pi0Config(action_horizon=1)
        tokenizer = PaligemmaTokenizer()
        inference = OvercookedInference(None, tokenizer, config)
        
        # Create dataloader
        dataset, dataloader = get_overcooked_dataloader(test_file, batch_size=2, by_episode=False)
        
        # Test dataset statistics calculation
        dataset_stats, _ = inference.get_dataset_stats(dataloader)
        
        print(f"Dataset stats keys: {list(dataset_stats.keys())}")
        print(f"Action norm stats: mean={dataset_stats['action'].mean}, std={dataset_stats['action'].std}")
        
        # Test process_output with dummy continuous actions
        batch_size = 2
        action_horizon = 1
        # Create actions centered around the dataset mean for realistic test
        mean_action = dataset_stats['action'].mean[0]
        dummy_actions = np.random.normal(loc=0.0, scale=0.1, size=(batch_size, action_horizon, 32))
        
        discrete_actions = inference.process_output(dummy_actions, dataset_stats)
        
        print(f"Continuous actions shape: {dummy_actions.shape}")
        print(f"Continuous actions (first dim): {dummy_actions[..., 0].flatten()}")
        print(f"Discrete actions: {discrete_actions.flatten()}")
        
        # Verify all actions are in valid range
        num_actions = inference.num_discrete_actions
        assert np.all(discrete_actions >= 0) and np.all(discrete_actions < num_actions), \
            f"Invalid discrete action indices, expected range [0, {num_actions})"
        
        print("Normalization pipeline tests passed!")
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def test_model_integration():
    """Test integration with Pi0 model components (without full model loading)."""
    print("\nTesting model integration...")
    
    test_file = create_test_pickle_file()
    
    try:
        # Initialize components like the main script does
        config = pi0.Pi0Config(action_horizon=1)
        tokenizer = PaligemmaTokenizer()
        key = jax.random.key(42)
        
        # Test if we can create the inference object properly
        inference = OvercookedInference(None, tokenizer, config)  # Model=None for testing
        
        # Verify action space matches expectation
        assert inference.num_discrete_actions == 36, f"Expected 36 actions, got {inference.num_discrete_actions}"
        
        # Test diagonal move detection setup
        inference._log_raw_outputs = True
        inference._diagonal_count = 0
        
        print(f"Action space: {inference.num_discrete_actions} discrete actions")
        print(f"Config: action_horizon={config.action_horizon}")
        print(f"Diagonal detection: enabled")
        
        print("Model integration tests passed!")
        
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)


def main():
    """Run all tests to validate the Overcooked Pi0 inference pipeline."""
    print("Running Overcooked Pi0 Inference Tests")
    print("======================================\n")
    
    try:
        test_discrete_action_mapping()
        test_dataloader()
        test_observation_preparation()
        test_normalization_pipeline()
        test_model_integration()
        
        print("\nAll tests passed! The inference pipeline is ready.")
        print("\nTo run full inference on real data:")
        print("   python overcooked_inference.py --data_file /path/to/data.pkl --output_dir /path/to/output --max_samples 100")
        print("\nExample with real Overcooked dataset:")
        print("   python overcooked_inference.py --data_file /path/to/2020_hh_trials_test.pickle --output_dir results --batch_size 5 --max_samples 200")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease fix the issues above before running full inference.")
        sys.exit(1)


if __name__ == "__main__":
    main()
