#!/usr/bin/env python3
"""
Test script for Overcooked inference with Pi0 model.
This script tests the core functionality before running on full datasets.
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
                joint_action = str(['interact', [1, 0]])  # Player 0: INTERACT, Player 1: EAST
            elif timestep == 1:
                joint_action = str([[0, -1], 'interact'])  # Player 0: NORTH, Player 1: INTERACT  
            elif timestep == 2:
                joint_action = str(['interact', 'interact'])  # Both INTERACT
            else:
                joint_action = str([[0, -1], [1, 0]])  # Player 0: NORTH, Player 1: EAST
                
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


def test_action_mapping():
    """Test the continuous to discrete action mapping."""
    print("Testing action mapping...")
    
    # Initialize model components
    config = pi0.Pi0Config(action_horizon=1)
    tokenizer = PaligemmaTokenizer()
    key = jax.random.key(0)
    
    # Create inference object (without loading full model for this test)
    inference = OvercookedInference(None, tokenizer, config)
    
    # Test continuous to discrete mapping
    test_cases = [
        (np.array([1.0, 0.0]), (1, 0)),    # EAST
        (np.array([-1.0, 0.0]), (-1, 0)),  # WEST
        (np.array([0.0, 1.0]), (0, 1)),    # SOUTH
        (np.array([0.0, -1.0]), (0, -1)),  # NORTH
        (np.array([0.0, 0.0]), (0, 0)),    # STAY
        (np.array([1.0, 1.0]), 'interact'), # INTERACT
    ]
    
    for continuous_action, expected_discrete in test_cases:
        result = inference._continuous_to_discrete_action(continuous_action)
        print(f"Continuous {continuous_action} -> Discrete {result} (expected {expected_discrete})")
        assert result == expected_discrete, f"Expected {expected_discrete}, got {result}"
    
    print("Action mapping tests passed!")


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


def test_full_inference():
    """Test full inference pipeline with dummy data (without loading full Pi0 model)."""
    print("\nTesting full inference pipeline (mocked)...")
    
    # Create test data
    test_file = create_test_pickle_file()
    
    try:
        # Initialize components
        config = pi0.Pi0Config(action_horizon=1)
        tokenizer = PaligemmaTokenizer()
        inference = OvercookedInference(None, tokenizer, config)
        
        # Create dataloader
        dataset, dataloader = get_overcooked_dataloader(test_file, batch_size=2, by_episode=False)
        
        # Test process_output with dummy continuous actions
        batch_size = 2
        action_horizon = 1
        dummy_actions = np.random.randn(batch_size, action_horizon, 32)  # Pi0 outputs 32-dim actions
        
        discrete_actions = inference.process_output(dummy_actions)
        
        print(f"Dummy continuous actions shape: {dummy_actions.shape}")
        print(f"Discrete actions shape: {discrete_actions.shape}")
        print(f"Sample discrete actions: {discrete_actions}")
        
        # Get the actual number of discrete actions from the inference object
        num_actions = inference.num_discrete_actions
        assert np.all(discrete_actions >= 0) and np.all(discrete_actions < num_actions), f"Invalid discrete action indices, expected range [0, {num_actions})"
        
        print("✓ Full inference pipeline tests passed!")
        
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.unlink(test_file)


def main():
    """Run all tests."""
    print("🧪 Running Overcooked inference tests...\n")
    
    try:
        test_action_mapping()
        test_dataloader()
        test_observation_preparation()
        test_full_inference()
        
        print("\nAll tests passed! The Overcooked inference script should work correctly.")
        print("\nTo run the full inference, use:")
        print("python overcooked_inference.py --output_dir /path/to/output --data_file /path/to/overcooked_data.pkl")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
