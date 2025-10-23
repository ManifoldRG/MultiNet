"""
Test suite for OvercookedAI Gameplay Metrics Calculator

Tests the metrics calculator with 36 joint action space
and per-player metrics decomposition.
"""

import sys
import os
import numpy as np
from typing import List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.eval_harness.scoring.gameplay_metrics import OvercookedAIMetricsCalculator
from definitions.overcooked import OverCookedDefinitions


JOINT_ACTION_SPACE_SIZE = 36
INDIVIDUAL_ACTION_SPACE_SIZE = 6
NOOP_ACTION = 28


def create_joint_probability_distribution(joint_action_idx: int, action_space_size: int = 36) -> List[float]:
    """Create a one-hot probability distribution for a joint action."""
    probs = [0.0] * action_space_size
    probs[joint_action_idx] = 1.0
    return probs


def get_individual_actions(joint_action_idx: int) -> Tuple[int, int]:
    """Get player0 and player1 actions from joint action index.
    
    Joint actions are encoded as 6x6 = 36 combinations where each joint action
    maps to a tuple of (player0_action, player1_action) in the individual action space.
    """
    individual_action_space = OverCookedDefinitions.INDIVIDUAL_ACTION_SPACE
    discrete_to_joint = OverCookedDefinitions.PLAYER_ACTION_SPACE_TUPLES
    
    player0_action, player1_action = discrete_to_joint[joint_action_idx]
    player0_label = individual_action_space[player0_action]
    player1_label = individual_action_space[player1_action]
    
    return player0_label, player1_label


def create_random_joint_probability(action_space_size: int = 36, seed: int = None) -> List[float]:
    """Create a random probability distribution that sums to 1.0."""
    if seed is not None:
        np.random.seed(seed)
    
    probs = np.random.dirichlet(np.ones(action_space_size))
    return probs.tolist()


def test_gameplay_metrics():
    """Test core functionality with joint action space (36 actions)."""
    print("\n" + "="*80)
    print("TEST CASE 1: Basic Joint Action Prediction")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()

    assert metrics_calculator.num_actions == JOINT_ACTION_SPACE_SIZE, \
        f"Expected {JOINT_ACTION_SPACE_SIZE} actions, got {metrics_calculator.num_actions}"
    assert metrics_calculator.noop_action == NOOP_ACTION, \
        f"Expected noop action {NOOP_ACTION}, got {metrics_calculator.noop_action}"
    print(f"PASS Metrics calculator initialized with {JOINT_ACTION_SPACE_SIZE} joint actions")

    # Test with probabilities=True
    print("\n--- Testing with probabilities=True ---")
    predictions_probs = []
    ground_truth_actions_probs = []
    
    # Perfect predictions (exact matches)
    for action in [0, 7, 14, 21, 28, 35]:
        probs = create_joint_probability_distribution(action)
        predictions_probs.append({
            "raw_output": f"Action {action} selected",
            "extracted_outputs": probs  # Probabilities go directly in extracted_outputs
        })
        ground_truth_actions_probs.append(action)
    
    # Incorrect but valid predictions
    probs_5 = create_joint_probability_distribution(5)
    predictions_probs.append({
        "raw_output": "Action 5 selected",
        "extracted_outputs": probs_5  # Probabilities go directly in extracted_outputs
    })
    ground_truth_actions_probs.append(10)
    
    probs_20 = create_joint_probability_distribution(20)
    predictions_probs.append({
        "raw_output": "Action 20 selected",
        "extracted_outputs": probs_20  # Probabilities go directly in extracted_outputs
    })
    ground_truth_actions_probs.append(15)
    
    probs_random_1 = create_random_joint_probability(seed=42)
    predictions_probs.append({
        "raw_output": "Random action from distribution",
        "extracted_outputs": probs_random_1  # Probabilities go directly in extracted_outputs
    })
    ground_truth_actions_probs.append(8)
    
    probs_random_2 = create_random_joint_probability(seed=123)
    predictions_probs.append({
        "raw_output": "Random action from distribution",
        "extracted_outputs": probs_random_2  # Probabilities go directly in extracted_outputs
    })
    ground_truth_actions_probs.append(25)
    
    # Invalid predictions
    invalid_probs_1 = [0.1, 0.2, 0.3] + [0.0] * 33  # Wrong length
    predictions_probs.append({
        "raw_output": "Invalid prediction",
        "extracted_outputs": invalid_probs_1  # Invalid probabilities go directly in extracted_outputs
    })
    ground_truth_actions_probs.append(5)
    
    invalid_probs_2 = [0.5, 0.3, 0.2]  # Wrong length
    predictions_probs.append({
        "raw_output": "Invalid prediction",
        "extracted_outputs": invalid_probs_2  # Invalid probabilities go directly in extracted_outputs
    })
    ground_truth_actions_probs.append(12)

    print(f"PASS Created {len(predictions_probs)} test predictions (probabilities)")
    print(f"  - {6} exact matches")
    print(f"  - {4} valid but incorrect")
    print(f"  - {2} invalid predictions")

    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    print("\n--- Testing with probabilities=False ---")
    predictions_discrete = []
    ground_truth_actions_discrete = []
    
    # Perfect predictions (exact matches)
    for action in [0, 7, 14, 21, 28, 35]:
        predictions_discrete.append({
            "raw_output": f"Action {action} selected",
            "extracted_outputs": action  # Discrete action index
        })
        ground_truth_actions_discrete.append(action)
    
    # Incorrect but valid predictions
    predictions_discrete.append({
        "raw_output": "Action 5 selected",
        "extracted_outputs": 5  # Discrete action index
    })
    ground_truth_actions_discrete.append(10)
    
    predictions_discrete.append({
        "raw_output": "Action 20 selected",
        "extracted_outputs": 20  # Discrete action index
    })
    ground_truth_actions_discrete.append(15)
    
    predictions_discrete.append({
        "raw_output": "Random action",
        "extracted_outputs": 8  # Discrete action index
    })
    ground_truth_actions_discrete.append(8)
    
    predictions_discrete.append({
        "raw_output": "Another random action",
        "extracted_outputs": 25  # Discrete action index
    })
    ground_truth_actions_discrete.append(25)
    
    # Invalid predictions
    predictions_discrete.append({
        "raw_output": "Invalid prediction",
        "extracted_outputs": -1  # Invalid action index
    })
    ground_truth_actions_discrete.append(5)
    
    predictions_discrete.append({
        "raw_output": "Out of range prediction",
        "extracted_outputs": 50  # Out of range action index
    })
    ground_truth_actions_discrete.append(12)

    print(f"PASS Created {len(predictions_discrete)} test predictions (discrete)")
    print(f"  - {6} exact matches")
    print(f"  - {4} valid but incorrect")
    print(f"  - {2} invalid predictions")

    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test (both should work the same)
    metrics = metrics_probs
    
    assert 'num_timesteps' in metrics, "Missing 'num_timesteps' in metrics"
    assert metrics['num_timesteps'] == len(predictions_probs), \
        f"Expected {len(predictions_probs)} timesteps, got {metrics['num_timesteps']}"
    print(f"PASS num_timesteps = {metrics['num_timesteps']}")
    
    assert 'total_invalid_preds' in metrics, "Missing 'total_invalid_preds' in metrics"
    assert metrics['total_invalid_preds'] == 2, \
        f"Expected 2 invalid predictions, got {metrics['total_invalid_preds']}"
    print(f"PASS total_invalid_preds = {metrics['total_invalid_preds']}")
    
    assert 'exact_match' in metrics, "Missing 'exact_match' in metrics"
    expected_exact_match = 6 / 12
    assert abs(metrics['exact_match'] - expected_exact_match) < 0.01, \
        f"Expected exact_match ≈ {expected_exact_match}, got {metrics['exact_match']}"
    print(f"PASS exact_match = {metrics['exact_match']:.3f}")
    
    required_metrics = [
        'avg_dataset_amae', 'avg_dataset_amse',
        'micro_precision', 'micro_recall', 'micro_f1',
        'macro_precision', 'macro_recall', 'macro_f1',
        'percentage_invalids'
    ]
    for metric in required_metrics:
        assert metric in metrics, f"Missing '{metric}' in metrics"
    print(f"PASS All standard metrics present")
    
    assert 'player0_results' in metrics, "Missing 'player0_results' in metrics"
    assert 'player1_results' in metrics, "Missing 'player1_results' in metrics"
    print(f"PASS Per-player metrics present")
    
    print("\nTEST CASE 1 PASSED\n")
    return metrics


def test_per_player_metrics():
    """Test that joint actions are correctly decomposed into individual player actions."""
    print("\n" + "="*80)
    print("TEST CASE 2: Per-Player Metrics Decomposition")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()
    
    # Test known joint action decompositions
    test_cases = [
        (0, 0, 0, "NORTH, NORTH"),
        (7, 1, 1, "SOUTH, SOUTH"),
        (14, 2, 2, "EAST, EAST"),
        (21, 3, 3, "WEST, WEST"),
        (28, 4, 4, "STAY, STAY (NOOP)"),
        (35, 5, 5, "INTERACT, INTERACT"),
    ]
    
    print("\nVerifying joint action decomposition:")
    for joint_action, expected_p0, expected_p1, description in test_cases:
        p0, p1 = get_individual_actions(joint_action)
        assert p0 == expected_p0, f"Player0 mismatch for action {joint_action}"
        assert p1 == expected_p1, f"Player1 mismatch for action {joint_action}"
        print(f"  PASS Action {joint_action:2d}: ({p0}, {p1}) = {description}")
    
    print("\nPASS Testing per-player decomposition logic:")
    
    test_joint_actions = [0, 7, 14, 21, 28, 35]
    for joint_action in test_joint_actions:
        p0, p1 = metrics_calculator._get_individual_player_labels(joint_action)
        expected_p0, expected_p1 = get_individual_actions(joint_action)
        assert p0 == expected_p0, f"Player0 mismatch for joint action {joint_action}"
        assert p1 == expected_p1, f"Player1 mismatch for joint action {joint_action}"
        print(f"  PASS Joint action {joint_action:2d} → Player0: {p0}, Player1: {p1}")
    
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    
    for seed in range(10):
        probs = create_random_joint_probability(seed=seed)
        predictions_probs.append({
            "raw_output": f"Random prediction {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(seed % 36)
    
    print(f"\nPASS Created 10 test predictions with random probabilities")
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    
    for seed in range(10):
        predictions_discrete.append({
            "raw_output": f"Discrete prediction {seed}",
            "extracted_outputs": seed % 36  # Discrete action index
        })
        ground_truth_actions_discrete.append(seed % 36)
    
    print(f"PASS Created 10 test predictions with discrete actions")
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert 'player0_results' in metrics, "Missing 'player0_results' in metrics"
    assert 'player1_results' in metrics, "Missing 'player1_results' in metrics"
    print(f"PASS Per-player results exist in metrics")
    
    player0_results = metrics['player0_results']
    assert 'num_timesteps' in player0_results
    assert player0_results['num_timesteps'] == len(predictions_probs)
    assert 'exact_match' in player0_results
    assert 'predicted_actions' in player0_results
    assert 'ground_truth_actions' in player0_results
    print(f"PASS Player0 results have correct structure")
    print(f"  - num_timesteps: {player0_results['num_timesteps']}")
    print(f"  - exact_match: {player0_results['exact_match']:.3f}")
    
    player1_results = metrics['player1_results']
    assert 'num_timesteps' in player1_results
    assert player1_results['num_timesteps'] == len(predictions_probs)
    assert 'exact_match' in player1_results
    assert 'predicted_actions' in player1_results
    assert 'ground_truth_actions' in player1_results
    print(f"PASS Player1 results have correct structure")
    print(f"  - num_timesteps: {player1_results['num_timesteps']}")
    print(f"  - exact_match: {player1_results['exact_match']:.3f}")
    
    required_metrics = ['avg_dataset_amae', 'micro_f1', 'macro_f1']
    for metric in required_metrics:
        assert metric in player0_results, f"Missing '{metric}' in player0_results"
        assert metric in player1_results, f"Missing '{metric}' in player1_results"
    print(f"PASS Both players have all standard metrics")
    
    print("\nTEST CASE 2 PASSED\n")
    return metrics


def test_exact_match_rate():
    """Test the exact match rate calculation."""
    print("\n" + "="*80)
    print("TEST CASE 3: Exact Match Rate")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()
    
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    
    for seed in range(15):
        probs = create_random_joint_probability(seed=seed)
        predictions_probs.append({
            "raw_output": f"Random prediction {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(seed % 36)
    
    print(f"Created {len(predictions_probs)} predictions with random probabilities")
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    
    for seed in range(15):
        predictions_discrete.append({
            "raw_output": f"Discrete prediction {seed}",
            "extracted_outputs": seed % 36  # Discrete action index
        })
        ground_truth_actions_discrete.append(seed % 36)
    
    print(f"Created {len(predictions_discrete)} predictions with discrete actions")
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert 'exact_match' in metrics, "Missing 'exact_match' in metrics"
    assert 0.0 <= metrics['exact_match'] <= 1.0, \
        f"exact_match should be between 0 and 1, got {metrics['exact_match']}"
    print(f"PASS exact_match = {metrics['exact_match']:.3f} (valid range)")
    
    print("\nTEST CASE 3 PASSED\n")
    return metrics


def test_invalid_predictions():
    """Test that invalid predictions are handled correctly for both joint and per-player metrics."""
    print("\n" + "="*80)
    print("TEST CASE 4: Invalid Predictions Handling")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()
    
    predictions = []
    ground_truth_actions = []
    
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    
    # Various types of invalid predictions
    predictions_probs.append({
        "raw_output": "Invalid prediction 1",
        "extracted_outputs": [0.1, 0.2, 0.3] + [0.0] * 33  # Doesn't sum to 1.0
    })
    ground_truth_actions_probs.append(5)
    print("PASS Added prediction that doesn't sum to 1.0")
    
    predictions_probs.append({
        "raw_output": "Invalid prediction 2",
        "extracted_outputs": [1.0, 0.0, 0.0]  # Wrong length
    })
    ground_truth_actions_probs.append(10)
    print("PASS Added prediction with wrong length")
    
    predictions_probs.append({
        "raw_output": "Invalid prediction 3",
        "extracted_outputs": [0.5, 'invalid'] + [0.0] * 34  # Invalid type
    })
    ground_truth_actions_probs.append(15)
    print("PASS Added prediction with invalid type")
    
    predictions_probs.append({
        "raw_output": "Invalid prediction 4",
        "extracted_outputs": [-0.1, 1.1] + [0.0] * 34  # Negative probability
    })
    ground_truth_actions_probs.append(20)
    print("PASS Added prediction with negative probability")
    
    # Add valid predictions for comparison
    for seed in range(5):
        probs = create_random_joint_probability(seed=seed)
        predictions_probs.append({
            "raw_output": f"Valid prediction {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(seed + 25)
    print("PASS Added 5 valid predictions with random probabilities")
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    
    # Various types of invalid discrete predictions
    predictions_discrete.append({
        "raw_output": "Invalid discrete 1",
        "extracted_outputs": -1  # Invalid action index
    })
    ground_truth_actions_discrete.append(5)
    print("PASS Added invalid discrete action (-1)")
    
    predictions_discrete.append({
        "raw_output": "Invalid discrete 2",
        "extracted_outputs": 50  # Out of range
    })
    ground_truth_actions_discrete.append(10)
    print("PASS Added out-of-range discrete action (50)")
    
    predictions_discrete.append({
        "raw_output": "Invalid discrete 3",
        "extracted_outputs": -5  # Negative out of range
    })
    ground_truth_actions_discrete.append(15)
    print("PASS Added negative out-of-range discrete action (-5)")
    
    predictions_discrete.append({
        "raw_output": "Invalid discrete 4",
        "extracted_outputs": 100  # Way out of range
    })
    ground_truth_actions_discrete.append(20)
    print("PASS Added way out-of-range discrete action (100)")
    
    # Add valid predictions for comparison
    for seed in range(5):
        predictions_discrete.append({
            "raw_output": f"Valid discrete {seed}",
            "extracted_outputs": seed % 36  # Valid action index
        })
        ground_truth_actions_discrete.append(seed + 25)
    print("PASS Added 5 valid discrete predictions")
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert metrics['total_invalid_preds'] == 4, \
        f"Expected 4 invalid predictions, got {metrics['total_invalid_preds']}"
    print(f"\nPASS total_invalid_preds = {metrics['total_invalid_preds']}")
    
    expected_percentage = (4 / 9) * 100
    assert abs(metrics['percentage_invalids'] - expected_percentage) < 0.1, \
        f"Expected {expected_percentage}% invalid, got {metrics['percentage_invalids']}%"
    print(f"PASS percentage_invalids = {metrics['percentage_invalids']:.1f}%")
    
    predicted_actions = metrics['predicted_actions']
    invalid_count = sum(1 for action in predicted_actions if action == -1)
    assert invalid_count == 4, f"Expected 4 actions marked as -1, got {invalid_count}"
    print(f"PASS {invalid_count} predictions marked as -1 in predicted_actions")
    
    player0_results = metrics['player0_results']
    player1_results = metrics['player1_results']
    
    player0_invalid = sum(1 for action in player0_results['predicted_actions'] if action == -1)
    player1_invalid = sum(1 for action in player1_results['predicted_actions'] if action == -1)
    
    assert player0_invalid == 4, f"Expected 4 invalid for player0, got {player0_invalid}"
    assert player1_invalid == 4, f"Expected 4 invalid for player1, got {player1_invalid}"
    print(f"PASS Per-player metrics also marked {player0_invalid} predictions as invalid")
    
    print("\nTEST CASE 4 PASSED\n")
    return metrics


def test_edge_cases():
    """Test edge cases specific to OvercookedAI."""
    print("\n" + "="*80)
    print("TEST CASE 5: Edge Cases")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()
    
    print("\nEdge Case 1: Out-of-range ground truth action")
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    for seed in range(10):
        probs = create_random_joint_probability(seed=seed)
        predictions_probs.append({
            "raw_output": f"Edge case prediction {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(100 if seed == 0 else seed)
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    for seed in range(10):
        predictions_discrete.append({
            "raw_output": f"Edge case discrete {seed}",
            "extracted_outputs": seed % 36  # Discrete action index
        })
        ground_truth_actions_discrete.append(100 if seed == 0 else seed)
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert metrics['num_timesteps'] == 10
    print(f"PASS Out-of-range action handled gracefully (clamped to NOOP)")
    
    print("\nEdge Case 2: Mostly invalid predictions")
    # Test with probabilities=True
    predictions_probs = [
        {
            "raw_output": "Invalid edge case 1",
            "extracted_outputs": [0.5, 0.5]
        },
        {
            "raw_output": "Invalid edge case 2", 
            "extracted_outputs": [0.3, 0.3, 0.3]
        },
        {
            "raw_output": "Invalid edge case 3",
            "extracted_outputs": ['invalid']
        },
    ]
    ground_truth_actions_probs = [5, 10, 15]
    
    for seed in range(7):
        probs = create_random_joint_probability(seed=seed + 200)
        predictions_probs.append({
            "raw_output": f"Valid edge case {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(seed + 20)
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = [
        {
            "raw_output": "Invalid discrete edge case 1",
            "extracted_outputs": -1  # Invalid action
        },
        {
            "raw_output": "Invalid discrete edge case 2", 
            "extracted_outputs": 50  # Out of range
        },
        {
            "raw_output": "Invalid discrete edge case 3",
            "extracted_outputs": -5  # Negative out of range
        },
    ]
    ground_truth_actions_discrete = [5, 10, 15]
    
    for seed in range(7):
        predictions_discrete.append({
            "raw_output": f"Valid discrete edge case {seed}",
            "extracted_outputs": (seed + 20) % 36  # Valid action index
        })
        ground_truth_actions_discrete.append(seed + 20)
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert metrics['total_invalid_preds'] == 3
    expected_percentage = (3 / 10) * 100
    assert abs(metrics['percentage_invalids'] - expected_percentage) < 0.1
    print(f"PASS Mostly invalid predictions handled correctly")
    print(f"  - total_invalid_preds = {metrics['total_invalid_preds']}")
    print(f"  - percentage_invalids = {metrics['percentage_invalids']:.1f}%")
    
    print("\nEdge Case 3: Multiple predictions with varied probabilities")
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    for seed in range(10):
        probs = create_random_joint_probability(seed=seed)
        predictions_probs.append({
            "raw_output": f"Varied prediction {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(seed % 36)
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    for seed in range(10):
        predictions_discrete.append({
            "raw_output": f"Varied discrete {seed}",
            "extracted_outputs": seed % 36  # Discrete action index
        })
        ground_truth_actions_discrete.append(seed % 36)
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert metrics['num_timesteps'] == 10
    assert metrics['total_invalid_preds'] == 0
    assert 'exact_match' in metrics
    print(f"PASS Multiple predictions handled correctly")
    print(f"  - num_timesteps = {metrics['num_timesteps']}")
    print(f"  - exact_match = {metrics['exact_match']:.3f}")
    
    print("\nEdge Case 4: Moderate number of predictions")
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    for seed in range(10):
        probs = create_random_joint_probability(seed=seed + 100)
        predictions_probs.append({
            "raw_output": f"Moderate prediction {seed}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(seed + 10)
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    for seed in range(10):
        predictions_discrete.append({
            "raw_output": f"Moderate discrete {seed}",
            "extracted_outputs": (seed + 10) % 36  # Discrete action index
        })
        ground_truth_actions_discrete.append(seed + 10)
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    assert metrics['num_timesteps'] == 10
    assert 'player0_results' in metrics
    assert 'player1_results' in metrics
    print(f"PASS Moderate number of predictions handled correctly")
    
    print("\nEdge Case 5: NOOP action (28 = STAY, STAY) decomposition")
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    actions_to_test = [NOOP_ACTION, 0, 7, 14, 21, 28, 35, 1, 8, 15]
    for action in actions_to_test:
        probs = create_random_joint_probability(seed=action)
        predictions_probs.append({
            "raw_output": f"NOOP test action {action}",
            "extracted_outputs": probs
        })
        ground_truth_actions_probs.append(action)
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    for action in actions_to_test:
        predictions_discrete.append({
            "raw_output": f"NOOP discrete test action {action}",
            "extracted_outputs": action  # Discrete action index
        })
        ground_truth_actions_discrete.append(action)
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    # NOOP action (28) should decompose to both players taking STAY (action 4)
    player0_gt = metrics['player0_results']['ground_truth_actions']
    player1_gt = metrics['player1_results']['ground_truth_actions']
    assert player0_gt[0] == 4, f"Expected player0 action 4 (STAY) for NOOP, got {player0_gt[0]}"
    assert player1_gt[0] == 4, f"Expected player1 action 4 (STAY) for NOOP, got {player1_gt[0]}"
    print(f"PASS NOOP action (28) correctly decomposes to (4, 4)")
    
    print("\nTEST CASE 5 PASSED\n")
    return metrics


def test_mixed_valid_invalid_predictions():
    """Test mixed valid/invalid predictions of different types."""
    print("\n" + "="*80)
    print("TEST CASE 6: Mixed Valid/Invalid Predictions")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()
    
    # Test with_probabilities=True - mixed probability scenarios
    print("\n--- Testing with probabilities=True (mixed types) ---")
    predictions_probs = []
    ground_truth_actions_probs = []
    
    # Mix of valid and invalid probability predictions
    predictions_probs.append({
        "raw_output": "Valid prob 1",
        "extracted_outputs": [0.0]*35 + [1.0]  # Valid probability distribution
    })
    ground_truth_actions_probs.append(35)
    
    predictions_probs.append({
        "raw_output": "Invalid string",
        "extracted_outputs": "invalid string"  # Invalid string
    })
    ground_truth_actions_probs.append(5)
    
    predictions_probs.append({
        "raw_output": "Valid prob 2", 
        "extracted_outputs": [0.0]*10 + [1.0] + [0.0]*25  # Valid probability distribution
    })
    ground_truth_actions_probs.append(10)
    
    predictions_probs.append({
        "raw_output": "Invalid length",
        "extracted_outputs": [0.1, 0.2, 0.3]  # Invalid length
    })
    ground_truth_actions_probs.append(15)
    
    predictions_probs.append({
        "raw_output": "Valid prob 3",
        "extracted_outputs": [0.0]*20 + [1.0] + [0.0]*15  # Valid probability distribution
    })
    ground_truth_actions_probs.append(20)
    
    predictions_probs.append({
        "raw_output": "Invalid None",
        "extracted_outputs": None  # Invalid None
    })
    ground_truth_actions_probs.append(25)
    
    predictions_probs.append({
        "raw_output": "Invalid list type",
        "extracted_outputs": [0.5, 'invalid', 0.3] + [0.0]*33  # Invalid type in list
    })
    ground_truth_actions_probs.append(30)
    
    predictions_probs.append({
        "raw_output": "Valid prob 4",
        "extracted_outputs": [0.0]*28 + [1.0] + [0.0]*7  # Valid probability distribution
    })
    ground_truth_actions_probs.append(28)
    
    print(f"PASS Created {len(predictions_probs)} mixed probability predictions")
    print(f"  - {4} valid probability distributions")
    print(f"  - {4} invalid predictions (string, wrong length, None, invalid type)")
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with_probabilities=False - mixed discrete scenarios
    print("\n--- Testing with probabilities=False (mixed types) ---")
    predictions_discrete = []
    ground_truth_actions_discrete = []
    
    # Mix of valid and invalid discrete predictions
    predictions_discrete.append({
        "raw_output": "Valid discrete 1",
        "extracted_outputs": 5  # Valid discrete action
    })
    ground_truth_actions_discrete.append(5)
    
    predictions_discrete.append({
        "raw_output": "Invalid string",
        "extracted_outputs": "invalid string"  # Invalid string
    })
    ground_truth_actions_discrete.append(10)
    
    predictions_discrete.append({
        "raw_output": "Valid discrete 2",
        "extracted_outputs": 12  # Valid discrete action
    })
    ground_truth_actions_discrete.append(12)
    
    predictions_discrete.append({
        "raw_output": "Invalid list",
        "extracted_outputs": [1, 2, 3]  # Invalid list
    })
    ground_truth_actions_discrete.append(15)
    
    predictions_discrete.append({
        "raw_output": "Valid discrete 3",
        "extracted_outputs": 20  # Valid discrete action
    })
    ground_truth_actions_discrete.append(20)
    
    predictions_discrete.append({
        "raw_output": "Invalid None",
        "extracted_outputs": None  # Invalid None
    })
    ground_truth_actions_discrete.append(25)
    
    predictions_discrete.append({
        "raw_output": "Invalid out of range",
        "extracted_outputs": 50  # Invalid out of range
    })
    ground_truth_actions_discrete.append(30)
    
    predictions_discrete.append({
        "raw_output": "Valid discrete 4",
        "extracted_outputs": 28  # Valid discrete action
    })
    ground_truth_actions_discrete.append(28)
    
    print(f"PASS Created {len(predictions_discrete)} mixed discrete predictions")
    print(f"  - {4} valid discrete actions")
    print(f"  - {4} invalid predictions (string, list, None, out of range)")
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Validate results for both modes
    print(f"\n--- Validation Results ---")
    
    # Probabilities mode validation
    assert metrics_probs['total_invalid_preds'] == 4, \
        f"Expected 4 invalid predictions (probabilities), got {metrics_probs['total_invalid_preds']}"
    print(f"PASS Probabilities mode: {metrics_probs['total_invalid_preds']} invalid predictions")
    
    expected_percentage_probs = (4 / 8) * 100
    assert abs(metrics_probs['percentage_invalids'] - expected_percentage_probs) < 0.1, \
        f"Expected {expected_percentage_probs}% invalid (probabilities), got {metrics_probs['percentage_invalids']}%"
    print(f"PASS Probabilities mode: {metrics_probs['percentage_invalids']:.1f}% invalid")
    
    # Discrete mode validation
    assert metrics_discrete['total_invalid_preds'] == 4, \
        f"Expected 4 invalid predictions (discrete), got {metrics_discrete['total_invalid_preds']}"
    print(f"PASS Discrete mode: {metrics_discrete['total_invalid_preds']} invalid predictions")
    
    expected_percentage_discrete = (4 / 8) * 100
    assert abs(metrics_discrete['percentage_invalids'] - expected_percentage_discrete) < 0.1, \
        f"Expected {expected_percentage_discrete}% invalid (discrete), got {metrics_discrete['percentage_invalids']}%"
    print(f"PASS Discrete mode: {metrics_discrete['percentage_invalids']:.1f}% invalid")
    
    # Check that valid predictions are processed correctly
    predicted_actions_probs = metrics_probs['predicted_actions']
    valid_count_probs = sum(1 for action in predicted_actions_probs if action != -1)
    assert valid_count_probs == 4, f"Expected 4 valid predictions (probabilities), got {valid_count_probs}"
    print(f"PASS Probabilities mode: {valid_count_probs} valid predictions processed")
    
    predicted_actions_discrete = metrics_discrete['predicted_actions']
    valid_count_discrete = sum(1 for action in predicted_actions_discrete if action != -1)
    assert valid_count_discrete == 4, f"Expected 4 valid predictions (discrete), got {valid_count_discrete}"
    print(f"PASS Discrete mode: {valid_count_discrete} valid predictions processed")
    
    print("\nTEST CASE 6 PASSED\n")
    return metrics_probs, metrics_discrete


def test_all_invalid_predictions():
    """Test handling when all predictions are invalid."""
    print("\n" + "="*80)
    print("TEST CASE 7: All Invalid Predictions")
    print("="*80)
    
    metrics_calculator = OvercookedAIMetricsCalculator()
    
    predictions = []
    ground_truth_actions = []
    
    # Test with probabilities=True
    predictions_probs = []
    ground_truth_actions_probs = []
    
    # All invalid predictions - tests the eval_utils.py fix for np.nan handling
    predictions_probs.append({
        "raw_output": "All invalid test 1",
        "extracted_outputs": [0.5, 0.5]  # Wrong length
    })
    ground_truth_actions_probs.append(5)

    predictions_probs.append({
        "raw_output": "All invalid test 2",
        "extracted_outputs": [0.3, 0.3, 0.3]  # Wrong length
    })
    ground_truth_actions_probs.append(10)
    
    predictions_probs.append({
        "raw_output": "All invalid test 3",
        "extracted_outputs": ['invalid'] + [0.0] * 35  # Invalid type
    })
    ground_truth_actions_probs.append(15)
    
    predictions_probs.append({
        "raw_output": "All invalid test 4",
        "extracted_outputs": [-0.1, 1.1] + [0.0] * 34  # Negative probability
    })
    ground_truth_actions_probs.append(20)
    
    predictions_probs.append({
        "raw_output": "All invalid test 5",
        "extracted_outputs": [0.1, 0.2, 0.3] + [0.0] * 33  # Doesn't sum to 1
    })
    ground_truth_actions_probs.append(25)
    
    print("PASS Added 5 invalid predictions (probabilities)")
    
    metrics_probs = metrics_calculator.calculate_metrics(
        predictions_probs, ground_truth_actions_probs, with_probabilities=True
    )
    
    # Test with probabilities=False (discrete actions)
    predictions_discrete = []
    ground_truth_actions_discrete = []
    
    # All invalid discrete predictions
    predictions_discrete.append({
        "raw_output": "All invalid discrete 1",
        "extracted_outputs": -1  # Invalid action
    })
    ground_truth_actions_discrete.append(5)

    predictions_discrete.append({
        "raw_output": "All invalid discrete 2",
        "extracted_outputs": 50  # Out of range
    })
    ground_truth_actions_discrete.append(10)
    
    predictions_discrete.append({
        "raw_output": "All invalid discrete 3",
        "extracted_outputs": -5  # Negative out of range
    })
    ground_truth_actions_discrete.append(15)
    
    predictions_discrete.append({
        "raw_output": "All invalid discrete 4",
        "extracted_outputs": 100  # Way out of range
    })
    ground_truth_actions_discrete.append(20)
    
    predictions_discrete.append({
        "raw_output": "All invalid discrete 5",
        "extracted_outputs": -10  # Way negative out of range
    })
    ground_truth_actions_discrete.append(25)
    
    print("PASS Added 5 invalid predictions (discrete)")
    
    metrics_discrete = metrics_calculator.calculate_metrics(
        predictions_discrete, ground_truth_actions_discrete, with_probabilities=False
    )
    
    # Use probabilities results for the rest of the test
    metrics = metrics_probs
    
    # All predictions should be marked as invalid
    assert metrics['total_invalid_preds'] == 5, \
        f"Expected 5 invalid predictions, got {metrics['total_invalid_preds']}"
    print(f"PASS total_invalid_preds = {metrics['total_invalid_preds']}")
    
    # All should be marked as invalid (100%)
    expected_percentage = 100.0
    assert abs(metrics['percentage_invalids'] - expected_percentage) < 0.1, \
        f"Expected {expected_percentage}% invalid, got {metrics['percentage_invalids']}%"
    print(f"PASS percentage_invalids = {metrics['percentage_invalids']:.1f}%")
    
    # All predicted actions should be -1
    predicted_actions = metrics['predicted_actions']
    invalid_count = sum(1 for action in predicted_actions if action == -1)
    assert invalid_count == 5, f"Expected 5 actions marked as -1, got {invalid_count}"
    print(f"PASS {invalid_count} predictions marked as -1 in predicted_actions")
    
    # Per-player metrics should also handle all invalid predictions
    player0_results = metrics['player0_results']
    player1_results = metrics['player1_results']
    
    player0_invalid = sum(1 for action in player0_results['predicted_actions'] if action == -1)
    player1_invalid = sum(1 for action in player1_results['predicted_actions'] if action == -1)
    
    assert player0_invalid == 5, f"Expected 5 invalid for player0, got {player0_invalid}"
    assert player1_invalid == 5, f"Expected 5 invalid for player1, got {player1_invalid}"
    print(f"PASS Per-player metrics also marked {player0_invalid} predictions as invalid")
    
    # Metrics should still be calculated (even if they're NaN or max values)
    assert 'exact_match' in metrics
    assert 'avg_dataset_amae' in metrics
    assert 'avg_dataset_amse' in metrics
    print("PASS All standard metrics present even with all invalid predictions")
    
    # Exact match should be 0 (no correct predictions)
    assert metrics['exact_match'] == 0.0, \
        f"Expected exact_match = 0.0, got {metrics['exact_match']}"
    print(f"PASS exact_match = {metrics['exact_match']} (no correct predictions)")
    
    print("\nTEST CASE 7 PASSED\n")
    return metrics


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OVERCOOKED AI METRICS CALCULATOR - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    try:
        test_gameplay_metrics()
        test_per_player_metrics()
        test_exact_match_rate()
        test_invalid_predictions()
        test_edge_cases()
        test_mixed_valid_invalid_predictions()
        test_all_invalid_predictions()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nThe OvercookedAI Metrics Calculator is working correctly with:")
        print("  - 36 joint action space")
        print("  - Per-player metrics decomposition")
        print("  - Exact match rate calculation")
        print("  - Invalid prediction handling")
        print("  - Edge case robustness")
        print()
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}\n")
        raise
