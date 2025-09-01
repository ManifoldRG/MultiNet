import sys
import os
import random
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.eval_harness.scoring.gameplay_metrics import OvercookedAIMetricsCalculator
import src.eval_harness.examples.gameplay_adapter_example as gameplay_model_adapter


def test_gameplay_metrics():
    num_actions = 6
    use_probabilities = True

    # Create a gameplay metrics calculator
    metrics_calculator = OvercookedAIMetricsCalculator()

    # Create a gameplay model adapter
    adapter = gameplay_model_adapter.SimpleGameplayAdapter(action_space_size=num_actions)
    
    # Initialize the adapter
    adapter.initialize()

    # Create some random sample observations
    observations = [
        {
            'image': Image.new('RGB', (84, 84), color='blue'),
            'state': np.array([random.random() for _ in range(3)])
        }
        for _ in range(3)
    ]

    # Add some with instructions
    observations.append({
        'image': Image.new('RGB', (84, 84), color='blue'),
        'state': np.array([random.random() for _ in range(3)]),
        'instruction': "fulfill the order"
    })

    # Predict the action
    outputs = adapter.batch_predict_actions(observations, dataset_name="overcooked_ai", return_probabilities=use_probabilities)
    actions = [output['probabilities'] if use_probabilities else output['action'] for output in outputs]

    # Create some ground truth actions
    ground_truth_actions = [0, 1, 2, 3]

    # Add some valid predictions
    actions.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ground_truth_actions.append(0)
    actions.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    ground_truth_actions.append(1)
    actions.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    ground_truth_actions.append(2)
    actions.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    ground_truth_actions.append(3)
    actions.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    ground_truth_actions.append(4)
    actions.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ground_truth_actions.append(5)

    # Add some invalid predictions
    actions.append([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    ground_truth_actions.append(4)
    actions.append([0.1, {0.2}, 0.3, 0.4, 0.5, 0.6])
    ground_truth_actions.append(5)

    # Calculate the metrics
    metrics = metrics_calculator.calculate_metrics(actions, ground_truth_actions, with_probabilities=use_probabilities)
    print(metrics)

if __name__ == "__main__":
    test_gameplay_metrics()