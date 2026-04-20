#!/usr/bin/env python3
"""
Example usage of the MultiGrid environment.

This script demonstrates the basic functionality of the MultiGrid system.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multigrid.env import MultiGridEnv
from multigrid.agent import Action


def basic_example():
    """Basic example: Create environment and execute actions."""
    print("=" * 60)
    print("BASIC EXAMPLE: Square Grid Navigation")
    print("=" * 60)

    # Create a simple task
    task_spec = {
        "task_id": "example_001",
        "seed": 42,
        "scene": {
            "bounds": {"width": 1.0, "height": 1.0},
            "objects": [
                {
                    "id": "cube_red",
                    "type": "movable",
                    "color": "red",
                    "position": {"x": 0.7, "y": 0.7},
                    "size": 0.1
                }
            ],
            "agent": {
                "position": {"x": 0.2, "y": 0.2},
                "facing": 0  # Facing north
            }
        },
        "goal": {
            "predicate": "object_in_zone",
            "object_id": "cube_red",
            "zone_id": "zone_blue"
        },
        "limits": {"max_steps": 100},
        "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}}
    }

    # Create environment
    env = MultiGridEnv(task_spec, tiling="square")
    obs, info = env.reset(seed=42)

    print(f"\nInitial state:")
    state = env.get_state_dict()
    print(f"  Agent position: {state['agent']['cell_id']}")
    print(f"  Agent facing: {state['agent']['facing_direction']}")
    print(f"  Agent holding: {state['agent']['holding']}")

    # Execute some actions
    actions = [
        (Action.FORWARD, "Move forward"),
        (Action.TURN_RIGHT, "Turn right"),
        (Action.FORWARD, "Move forward"),
        (Action.FORWARD, "Move forward"),
    ]

    print(f"\nExecuting {len(actions)} actions:")
    for action, description in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        state = env.get_state_dict()

        print(f"\n  Action: {description}")
        print(f"    New position: {state['agent']['cell_id']}")
        print(f"    Facing: {state['agent']['facing_direction']}")
        print(f"    Reward: {reward:.2f}")
        if info.get('invalid_action'):
            print(f"    ⚠️  Invalid action!")


def multi_tiling_example():
    """Demonstrate the same task on different tilings."""
    print("\n" + "=" * 60)
    print("MULTI-TILING EXAMPLE: Same Task, Different Grids")
    print("=" * 60)

    task_spec = {
        "task_id": "example_002",
        "seed": 42,
        "scene": {
            "bounds": {"width": 1.0, "height": 1.0},
            "objects": [],
            "agent": {
                "position": {"x": 0.5, "y": 0.5},
                "facing": 0
            }
        },
        "goal": {},
        "limits": {"max_steps": 100},
        "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}}
    }

    for tiling_name in ["square", "hex", "triangle"]:
        print(f"\n{tiling_name.upper()} TILING:")

        env = MultiGridEnv(task_spec, tiling=tiling_name)
        obs, info = env.reset()

        tiling = env.tiling
        print(f"  Directions: {tiling.directions}")
        print(f"  Direction count: {len(tiling.directions)}")
        print(f"  Total cells: {len(tiling.cells)}")

        # Check a cell's neighbors
        first_cell_id = list(tiling.cells.keys())[50]  # Pick a middle cell
        cell = tiling.cells[first_cell_id]
        print(f"  Sample cell {first_cell_id} has {len(cell.neighbors)} neighbors")


def object_interaction_example():
    """Demonstrate object interaction (pickup, drop, push)."""
    print("\n" + "=" * 60)
    print("OBJECT INTERACTION EXAMPLE")
    print("=" * 60)

    task_spec = {
        "task_id": "example_003",
        "seed": 42,
        "scene": {
            "bounds": {"width": 1.0, "height": 1.0},
            "objects": [
                {
                    "id": "cube_red",
                    "type": "movable",
                    "color": "red",
                    "position": {"x": 0.4, "y": 0.2},
                    "size": 0.1
                }
            ],
            "agent": {
                "position": {"x": 0.2, "y": 0.2},
                "facing": 1  # Facing east
            }
        },
        "goal": {},
        "limits": {"max_steps": 100},
        "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}}
    }

    env = MultiGridEnv(task_spec, tiling="square")
    obs, info = env.reset()

    print(f"\nInitial state:")
    state = env.get_state_dict()
    print(f"  Agent: {state['agent']['cell_id']} (facing {state['agent']['facing_direction']})")
    print(f"  Red cube: {state['objects']['cube_red']['cell_id']}")
    print(f"  Holding: {state['agent']['holding']}")

    # Move to object and pick it up
    print(f"\n1. Moving forward to object...")
    obs, reward, _, _, info = env.step(Action.FORWARD)
    state = env.get_state_dict()
    print(f"  Agent: {state['agent']['cell_id']}")

    print(f"\n2. Picking up object...")
    obs, reward, _, _, info = env.step(Action.PICKUP)
    state = env.get_state_dict()
    print(f"  Holding: {state['agent']['holding']}")
    if state['agent']['holding']:
        print(f"  ✓ Successfully picked up {state['agent']['holding']}!")

    print(f"\n3. Moving with object...")
    obs, reward, _, _, info = env.step(Action.FORWARD)
    state = env.get_state_dict()
    print(f"  Agent: {state['agent']['cell_id']} (still holding {state['agent']['holding']})")

    print(f"\n4. Dropping object...")
    obs, reward, _, _, info = env.step(Action.DROP)
    state = env.get_state_dict()
    print(f"  Holding: {state['agent']['holding']}")
    print(f"  ✓ Object dropped at agent's location!")


def distance_calculation_example():
    """Demonstrate distance calculations on different tilings."""
    print("\n" + "=" * 60)
    print("DISTANCE CALCULATION EXAMPLE")
    print("=" * 60)

    for tiling_name in ["square", "hex", "triangle"]:
        from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling

        tiling_class = {
            "square": SquareTiling,
            "hex": HexTiling,
            "triangle": TriangleTiling
        }[tiling_name]

        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        # Calculate distance between two cells
        cell_ids = list(tiling.cells.keys())
        cell_a = cell_ids[10]
        cell_b = cell_ids[50]

        distance = tiling.distance(cell_a, cell_b)

        print(f"\n{tiling_name.upper()} TILING:")
        print(f"  Distance from {cell_a} to {cell_b}: {distance} hops")

        # Get coordinates
        pos_a = tiling.cell_to_canonical(cell_a)
        pos_b = tiling.cell_to_canonical(cell_b)
        print(f"  Canonical positions: {pos_a} -> {pos_b}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# MultiGrid v1.1 - Usage Examples")
    print("#" * 60)

    basic_example()
    multi_tiling_example()
    object_interaction_example()
    distance_calculation_example()

    print("\n" + "#" * 60)
    print("# All examples completed successfully!")
    print("#" * 60)
    print("\nTo run tests: python -m pytest tests/ -v")
    print("To visualize: python visualize_grid.py")


if __name__ == "__main__":
    main()
