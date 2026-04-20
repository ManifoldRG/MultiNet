#!/usr/bin/env python3
"""
Test script for the multigrid module.

Tests rendering, goal system, and all tiling types.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from multigrid.env import MultiGridEnv, TilingRegistry
from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling
from multigrid.goals import (
    ReachPositionGoal,
    ReachCanonicalPositionGoal,
    CollectAllGoal,
    create_goal_from_spec,
)
from multigrid.rendering import render_multigrid
from multigrid.agent import Action


def test_tiling_registry():
    """Test tiling registry returns correct types."""
    print("Testing TilingRegistry...")

    square = TilingRegistry.get("square")
    assert isinstance(square, SquareTiling), "Expected SquareTiling"

    hex_tiling = TilingRegistry.get("hex")
    assert isinstance(hex_tiling, HexTiling), "Expected HexTiling"

    triangle = TilingRegistry.get("triangle")
    assert isinstance(triangle, TriangleTiling), "Expected TriangleTiling"

    print("  ✓ TilingRegistry works correctly")


def test_square_tiling():
    """Test square tiling basic operations."""
    print("Testing SquareTiling...")

    tiling = SquareTiling()
    tiling.generate_graph(5, 5, seed=42)

    # Check cell count
    assert len(tiling.cells) == 25, f"Expected 25 cells, got {len(tiling.cells)}"

    # Check directions
    assert len(tiling.directions) == 4, "Square should have 4 directions"

    # Check neighbor connectivity
    center = "sq_2_2"
    neighbors = []
    for d in tiling.directions:
        n = tiling.get_neighbor(center, d)
        if n:
            neighbors.append(n)
    assert len(neighbors) == 4, f"Center cell should have 4 neighbors, got {len(neighbors)}"

    print("  ✓ SquareTiling works correctly")


def test_hex_tiling():
    """Test hex tiling basic operations."""
    print("Testing HexTiling...")

    tiling = HexTiling()
    tiling.generate_graph(3, 3, seed=42)

    # Check directions
    assert len(tiling.directions) == 6, "Hex should have 6 directions"

    # Check cell count (varies with grid arrangement)
    assert len(tiling.cells) > 0, "Should have some cells"

    print(f"  ✓ HexTiling works correctly ({len(tiling.cells)} cells)")


def test_triangle_tiling():
    """Test triangle tiling - this was the problematic one."""
    print("Testing TriangleTiling...")

    tiling = TriangleTiling()
    tiling.generate_graph(3, 3, seed=42)

    # Check directions
    assert len(tiling.directions) == 3, "Triangle should have 3 directions"

    # Check cell count
    assert len(tiling.cells) > 0, "Should have some cells"

    # Verify all cells have some neighbors
    for cell_id, cell in tiling.cells.items():
        neighbor_count = sum(1 for d in tiling.directions if tiling.get_neighbor(cell_id, d))
        # Triangles can have 1-3 neighbors depending on position
        assert neighbor_count >= 1, f"Cell {cell_id} has no neighbors"

    print(f"  ✓ TriangleTiling works correctly ({len(tiling.cells)} cells)")


def test_goals():
    """Test goal system."""
    print("Testing Goal System...")

    tiling = SquareTiling()
    tiling.generate_graph(5, 5, seed=42)

    # Test creating goals from spec
    goal_spec = {
        "type": "reach_position",
        "target": {"x": 0.9, "y": 0.9}
    }
    goal = create_goal_from_spec(goal_spec, tiling)
    assert goal is not None, "Goal should be created"
    assert hasattr(goal, 'check'), "Goal should have check method"

    # Test collect_all goal
    collect_spec = {
        "type": "collect_all",
        "target_ids": ["key_1", "key_2"]
    }
    collect_goal = create_goal_from_spec(collect_spec, tiling)
    assert isinstance(collect_goal, CollectAllGoal), "Should be CollectAllGoal"

    print("  ✓ Goal system works correctly")


def test_rendering():
    """Test rendering for all tiling types."""
    print("Testing Rendering...")

    for tiling_name, tiling_class in [
        ("square", SquareTiling),
        ("hex", HexTiling),
        ("triangle", TriangleTiling)
    ]:
        print(f"  Testing {tiling_name} rendering...")

        task_spec = {
            "task_id": f"test_{tiling_name}",
            "seed": 42,
            "tiling": {
                "type": tiling_name,
                "grid_size": {"width": 5, "height": 5}
            },
            "scene": {
                "agent": {
                    "position": {"x": 0.1, "y": 0.1},
                    "facing": 0
                },
                "objects": [
                    {
                        "id": "box_1",
                        "type": "movable",
                        "color": "blue",
                        "position": {"x": 0.5, "y": 0.5}
                    }
                ]
            },
            "goal": {
                "type": "reach_position",
                "target": {"x": 0.9, "y": 0.9}
            },
            "limits": {
                "max_steps": 100
            }
        }

        env = MultiGridEnv(task_spec, tiling=tiling_name, render_mode="rgb_array")
        obs, info = env.reset()

        # Check observation is valid
        assert obs.shape == (64, 64, 3), f"Expected (64,64,3), got {obs.shape}"
        assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"

        # Check it's not all black
        assert obs.sum() > 0, "Observation should not be all black"

        # Test high-res render
        frame = env.render()
        assert frame.shape == (640, 640, 3), f"Expected (640,640,3), got {frame.shape}"
        assert frame.sum() > 0, "Render should not be all black"

        print(f"    ✓ {tiling_name} renders correctly")

    print("  ✓ All rendering works correctly")


def test_env_step():
    """Test environment stepping."""
    print("Testing Environment Step...")

    task_spec = {
        "task_id": "test_step",
        "seed": 42,
        "tiling": {
            "type": "square",
            "grid_size": {"width": 5, "height": 5}
        },
        "scene": {
            "agent": {
                "position": {"x": 0.5, "y": 0.5},
                "facing": 0
            },
            "objects": []
        },
        "goal": {
            "type": "reach_position",
            "target": {"x": 0.9, "y": 0.9}
        },
        "limits": {
            "max_steps": 100
        }
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="rgb_array")
    obs, info = env.reset()

    initial_cell = env.state.agent.cell_id

    # Turn right
    obs, reward, terminated, truncated, info = env.step(Action.TURN_RIGHT.value)
    assert not terminated, "Should not terminate from turn"

    # Move forward
    obs, reward, terminated, truncated, info = env.step(Action.FORWARD.value)
    new_cell = env.state.agent.cell_id

    # Should have moved (or stayed if blocked)
    print(f"    Agent moved from {initial_cell} to {new_cell}")

    print("  ✓ Environment stepping works correctly")


def test_state_dict():
    """Test state dictionary export."""
    print("Testing State Dict Export...")

    task_spec = {
        "task_id": "test_state",
        "seed": 42,
        "tiling": {
            "type": "square",
            "grid_size": {"width": 5, "height": 5}
        },
        "scene": {
            "agent": {
                "position": {"x": 0.5, "y": 0.5},
                "facing": 0
            },
            "objects": []
        },
        "goal": {
            "type": "reach_position",
            "target": {"x": 0.9, "y": 0.9}
        },
        "limits": {
            "max_steps": 100
        }
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="state_dict")
    env.reset()

    state_dict = env.get_state_dict()

    assert "agent" in state_dict, "State should have agent"
    assert "objects" in state_dict, "State should have objects"
    assert "step" in state_dict, "State should have step"
    assert "goal_achieved" in state_dict, "State should have goal_achieved"

    print("  ✓ State dict export works correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MultiGrid Module Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_tiling_registry,
        test_square_tiling,
        test_hex_tiling,
        test_triangle_tiling,
        test_goals,
        test_rendering,
        test_env_step,
        test_state_dict,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
