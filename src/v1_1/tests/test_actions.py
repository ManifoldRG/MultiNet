# test_actions.py

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.env import MultiGridEnv, Action


class TestActions:
    """Tests for action execution."""

    @pytest.fixture
    def simple_task(self):
        """Simple task spec for testing."""
        return {
            "task_id": "test_001",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "cube_red",
                        "type": "movable",
                        "color": "red",
                        "position": {"x": 0.5, "y": 0.5},
                        "size": 0.1
                    }
                ],
                "agent": {
                    "position": {"x": 0.2, "y": 0.2},
                    "facing": 0
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

    def test_forward_movement(self, simple_task):
        """Agent moves forward in facing direction."""
        env = MultiGridEnv(simple_task, tiling="square")
        obs, info = env.reset(seed=42)

        initial_cell = env.state.agent.cell_id
        initial_facing = env.state.agent.facing

        obs, reward, term, trunc, info = env.step(Action.FORWARD)

        # Agent should have moved
        assert env.state.agent.cell_id != initial_cell or info.get("invalid_action")

    def test_turn_changes_facing(self, simple_task):
        """Turn actions change facing without moving."""
        env = MultiGridEnv(simple_task, tiling="square")
        env.reset(seed=42)

        initial_cell = env.state.agent.cell_id
        initial_facing = env.state.agent.facing

        env.step(Action.TURN_RIGHT)

        assert env.state.agent.cell_id == initial_cell  # Didn't move
        assert env.state.agent.facing == (initial_facing + 1) % 4  # Facing changed

    def test_invalid_move_into_wall(self, simple_task):
        """Moving into boundary returns invalid_action."""
        # Modify task to put agent at corner facing wall
        simple_task["scene"]["agent"]["position"] = {"x": 0.05, "y": 0.05}
        simple_task["scene"]["agent"]["facing"] = 0  # Facing north (into wall)

        env = MultiGridEnv(simple_task, tiling="square")
        env.reset(seed=42)

        obs, reward, term, trunc, info = env.step(Action.FORWARD)

        assert info.get("invalid_action") == True

    def test_pickup_object(self, simple_task):
        """Agent can pick up adjacent objects."""
        # Position agent next to object
        simple_task["scene"]["agent"]["position"] = {"x": 0.4, "y": 0.5}
        simple_task["scene"]["agent"]["facing"] = 1  # Facing east (toward object)

        env = MultiGridEnv(simple_task, tiling="square")
        env.reset(seed=42)

        assert env.state.agent.holding is None

        # Move forward to object's cell
        env.step(Action.FORWARD)

        # Pick up
        env.step(Action.PICKUP)

        assert env.state.agent.holding is not None
        assert env.state.agent.holding.id == "cube_red"
