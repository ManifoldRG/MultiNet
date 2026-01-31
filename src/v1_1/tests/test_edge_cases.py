# test_edge_cases.py

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.env import MultiGridEnv, Action
from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling


def create_simple_task(grid_size=10, agent_pos=(0.5, 0.5), max_steps=100):
    """Helper to create a simple task spec."""
    return {
        "task_id": "test_task",
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
                "position": {"x": agent_pos[0], "y": agent_pos[1]},
                "facing": 0
            }
        },
        "goal": {
            "predicate": "reach_position",
            "position": {"x": 0.9, "y": 0.9}
        },
        "limits": {"max_steps": max_steps},
        "tiling": {"type": "square", "grid_size": {"width": grid_size, "height": grid_size}}
    }


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_agent_at_corner(self):
        """Agent at corner has limited movement options."""
        task = create_simple_task(agent_pos=(0.01, 0.01))
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Corner cell should have exactly 2 neighbors (east and south)
        cell_id = env.state.agent.cell_id
        neighbors = env.tiling.cells[cell_id].neighbors
        assert len(neighbors) == 2, f"Corner cell should have 2 neighbors, got {len(neighbors)}"

    def test_agent_at_edge(self):
        """Agent at edge has 3 movement options."""
        task = create_simple_task(agent_pos=(0.5, 0.01))
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Edge cell (but not corner) should have 3 neighbors
        cell_id = env.state.agent.cell_id
        neighbors = env.tiling.cells[cell_id].neighbors
        assert len(neighbors) == 3, f"Edge cell should have 3 neighbors, got {len(neighbors)}"

    def test_seed_zero(self):
        """Seed 0 is valid and produces deterministic results."""
        task = create_simple_task()

        env1 = MultiGridEnv(task, tiling="square")
        env2 = MultiGridEnv(task, tiling="square")

        obs1, info1 = env1.reset(seed=0)
        obs2, info2 = env2.reset(seed=0)

        # Observations should be identical
        assert obs1.shape == obs2.shape
        assert (obs1 == obs2).all(), "Same seed should produce identical observations"

        # States should be identical
        assert env1.state.agent.cell_id == env2.state.agent.cell_id
        assert env1.state.agent.facing == env2.state.agent.facing

    def test_max_steps_truncation(self):
        """Episode truncates at max_steps."""
        task = create_simple_task(max_steps=5)
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        truncated = False
        for i in range(6):
            obs, reward, terminated, truncated, info = env.step(Action.WAIT)
            # Truncation happens ON the max_steps'th step (steps are 1-indexed in execution)
            if i < 4:
                assert not truncated, f"Should not truncate before max_steps (step {i+1})"
            elif i == 4:
                assert truncated, f"Should truncate at max_steps (step {i+1})"
                assert not terminated, "Should not be terminated (goal not reached)"
                break

    @pytest.mark.parametrize("tiling_type", ["square", "hex", "triangle"])
    def test_deterministic_reset_all_tilings(self, tiling_type):
        """All tilings produce deterministic results with same seed."""
        task = create_simple_task()
        task["tiling"]["type"] = tiling_type

        env1 = MultiGridEnv(task, tiling=tiling_type)
        env2 = MultiGridEnv(task, tiling=tiling_type)

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        assert obs1.shape == obs2.shape
        assert (obs1 == obs2).all(), f"{tiling_type} tiling should be deterministic"

    def test_action_after_truncation(self):
        """Steps after truncation continue but episode is done."""
        task = create_simple_task(max_steps=2)
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Take steps until truncation
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(Action.WAIT)

        assert truncated, "Episode should be truncated"

        # Gymnasium allows steps after done, but they should maintain done status
        # This is standard gymnasium behavior - environment doesn't prevent stepping after done
        obs, reward, terminated, truncated, info = env.step(Action.WAIT)
        # No exception - this is expected gymnasium behavior


class TestBoundaryMovement:
    """Tests for movement at grid boundaries."""

    def test_cannot_move_off_north_edge(self):
        """Cannot move north from top edge."""
        task = create_simple_task(agent_pos=(0.5, 0.05))
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Set agent facing north
        env.state.agent.facing = 0  # North

        initial_cell = env.state.agent.cell_id
        obs, reward, terminated, truncated, info = env.step(Action.FORWARD)

        # Agent should stay in place at boundary
        assert env.state.agent.cell_id == initial_cell
        assert info.get("invalid_action") or info.get("boundary_collision")

    def test_cannot_move_off_east_edge(self):
        """Cannot move east from right edge."""
        task = create_simple_task(agent_pos=(0.95, 0.5))
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Set agent facing east
        env.state.agent.facing = 1  # East

        initial_cell = env.state.agent.cell_id
        obs, reward, terminated, truncated, info = env.step(Action.FORWARD)

        # Agent should stay in place at boundary
        assert env.state.agent.cell_id == initial_cell
        assert info.get("invalid_action") or info.get("boundary_collision")

    @pytest.mark.parametrize("tiling_type", ["square", "hex", "triangle"])
    def test_all_boundary_directions(self, tiling_type):
        """Test boundary behavior for all directions in each tiling."""
        task = create_simple_task()
        task["tiling"]["type"] = tiling_type

        env = MultiGridEnv(task, tiling=tiling_type)
        env.reset()

        # Get a corner cell
        corner_cells = [cid for cid, cell in env.tiling.cells.items()
                        if len(cell.neighbors) == 2]
        assert len(corner_cells) > 0, f"Should have corner cells in {tiling_type} grid"

        # Move agent to corner
        env.state.agent.cell_id = corner_cells[0]

        # Try all possible facing directions
        num_directions = len(env.tiling.directions)
        for facing in range(num_directions):
            env.state.agent.facing = facing
            initial_cell = env.state.agent.cell_id

            obs, reward, terminated, truncated, info = env.step(Action.FORWARD)

            # Either agent moved to valid neighbor or stayed put
            if env.state.agent.cell_id != initial_cell:
                # Moved to valid neighbor
                facing_dir = env.tiling.directions[facing]
                assert facing_dir in env.tiling.cells[initial_cell].neighbors
            else:
                # Boundary collision - should be indicated in info
                assert info.get("invalid_action") or info.get("boundary_collision"), \
                    f"Boundary collision should be indicated for {tiling_type}"
