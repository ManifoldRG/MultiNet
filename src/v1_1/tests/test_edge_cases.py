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


    def test_push_at_boundary(self):
        """Pushing object at grid boundary fails (destination off-grid)."""
        # Place movable object at east edge, agent behind it facing east
        task = create_simple_task(grid_size=8)
        # Object at right edge
        task["scene"]["objects"][0]["position"] = {"x": 0.95, "y": 0.5}
        # Agent one cell to the left of object
        task["scene"]["agent"]["position"] = {"x": 0.80, "y": 0.5}

        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Place agent facing east (toward the boundary object)
        env.state.agent.facing = 1  # East

        # Find the object and ensure agent is adjacent
        obj = list(env.state.objects.values())[0]
        obj_cell = obj.cell_id

        # Move agent to the cell west of the object
        west_of_obj = env.tiling.get_neighbor(obj_cell, "west")
        assert west_of_obj is not None, "Object should not be at west edge"
        env.state.agent.cell_id = west_of_obj
        env.state.agent.facing = 1  # East

        # Push should fail because destination (east of object) is off-grid or blocked
        obs, reward, terminated, truncated, info = env.step(Action.PUSH)
        assert info["invalid_action"] is True, "Push at boundary should be invalid"


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


class TestObjectInteractions:
    """Tests for object interaction edge cases."""

    def _create_task_with_two_movables(self):
        """Helper: task with two movable objects next to agent."""
        return {
            "task_id": "test_obj_interact",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "obj_a",
                        "type": "movable",
                        "color": "red",
                        "position": {"x": 0.5, "y": 0.3},
                        "size": 0.1,
                    },
                    {
                        "id": "obj_b",
                        "type": "movable",
                        "color": "blue",
                        "position": {"x": 0.5, "y": 0.7},
                        "size": 0.1,
                    },
                ],
                "agent": {"position": {"x": 0.5, "y": 0.5}, "facing": 0},
            },
            "goal": {"predicate": "reach_position", "position": {"x": 0.9, "y": 0.9}},
            "limits": {"max_steps": 50},
            "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}},
        }

    def test_pickup_while_holding(self):
        """Picking up a second object while already holding one is invalid."""
        task = self._create_task_with_two_movables()
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Face north toward obj_a and pick it up
        env.state.agent.facing = 0  # North
        obj_a = env.state.objects["obj_a"]

        # Place agent directly south of obj_a
        south_of_a = env.tiling.get_neighbor(obj_a.cell_id, "south")
        if south_of_a:
            env.state.agent.cell_id = south_of_a
        env.state.agent.facing = 0  # North

        obs, reward, terminated, truncated, info = env.step(Action.PICKUP)
        assert env.state.agent.holding is not None, "Should have picked up obj_a"

        # Now try to pick up obj_b — should fail
        obj_b = env.state.objects["obj_b"]
        south_of_b = env.tiling.get_neighbor(obj_b.cell_id, "south")
        if south_of_b:
            env.state.agent.cell_id = south_of_b
        env.state.agent.facing = 0  # North

        obs, reward, terminated, truncated, info = env.step(Action.PICKUP)
        assert info["invalid_action"] is True, "Pickup while holding should be invalid"

    def test_drop_with_nothing(self):
        """Dropping when not holding anything is invalid."""
        task = create_simple_task()
        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Agent starts empty-handed
        assert env.state.agent.holding is None

        obs, reward, terminated, truncated, info = env.step(Action.DROP)
        assert info["invalid_action"] is True, "Drop with nothing should be invalid"

    def test_push_nothing(self):
        """Pushing when facing an empty cell is invalid."""
        task = create_simple_task(grid_size=10, agent_pos=(0.5, 0.5))
        # Remove all objects so agent faces empty cells
        task["scene"]["objects"] = []

        env = MultiGridEnv(task, tiling="square")
        env.reset()

        env.state.agent.facing = 1  # East

        obs, reward, terminated, truncated, info = env.step(Action.PUSH)
        assert info["invalid_action"] is True, "Push nothing should be invalid"

    def test_push_chain(self):
        """Pushing object into another object (chain) is invalid."""
        task = {
            "task_id": "test_push_chain",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "block_near",
                        "type": "movable",
                        "color": "red",
                        "position": {"x": 0.5, "y": 0.5},
                        "size": 0.1,
                    },
                    {
                        "id": "block_far",
                        "type": "movable",
                        "color": "blue",
                        "position": {"x": 0.5, "y": 0.3},
                        "size": 0.1,
                    },
                ],
                "agent": {"position": {"x": 0.5, "y": 0.7}, "facing": 0},
            },
            "goal": {"predicate": "reach_position", "position": {"x": 0.9, "y": 0.9}},
            "limits": {"max_steps": 50},
            "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}},
        }

        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # Arrange: agent south of block_near, block_far north of block_near
        block_near = env.state.objects["block_near"]
        block_far = env.state.objects["block_far"]

        # Ensure they're in a north-south line
        north_of_near = env.tiling.get_neighbor(block_near.cell_id, "north")
        south_of_near = env.tiling.get_neighbor(block_near.cell_id, "south")

        # Place block_far directly north of block_near
        block_far.cell_id = north_of_near
        # Place agent directly south of block_near
        env.state.agent.cell_id = south_of_near
        env.state.agent.facing = 0  # North

        obs, reward, terminated, truncated, info = env.step(Action.PUSH)
        assert info["invalid_action"] is True, "Push chain should be invalid (destination blocked)"


class TestZones:
    """Tests for zone functionality (covered_cells and ObjectInZoneGoal)."""

    def test_zone_at_boundary(self):
        """Zone at grid corner: all covered cells must be valid."""
        task = {
            "task_id": "test_zone_boundary",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "zone_corner",
                        "type": "zone",
                        "color": "blue",
                        "position": {"x": 0.01, "y": 0.01},
                        "radius_hops": 2,
                    }
                ],
                "agent": {"position": {"x": 0.5, "y": 0.5}, "facing": 0},
            },
            "goal": {"predicate": "reach_position", "position": {"x": 0.9, "y": 0.9}},
            "limits": {"max_steps": 50},
            "tiling": {"type": "square", "grid_size": {"width": 8, "height": 8}},
        }

        env = MultiGridEnv(task, tiling="square")
        env.reset()

        zone = env.state.objects["zone_corner"]
        assert len(zone.covered_cells) > 0, "Zone should have covered cells"

        # All covered cells must exist in the tiling
        for cell_id in zone.covered_cells:
            assert cell_id in env.tiling.cells, f"Covered cell {cell_id} not in tiling"

        # At a corner with radius 2, should have fewer cells than a center zone
        # (boundary limits expansion)
        assert len(zone.covered_cells) < (2 * 2 + 1) ** 2, \
            "Corner zone should have fewer cells than an unbounded zone"

    def test_zone_radius_zero(self):
        """Zone with radius_hops=0 covers exactly one cell (the center)."""
        task = {
            "task_id": "test_zone_r0",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "zone_single",
                        "type": "zone",
                        "color": "green",
                        "position": {"x": 0.5, "y": 0.5},
                        "radius_hops": 0,
                    }
                ],
                "agent": {"position": {"x": 0.2, "y": 0.2}, "facing": 0},
            },
            "goal": {"predicate": "reach_position", "position": {"x": 0.9, "y": 0.9}},
            "limits": {"max_steps": 50},
            "tiling": {"type": "square", "grid_size": {"width": 8, "height": 8}},
        }

        env = MultiGridEnv(task, tiling="square")
        env.reset()

        zone = env.state.objects["zone_single"]
        assert len(zone.covered_cells) == 1, \
            f"Radius-0 zone should cover exactly 1 cell, got {len(zone.covered_cells)}"
        assert zone.cell_id in zone.covered_cells, \
            "Radius-0 zone's covered cell should be its own cell"

    def test_consecutive_steps_in_zone(self):
        """ObjectInZoneGoal with consecutive_steps=3 requires 3 checks in a row."""
        from multigrid.goals import ObjectInZoneGoal

        task = {
            "task_id": "test_consec_zone",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "zone_target",
                        "type": "zone",
                        "color": "blue",
                        "position": {"x": 0.5, "y": 0.5},
                        "radius_hops": 2,
                    },
                    {
                        "id": "cube_red",
                        "type": "movable",
                        "color": "red",
                        "position": {"x": 0.5, "y": 0.5},
                        "size": 0.1,
                    },
                ],
                "agent": {"position": {"x": 0.2, "y": 0.2}, "facing": 0},
            },
            "goal": {
                "type": "object_in_zone",
                "object_id": "cube_red",
                "zone_id": "zone_target",
                "consecutive_steps": 3,
            },
            "limits": {"max_steps": 50},
            "tiling": {"type": "square", "grid_size": {"width": 8, "height": 8}},
        }

        env = MultiGridEnv(task, tiling="square")
        env.reset()

        # The cube starts in the zone. Step WAIT 3 times — goal should trigger
        # on the 3rd step (consecutive_steps=3).
        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(Action.WAIT)
            assert not terminated, f"Goal should not be achieved on step {i+1}"

        obs, reward, terminated, truncated, info = env.step(Action.WAIT)
        assert terminated, "Goal should be achieved after 3 consecutive steps in zone"
