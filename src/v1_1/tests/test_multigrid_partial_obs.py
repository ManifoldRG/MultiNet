"""Tests for MultiGrid partial observability (view cone and fog of war)."""

import pytest
import sys
import os
import math
import numpy as np
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from multigrid.env import MultiGridEnv
from multigrid.visibility import (
    compute_visible_cells,
    _facing_to_angle,
    _is_in_view_cone,
    _is_cell_blocking,
)


# --- Helpers ---

def _make_spec(width=5, height=5, walls=None, objects=None, goal_x=0.9, goal_y=0.9,
               agent_x=0.3, agent_y=0.3, agent_facing=0):
    """Create a minimal MultiGrid task spec dict."""
    spec = {
        "task_id": "test_partial_obs",
        "seed": 1,
        "tiling": {
            "type": "square",
            "grid_size": {"width": width, "height": height},
        },
        "scene": {
            "agent": {
                "position": {"x": agent_x, "y": agent_y},
                "facing": agent_facing,
            },
            "objects": objects or [],
            "walls": walls or [],
        },
        "goal": {
            "type": "reach_position",
            "target": {"x": goal_x, "y": goal_y},
        },
        "limits": {"max_steps": 50},
    }
    return spec


# --- Tests ---

class TestFullObservability:
    """Full observability: all cells should be visible."""

    def test_all_cells_visible(self):
        spec = _make_spec()
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           observability_mode="full")
        obs, info = env.reset(seed=42)
        assert env.state.visible_cells == set(env.tiling.cells.keys())
        assert env.state.explored_cells == set(env.tiling.cells.keys())

    def test_full_obs_no_visibility_info_in_info(self):
        spec = _make_spec()
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           observability_mode="full")
        obs, info = env.reset(seed=42)
        assert "visible_cells" not in info

    @pytest.mark.parametrize("tiling", ["square", "hex"])
    def test_full_obs_all_tilings(self, tiling):
        spec = _make_spec()
        spec["tiling"]["type"] = tiling
        env = MultiGridEnv(spec, tiling=tiling, render_mode="rgb_array",
                           observability_mode="full")
        obs, info = env.reset(seed=42)
        assert len(env.state.visible_cells) == len(env.tiling.cells)


class TestViewCone:
    """View cone: agent only sees cells in front."""

    def test_fewer_visible_than_total(self):
        spec = _make_spec(width=8, height=8)
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="view_cone")
        obs, info = env.reset(seed=42)
        # With radius 2, should see fewer cells than total
        assert len(env.state.visible_cells) < len(env.tiling.cells)
        assert len(env.state.visible_cells) > 0
        # Agent's own cell must always be visible
        assert env.state.agent.cell_id in env.state.visible_cells

    def test_visible_cells_change_on_turn(self):
        spec = _make_spec(width=8, height=8)
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=3,
                           observability_mode="view_cone")
        obs, info = env.reset(seed=42)
        visible_before = set(env.state.visible_cells)

        # Turn right (action 3 = TURN_RIGHT)
        env.step(3)
        visible_after = set(env.state.visible_cells)

        # Visible cells should differ after turning
        assert visible_before != visible_after

    @pytest.mark.parametrize("tiling", ["square", "hex"])
    def test_view_cone_different_tilings(self, tiling):
        spec = _make_spec(width=6, height=6)
        spec["tiling"]["type"] = tiling
        env = MultiGridEnv(spec, tiling=tiling, render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="view_cone")
        obs, info = env.reset(seed=42)
        assert len(env.state.visible_cells) < len(env.tiling.cells)
        assert env.state.agent.cell_id in env.state.visible_cells


class TestWallBlocking:
    """Walls should block BFS visibility propagation."""

    def test_wall_blocks_visibility(self):
        # Place a wall object between agent and some cells
        spec = _make_spec(width=7, height=7, objects=[
            {"id": "wall_1", "type": "wall", "color": "grey",
             "position": {"x": 0.5, "y": 0.3}},
        ])
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=5,
                           observability_mode="fog_of_war")
        obs, info = env.reset(seed=42)

        # The wall cell itself should be visible (walls are visible,
        # just block propagation beyond them)
        wall_cell = env.tiling.canonical_to_cell(0.5, 0.3)
        if wall_cell in env.tiling.cells:
            # Just check visibility is non-trivial (less than all cells)
            assert len(env.state.visible_cells) < len(env.tiling.cells)

    def test_closed_door_blocks(self):
        spec = _make_spec(width=7, height=7, objects=[
            {"id": "door_1", "type": "door", "color": "red",
             "position": {"x": 0.5, "y": 0.3}, "is_locked": True},
        ])
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=5,
                           observability_mode="fog_of_war")
        obs, info = env.reset(seed=42)

        # With a locked door blocking, should see fewer cells
        assert len(env.state.visible_cells) < len(env.tiling.cells)


class TestFogOfWar:
    """Fog of war: explored set grows monotonically."""

    def test_explored_grows_on_movement(self):
        spec = _make_spec(width=8, height=8)
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="fog_of_war")
        obs, info = env.reset(seed=42)
        explored_before = len(env.state.explored_cells)

        # Move forward (action 0 = FORWARD)
        env.step(0)
        explored_after = len(env.state.explored_cells)

        # Explored should be >= (monotonically growing)
        assert explored_after >= explored_before

    def test_explored_never_shrinks(self):
        spec = _make_spec(width=8, height=8)
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="fog_of_war")
        obs, info = env.reset(seed=42)

        # Take a sequence of actions and track explored
        prev_explored = set(env.state.explored_cells)
        actions = [0, 3, 0, 2, 0, 3, 0]  # forward, turn_right, forward, etc.
        for action in actions:
            env.step(action)
            current_explored = set(env.state.explored_cells)
            # Previous explored must be a subset of current
            assert prev_explored.issubset(current_explored), \
                f"Explored cells shrank: lost {prev_explored - current_explored}"
            prev_explored = current_explored

    def test_fog_of_war_omnidirectional(self):
        """Fog of war should be omnidirectional (no facing filter)."""
        spec = _make_spec(width=6, height=6)
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="fog_of_war")
        obs, info = env.reset(seed=42)
        visible_facing_0 = set(env.state.visible_cells)

        # Turn right
        env.step(3)
        visible_after_turn = set(env.state.visible_cells)

        # In fog of war mode (omnidirectional), visible cells should be the same
        # after turning (only position matters, not facing)
        assert visible_facing_0 == visible_after_turn


class TestRendering:
    """Partial observability should affect rendered images."""

    def test_partial_obs_renders_differently(self):
        spec = _make_spec(width=8, height=8)

        # Full observability render
        env_full = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                                observability_mode="full")
        env_full.reset(seed=42)
        img_full = env_full.render()

        # Partial observability render
        env_partial = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                                   partial_obs=True, obs_radius=2,
                                   observability_mode="view_cone")
        env_partial.reset(seed=42)
        img_partial = env_partial.render()

        # Images should differ (partial obs hides some cells)
        assert not np.array_equal(img_full, img_partial)

    def test_render_produces_valid_image(self):
        spec = _make_spec()
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="fog_of_war")
        obs, info = env.reset(seed=42)
        img = env.render()
        assert img.shape == (640, 640, 3)
        assert img.dtype == np.uint8


class TestVisibilityHelpers:
    """Unit tests for visibility module helper functions."""

    def test_facing_to_angle_square(self):
        from multigrid.tilings import SquareTiling
        tiling = SquareTiling()
        tiling.generate_graph(5, 5, seed=0)

        # Square: 0=N (up), 1=E (right), 2=S (down), 3=W (left)
        assert abs(_facing_to_angle(0, tiling) - (-math.pi / 2)) < 0.01
        assert abs(_facing_to_angle(1, tiling) - 0.0) < 0.01

    def test_is_in_view_cone_directly_ahead(self):
        agent_pos = (0.5, 0.5)
        cell_ahead = (0.5, 0.3)  # North (up = -y)
        facing = -math.pi / 2  # North

        assert _is_in_view_cone(agent_pos, cell_ahead, facing, math.pi / 2)

    def test_is_in_view_cone_behind(self):
        agent_pos = (0.5, 0.5)
        cell_behind = (0.5, 0.8)  # South (down = +y)
        facing = -math.pi / 2  # North

        assert not _is_in_view_cone(agent_pos, cell_behind, facing, math.pi / 4)

    def test_is_cell_blocking_empty(self):
        """Empty cell should not block."""
        from multigrid.world import WorldState
        from multigrid.tilings import SquareTiling

        tiling = SquareTiling()
        tiling.generate_graph(5, 5, seed=0)
        state = WorldState(tiling)

        cell_id = list(tiling.cells.keys())[0]
        assert not _is_cell_blocking(cell_id, state)


class TestInfoDict:
    """Test that info dict includes visibility counts."""

    def test_info_has_visibility_counts(self):
        spec = _make_spec()
        env = MultiGridEnv(spec, tiling="square", render_mode="rgb_array",
                           partial_obs=True, obs_radius=2,
                           observability_mode="view_cone")
        obs, info = env.reset(seed=42)

        assert "visible_cells" in info
        assert "explored_cells" in info
        assert "total_cells" in info
        assert info["visible_cells"] > 0
        assert info["explored_cells"] > 0
        assert info["total_cells"] == len(env.tiling.cells)
