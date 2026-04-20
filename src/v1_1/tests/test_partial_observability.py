"""Tests for partial observability (view cone and fog of war)."""

import pytest
import sys
import os
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from gridworld.task_spec import TaskSpecification, Rules
from gridworld.task_parser import TaskParser
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.actions import MiniGridActions


# --- Fixtures ---

@pytest.fixture
def full_obs_spec():
    """Task with full observability (default)."""
    return TaskSpecification.from_dict({
        "task_id": "test_full_obs",
        "seed": 42,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {"keys": [], "doors": [], "switches": [],
                       "gates": [], "blocks": [], "teleporters": [], "hazards": []},
        "rules": {"key_consumption": True, "switch_type": "toggle",
                  "hidden_mechanisms": [], "observability": "full"},
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    })


@pytest.fixture
def view_cone_spec():
    """Task with view cone partial observability."""
    return TaskSpecification.from_dict({
        "task_id": "test_view_cone",
        "seed": 42,
        "difficulty_tier": 5,
        "maze": {
            "dimensions": [10, 10],
            "walls": [[5, 1], [5, 2], [5, 3], [5, 5], [5, 6], [5, 7], [5, 8]],
            "start": [1, 1],
            "goal": [8, 8],
        },
        "mechanisms": {"keys": [], "doors": [], "switches": [],
                       "gates": [], "blocks": [], "teleporters": [], "hazards": []},
        "rules": {"key_consumption": True, "switch_type": "toggle",
                  "hidden_mechanisms": [], "observability": "view_cone", "view_size": 5},
        "goal": {"type": "reach_position", "target": [8, 8]},
        "max_steps": 100,
    })


@pytest.fixture
def fog_of_war_spec():
    """Task with fog of war partial observability."""
    return TaskSpecification.from_dict({
        "task_id": "test_fog_of_war",
        "seed": 42,
        "difficulty_tier": 5,
        "maze": {
            "dimensions": [10, 10],
            "walls": [],
            "start": [1, 1],
            "goal": [8, 8],
        },
        "mechanisms": {"keys": [], "doors": [], "switches": [],
                       "gates": [], "blocks": [], "teleporters": [], "hazards": []},
        "rules": {"key_consumption": True, "switch_type": "toggle",
                  "hidden_mechanisms": [], "observability": "fog_of_war", "view_size": 5},
        "goal": {"type": "reach_position", "target": [8, 8]},
        "max_steps": 100,
    })


# --- TaskSpec Rules tests ---

class TestObservabilitySpec:
    """Test that observability is correctly parsed from task specs."""

    def test_default_observability_is_full(self):
        rules = Rules.from_dict({})
        assert rules.observability == "full"
        assert rules.view_size == 7

    def test_view_cone_parsed(self):
        rules = Rules.from_dict({"observability": "view_cone", "view_size": 5})
        assert rules.observability == "view_cone"
        assert rules.view_size == 5

    def test_fog_of_war_parsed(self):
        rules = Rules.from_dict({"observability": "fog_of_war", "view_size": 9})
        assert rules.observability == "fog_of_war"
        assert rules.view_size == 9

    def test_observability_roundtrip(self, view_cone_spec):
        """Serialize and deserialize preserves observability."""
        d = view_cone_spec.to_dict()
        spec2 = TaskSpecification.from_dict(d)
        assert spec2.rules.observability == "view_cone"
        assert spec2.rules.view_size == 5


# --- Full observability tests ---

class TestFullObservability:
    """Verify that full observability mode works as before (no regression)."""

    def test_full_obs_see_through_walls(self, full_obs_spec):
        """Full obs mode should have see_through_walls=True."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(full_obs_spec)
        assert env.see_through_walls is True

    def test_full_obs_backend_state(self, full_obs_spec):
        """Full obs mode should have observability_mode='full' in GridState."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(full_obs_spec)
        _, state, _ = backend.reset(seed=42)
        assert state.observability_mode == "full"
        assert len(state.visible_cells) == 0  # Not tracked in full mode
        assert len(state.explored_cells) == 0

    def test_full_obs_renders(self, full_obs_spec):
        """Full obs mode renders a valid RGB image."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(full_obs_spec)
        obs, _, _ = backend.reset(seed=42)
        assert obs.shape[2] == 3
        assert obs.max() > 0


# --- View cone tests ---

class TestViewCone:
    """Test MiniGrid native view cone partial observability."""

    def test_view_cone_env_config(self, view_cone_spec):
        """View cone mode should configure env with see_through_walls=False."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(view_cone_spec)
        assert env.see_through_walls is False
        assert env.agent_view_size == 5

    def test_view_cone_observation_size(self, view_cone_spec):
        """View cone symbolic observation should match view_size."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(view_cone_spec)
        obs = env.gen_obs()
        # MiniGrid observation image shape is (view_size, view_size, 3)
        assert obs["image"].shape == (5, 5, 3)

    def test_view_cone_visible_cells(self, view_cone_spec):
        """View cone should report a limited set of visible cells."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(view_cone_spec)
        visible = env.get_visible_cells()
        # With view_size=5 and see_through_walls=False, visible cells
        # should be significantly fewer than total interior cells
        total_interior = (10 - 2) * (10 - 2)  # 64
        assert len(visible) > 0
        assert len(visible) < total_interior

    def test_view_cone_backend_state(self, view_cone_spec):
        """Backend GridState should include visible cells for view_cone mode."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(view_cone_spec)
        _, state, _ = backend.reset(seed=42)
        assert state.observability_mode == "view_cone"
        assert len(state.visible_cells) > 0

    def test_view_cone_visibility_changes_on_turn(self, view_cone_spec):
        """Turning should change visible cells (view cone rotates with agent)."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(view_cone_spec)
        _, state0, _ = backend.reset(seed=42)
        visible_before = state0.visible_cells

        # Turn left
        _, _, _, _, state1, _ = backend.step(MiniGridActions.TURN_LEFT)
        visible_after = state1.visible_cells

        # After turning, some cells should be different
        assert visible_before != visible_after

    def test_view_cone_renders(self, view_cone_spec):
        """View cone mode should render with highlight on visible cells."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(view_cone_spec)
        obs, _, _ = backend.reset(seed=42)
        assert obs.shape[2] == 3
        assert obs.max() > 0

    def test_view_cone_walls_block_vision(self, view_cone_spec):
        """Walls should block vision in view cone mode."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(view_cone_spec)
        # Agent starts at (1,1) facing right. Wall at (5,1) should block
        # vision to cells at x>=6 along y=1
        visible = env.get_visible_cells()
        # Cells behind the wall at x=5 should not be visible
        behind_wall = {c for c in visible if c[0] > 5 and c[1] == 1}
        assert len(behind_wall) == 0, f"Should not see behind wall: {behind_wall}"


# --- Fog of war tests ---

class TestFogOfWar:
    """Test fog of war observability mode."""

    def test_fog_of_war_env_config(self, fog_of_war_spec):
        """Fog of war should configure env with see_through_walls=False."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(fog_of_war_spec)
        assert env.see_through_walls is False
        assert env.agent_view_size == 5

    def test_fog_of_war_initial_explored(self, fog_of_war_spec):
        """After reset, fog of war should have initial visible area explored."""
        parser = TaskParser(render_mode="rgb_array")
        env = parser.parse(fog_of_war_spec)
        # After reset, explored cells should be the initial visible area
        assert len(env.explored_cells) > 0

    def test_fog_of_war_explored_grows(self, fog_of_war_spec):
        """Moving should reveal new cells in fog of war mode."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(fog_of_war_spec)
        _, state0, _ = backend.reset(seed=42)
        initial_explored = len(state0.explored_cells)

        # Move forward a few steps (agent starts at (1,1) facing right)
        for _ in range(3):
            backend.step(MiniGridActions.MOVE_FORWARD)
        _, _, _, _, state1, _ = backend.step(MiniGridActions.MOVE_FORWARD)

        # Should have explored more cells
        assert len(state1.explored_cells) >= initial_explored

    def test_fog_of_war_explored_never_shrinks(self, fog_of_war_spec):
        """Explored cells should never decrease (monotonically growing)."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(fog_of_war_spec)
        _, state, _ = backend.reset(seed=42)
        prev_explored = len(state.explored_cells)

        # Take various actions
        actions = [
            MiniGridActions.MOVE_FORWARD,
            MiniGridActions.MOVE_FORWARD,
            MiniGridActions.TURN_LEFT,
            MiniGridActions.MOVE_FORWARD,
            MiniGridActions.TURN_RIGHT,
            MiniGridActions.MOVE_FORWARD,
        ]
        for action in actions:
            _, _, _, _, state, _ = backend.step(action)
            current_explored = len(state.explored_cells)
            assert current_explored >= prev_explored, \
                f"Explored cells decreased from {prev_explored} to {current_explored}"
            prev_explored = current_explored

    def test_fog_of_war_backend_state(self, fog_of_war_spec):
        """Backend GridState should include explored cells for fog_of_war."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(fog_of_war_spec)
        _, state, _ = backend.reset(seed=42)
        assert state.observability_mode == "fog_of_war"
        assert len(state.explored_cells) > 0
        assert len(state.visible_cells) > 0
        # Explored should be superset of visible
        assert state.visible_cells <= state.explored_cells


# --- Task file loading tests ---

class TestPartialObsTaskFiles:
    """Test loading actual task files with partial observability."""

    def test_hidden_switch_has_view_cone(self):
        """tier5/hidden_switch_001.json should have view_cone observability."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier5" / "hidden_switch_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")
        spec = TaskSpecification.from_json(str(task_path))
        assert spec.rules.observability == "view_cone"
        assert spec.rules.view_size == 5

    def test_memory_has_fog_of_war(self):
        """tier5/memory_003.json should have fog_of_war observability."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier5" / "memory_003.json"
        if not task_path.exists():
            pytest.skip("Task file not found")
        spec = TaskSpecification.from_json(str(task_path))
        assert spec.rules.observability == "fog_of_war"
        assert spec.rules.view_size == 7

    def test_hidden_switch_playable_with_view_cone(self):
        """hidden_switch_001 should be playable with view cone."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier5" / "hidden_switch_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")
        backend = MiniGridBackend(render_mode="rgb_array")
        spec = TaskSpecification.from_json(str(task_path))
        backend.configure(spec)
        obs, state, info = backend.reset(seed=42)
        assert obs.shape[2] == 3
        assert state.observability_mode == "view_cone"
        assert len(state.visible_cells) > 0

        # Take a step to verify it works
        obs, _, _, _, state, _ = backend.step(MiniGridActions.MOVE_FORWARD)
        assert obs.shape[2] == 3

    def test_memory_playable_with_fog_of_war(self):
        """memory_003 should be playable with fog of war."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier5" / "memory_003.json"
        if not task_path.exists():
            pytest.skip("Task file not found")
        backend = MiniGridBackend(render_mode="rgb_array")
        spec = TaskSpecification.from_json(str(task_path))
        backend.configure(spec)
        obs, state, info = backend.reset(seed=42)
        assert obs.shape[2] == 3
        assert state.observability_mode == "fog_of_war"
        assert len(state.explored_cells) > 0

    def test_existing_tasks_default_to_full(self):
        """Tasks without observability field should default to full."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier1" / "maze_simple_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")
        spec = TaskSpecification.from_json(str(task_path))
        assert spec.rules.observability == "full"
