"""Tests for teleporter functionality in MiniGrid backend."""

import pytest
import sys
import os
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from gridworld.task_spec import TaskSpecification
from gridworld.task_parser import TaskParser
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.actions import MiniGridActions
from gridworld.custom_env import TeleporterObj


@pytest.fixture
def teleporter_spec():
    """Create a simple task with a teleporter."""
    return TaskSpecification.from_dict({
        "task_id": "test_teleporter",
        "seed": 42,
        "difficulty_tier": 5,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "teleporters": [
                {
                    "id": "tp1",
                    "position_a": [2, 1],
                    "position_b": [5, 5],
                    "bidirectional": True,
                }
            ]
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    })


@pytest.fixture
def oneway_teleporter_spec():
    """Create a task with a one-way teleporter."""
    return TaskSpecification.from_dict({
        "task_id": "test_oneway_teleporter",
        "seed": 42,
        "difficulty_tier": 5,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "teleporters": [
                {
                    "id": "tp1",
                    "position_a": [2, 1],
                    "position_b": [5, 5],
                    "bidirectional": False,
                }
            ]
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    })


class TestTeleporterValidation:
    """Test teleporter position validation in task_spec."""

    def test_valid_teleporter_passes_validation(self, teleporter_spec):
        is_valid, errors = teleporter_spec.validate()
        assert is_valid, f"Validation errors: {errors}"

    def test_oob_teleporter_a_fails(self):
        spec = TaskSpecification.from_dict({
            "task_id": "test",
            "seed": 42,
            "difficulty_tier": 5,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "teleporters": [{"id": "tp", "position_a": [10, 10], "position_b": [3, 3]}]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })
        is_valid, errors = spec.validate()
        assert not is_valid
        assert any("Teleporter" in e and "endpoint A" in e for e in errors)

    def test_oob_teleporter_b_fails(self):
        spec = TaskSpecification.from_dict({
            "task_id": "test",
            "seed": 42,
            "difficulty_tier": 5,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "teleporters": [{"id": "tp", "position_a": [3, 3], "position_b": [10, 10]}]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })
        is_valid, errors = spec.validate()
        assert not is_valid
        assert any("Teleporter" in e and "endpoint B" in e for e in errors)


class TestTeleporterPlacement:
    """Test that teleporters are placed in the environment."""

    def test_teleporter_objects_placed(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        obs, state, info = backend.reset(seed=42)

        assert len(backend.env.teleporters) == 2  # Two endpoints
        assert "tp1_a" in backend.env.teleporters
        assert "tp1_b" in backend.env.teleporters

    def test_teleporter_objects_are_correct_type(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        backend.reset(seed=42)

        for tp in backend.env.teleporters.values():
            assert isinstance(tp, TeleporterObj)

    def test_bidirectional_partners(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        backend.reset(seed=42)

        tp_a = backend.env.teleporters["tp1_a"]
        tp_b = backend.env.teleporters["tp1_b"]
        assert tp_a.partner is tp_b
        assert tp_b.partner is tp_a

    def test_oneway_partner(self, oneway_teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(oneway_teleporter_spec)
        backend.reset(seed=42)

        tp_a = backend.env.teleporters["tp1_a"]
        tp_b = backend.env.teleporters["tp1_b"]
        assert tp_a.partner is tp_b
        assert tp_b.partner is None  # One-way: B doesn't teleport to A


class TestTeleporterMechanics:
    """Test teleporter step mechanics."""

    def test_agent_teleports_on_step(self, teleporter_spec):
        """Agent at (1,1) facing right, move forward to (2,1) which is teleporter A -> should teleport to (5,5)."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        obs, state, info = backend.reset(seed=42)

        # Agent starts at (1,1) facing right (dir=0)
        assert state.agent_position == (1, 1)

        # Move forward: agent goes to (2,1) where teleporter A is
        obs, reward, term, trunc, state, info = backend.step(MiniGridActions.MOVE_FORWARD)

        # Should have been teleported to (5,5)
        assert state.agent_position == (5, 5), f"Expected (5,5), got {state.agent_position}"

    def test_teleporter_cooldown_in_state(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        obs, state, info = backend.reset(seed=42)

        # Check that teleporter cooldowns are tracked
        assert "tp1_a" in state.teleporter_cooldowns
        assert "tp1_b" in state.teleporter_cooldowns
        assert state.teleporter_cooldowns["tp1_a"] == 0
        assert state.teleporter_cooldowns["tp1_b"] == 0


class TestTeleporterTaskFile:
    """Test loading the tier5 teleporter task JSON."""

    def test_load_teleporter_task(self):
        task_path = Path(__file__).resolve().parent.parent / "gridworld" / "tasks" / "tier5" / "teleporter_004.json"
        spec = TaskSpecification.from_json(str(task_path))
        assert spec.task_id == "tier5_teleporter_004"
        assert len(spec.mechanisms.teleporters) == 2

    def test_teleporter_task_validates(self):
        task_path = Path(__file__).resolve().parent.parent / "gridworld" / "tasks" / "tier5" / "teleporter_004.json"
        spec = TaskSpecification.from_json(str(task_path))
        is_valid, errors = spec.validate()
        assert is_valid, f"Validation errors: {errors}"

    def test_teleporter_task_runs(self):
        task_path = Path(__file__).resolve().parent.parent / "gridworld" / "tasks" / "tier5" / "teleporter_004.json"
        spec = TaskSpecification.from_json(str(task_path))
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(spec)
        obs, state, info = backend.reset(seed=42)
        assert state.agent_position == (1, 1)
        assert len(backend.env.teleporters) == 4  # 2 teleporters * 2 endpoints
