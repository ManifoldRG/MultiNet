"""Tests for model interface, evaluation harness, and NL domain."""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

import numpy as np
from model_interface import ModelInterface, ModelInput, ModelOutput, RandomModelInterface
from evaluation_harness import EvaluationHarness, TierMetrics, EvaluationResult
from gridworld.task_spec import TaskSpecification
from gridworld.actions import ACTION_NAMES


class TestModelInput:
    def test_create_model_input(self):
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="Navigate to the goal",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        assert inp.image.shape == (64, 64, 3)
        assert inp.step_number == 1

    def test_optional_context(self):
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space={0: "left"},
            step_number=0,
            max_steps=10,
            additional_context="Extra info",
        )
        assert inp.additional_context == "Extra info"


class TestRandomModel:
    def test_random_model_name(self):
        model = RandomModelInterface(seed=42)
        assert model.model_name == "random"

    def test_random_model_predict(self):
        model = RandomModelInterface(seed=42)
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        output = model.predict(inp)
        assert isinstance(output, ModelOutput)
        assert output.action in ACTION_NAMES

    def test_random_model_deterministic(self):
        """Same seed should produce same sequence."""
        model1 = RandomModelInterface(seed=123)
        model2 = RandomModelInterface(seed=123)
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        actions1 = [model1.predict(inp).action for _ in range(10)]
        actions2 = [model2.predict(inp).action for _ in range(10)]
        assert actions1 == actions2

    def test_random_model_batch(self):
        model = RandomModelInterface(seed=42)
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        outputs = model.predict_batch([inp, inp, inp])
        assert len(outputs) == 3
        assert all(isinstance(o, ModelOutput) for o in outputs)


class TestEvaluationHarness:
    @pytest.fixture
    def simple_spec(self):
        return TaskSpecification.from_dict({
            "task_id": "test_simple",
            "seed": 42,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [6, 6],
                "walls": [],
                "start": [1, 1],
                "goal": [4, 4],
            },
            "goal": {"type": "reach_position", "target": [4, 4]},
            "max_steps": 20,
        })

    def test_evaluate_single_task(self, simple_spec):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        result = harness.evaluate_task(simple_spec)
        assert result.task_id == "test_simple"
        assert result.steps_taken > 0
        assert result.steps_taken <= 20
        harness.close()

    def test_evaluate_tier(self):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        task_dir = str(Path(__file__).resolve().parent.parent / "gridworld" / "tasks")
        metrics = harness.evaluate_tier(tier=1, task_dir=task_dir)
        assert isinstance(metrics, TierMetrics)
        assert metrics.tier == 1
        assert metrics.num_tasks == 3  # 3 tier1 tasks
        assert 0.0 <= metrics.success_rate <= 1.0
        harness.close()

    def test_evaluate_all(self):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        task_dir = str(Path(__file__).resolve().parent.parent / "gridworld" / "tasks")
        result = harness.evaluate_all(task_dir=task_dir, tiers=[1])
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "random"
        assert 1 in result.tier_metrics
        harness.close()

    def test_result_serialization(self):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        task_dir = str(Path(__file__).resolve().parent.parent / "gridworld" / "tasks")
        result = harness.evaluate_all(task_dir=task_dir, tiers=[1])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.save(f.name)
            with open(f.name) as fp:
                data = json.load(fp)
            assert "model_name" in data
            assert "tier_metrics" in data
            os.unlink(f.name)
        harness.close()


class TestNLActionParser:
    @pytest.fixture
    def parser(self):
        from nl_domain.nl_action_parser import NLActionParser
        return NLActionParser()

    def test_forward_commands(self, parser):
        for cmd in ["go forward", "move forward", "forward", "walk ahead", "advance"]:
            actions = parser.parse(cmd)
            assert actions == [2], f"'{cmd}' should parse to forward (2), got {actions}"

    def test_turn_commands(self, parser):
        assert parser.parse("turn left") == [0]
        assert parser.parse("turn right") == [1]
        assert parser.parse("rotate left") == [0]

    def test_interaction_commands(self, parser):
        assert parser.parse("pick up") == [3]
        assert parser.parse("grab") == [3]
        assert parser.parse("drop") == [4]
        assert parser.parse("toggle") == [5]
        assert parser.parse("open") == [5]
        assert parser.parse("press") == [5]

    def test_wait_commands(self, parser):
        for cmd in ["wait", "stay", "do nothing", "done"]:
            actions = parser.parse(cmd)
            assert actions == [6], f"'{cmd}' should parse to done (6), got {actions}"

    def test_compass_north(self, parser):
        """Moving north when facing right should turn left then forward."""
        # Agent facing right (0), need to face up (3)
        # Right to up: turn left once (CCW: 0->3 is one left turn)
        actions = parser.parse("move north", agent_facing=0)
        assert actions[-1] == 2  # Last action should be forward
        assert 0 in actions  # Should include turn_left

    def test_compass_same_direction(self, parser):
        """Moving north when already facing north should just go forward."""
        actions = parser.parse("move north", agent_facing=3)
        assert actions == [2]  # Just forward

    def test_compound_commands(self, parser):
        actions = parser.parse("turn left then go forward")
        assert actions == [0, 2]

    def test_empty_command(self, parser):
        actions = parser.parse("")
        assert actions == [6]  # Wait


class TestNLGridWorldEnv:
    def test_nl_env_basic(self):
        from nl_domain.nl_env import NLGridWorldEnv
        spec = TaskSpecification.from_dict({
            "task_id": "test_nl",
            "seed": 42,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [6, 6],
                "walls": [],
                "start": [1, 1],
                "goal": [4, 4],
            },
            "goal": {"type": "reach_position", "target": [4, 4]},
            "max_steps": 20,
        })

        env = NLGridWorldEnv(spec)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert "mission" in info

        obs, reward, term, trunc, info = env.step("go forward")
        assert obs is not None
        assert "parsed_actions" in info
        assert info["parsed_actions"] == [2]  # forward

        env.close()


class TestCrossDomain:
    def test_canonical_roundtrip(self):
        from cross_domain.canonical_task_spec import CanonicalTaskSpec, CanonicalGoal, CanonicalObject
        from cross_domain.gridworld_adapter import GridWorldDomainAdapter

        spec = TaskSpecification.from_dict({
            "task_id": "test_roundtrip",
            "seed": 42,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [10, 10],
                "walls": [[3, 3], [3, 4]],
                "start": [1, 1],
                "goal": [8, 8],
            },
            "mechanisms": {
                "keys": [{"id": "k1", "position": [2, 2], "color": "yellow"}],
            },
            "goal": {"type": "reach_position", "target": [8, 8]},
            "max_steps": 100,
        })

        adapter = GridWorldDomainAdapter()
        canonical = adapter.to_canonical(spec)

        assert canonical.task_id == "test_roundtrip"
        assert canonical.difficulty == 1
        assert 0.0 <= canonical.agent_start[0] <= 1.0
        assert 0.0 <= canonical.agent_start[1] <= 1.0
        assert canonical.goal.goal_type == "reach"
        assert len(canonical.objects) > 0  # walls + key

        # Find the key in canonical objects
        key_objs = [o for o in canonical.objects if o.obj_type == "collectible"]
        assert len(key_objs) == 1
        assert key_objs[0].id == "k1"

    def test_gui_action_dataclass(self):
        from cross_domain.domain_adapter import GUIAction
        action = GUIAction(action_type="mouse_click", x=0.5, y=0.3)
        assert action.action_type == "mouse_click"
        assert action.x == 0.5
