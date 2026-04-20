"""Tests for model interface, evaluation harness, and NL domain."""

import pytest
import sys
import os
import json
import tempfile
import urllib.error
import io
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

import numpy as np
from model_interface import ModelInterface, ModelInput, ModelOutput, RandomModelInterface
from evaluation_harness import (
    BenchmarkEvaluationResult,
    EvaluationHarness,
    EvaluationResult,
    TierMetrics,
)
from gridworld.task_spec import TaskSpecification
from gridworld.actions import ACTION_NAMES
from adapters.lmstudio_vlm_adapter import LMStudioVLMAdapter
from adapters.ollama_vlm_adapter import OllamaVLMAdapter


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


class TestLMStudioAdapter:
    def test_setup_fails_clearly_when_server_unavailable(self):
        adapter = LMStudioVLMAdapter(model="google/gemma-3-4b-it", base_url="http://localhost:9")
        with pytest.raises(RuntimeError, match="Could not reach LM Studio"):
            adapter.setup()

    def test_http_error_includes_response_body(self):
        adapter = LMStudioVLMAdapter(model="test-model")
        error = urllib.error.HTTPError(
            url="http://localhost:1234/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"too many images"}'),
        )

        detail = adapter._format_request_error(error, prior_count=2)

        assert "400" in detail
        assert "too many images" in detail
        assert "2 prior image" in detail

    def test_predict_retries_with_fewer_prior_images(self):
        class RetryAdapter(LMStudioVLMAdapter):
            def __init__(self):
                super().__init__(model="test-model", max_prior_images=2)
                self.attempts = []

            def _predict_once(self, input: ModelInput, text_prompt: str, prior_images: list[np.ndarray]) -> str:
                self.attempts.append(len(prior_images))
                if len(prior_images) > 1:
                    raise urllib.error.HTTPError(
                        url="http://localhost:1234/v1/chat/completions",
                        code=400,
                        msg="Bad Request",
                        hdrs=None,
                        fp=io.BytesIO(b'{"error":"payload too large"}'),
                    )
                return "2\nmove forward"

        adapter = RetryAdapter()
        output = adapter.predict(
            ModelInput(
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                text_prompt="Reach the goal",
                action_space=ACTION_NAMES,
                step_number=1,
                max_steps=20,
                prior_images=[
                    np.zeros((64, 64, 3), dtype=np.uint8),
                    np.ones((64, 64, 3), dtype=np.uint8),
                ],
            )
        )

        assert output.action == 2
        assert adapter.attempts == [2, 1]
        assert "reduced prior images from 2 to 1" in output.reasoning


class TestOllamaAdapter:
    def test_parse_response_prefers_explicit_action_line(self):
        adapter = OllamaVLMAdapter(model="test-model")
        action, confidence, reasoning = adapter._parse_response(
            "I see the agent facing right.\nAction: 1",
            ACTION_NAMES,
        )
        assert action == 1
        assert confidence is None
        assert "Action: 1" in reasoning

    def test_build_prompt_matches_image_only_policy(self):
        adapter = OllamaVLMAdapter(model="test-model")
        prompt = adapter._build_prompt(
            ModelInput(
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                text_prompt="unused mission",
                action_space=ACTION_NAMES,
                step_number=3,
                max_steps=20,
            )
        )
        assert "blue agent from images only" in prompt
        assert "green square goal" in prompt
        assert "Triangle: <direction the triangle is facing>" in prompt
        assert "Goal: <direction of the goal relative to the triangle>" in prompt
        assert "Mission:" not in prompt

    def test_build_messages_uses_previous_and_current_images(self):
        adapter = OllamaVLMAdapter(model="test-model")
        messages = adapter._build_messages(
            ModelInput(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                text_prompt="unused mission",
                action_space=ACTION_NAMES,
                step_number=3,
                max_steps=20,
                additional_context="Recent steps:\nstep 2: action=turn_right, agent_direction=0, agent_position=(1, 1)",
                prior_images=[np.ones((8, 8, 3), dtype=np.uint8)],
            )
        )
        assert len(messages) == 3
        assert messages[1]["content"] == "This is the previous image after the action turn_right was taken."
        assert len(messages[1]["images"]) == 1
        assert messages[2]["content"].startswith("This is the current image.")
        assert len(messages[2]["images"]) == 1


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

    def test_evaluate_task_records_model_outputs_in_trajectory(self, simple_spec):
        class StubModel(ModelInterface):
            @property
            def model_name(self) -> str:
                return "stub"

            def predict(self, input: ModelInput) -> ModelOutput:
                return ModelOutput(
                    action=6,
                    confidence=0.25,
                    reasoning="API error: channel closed",
                    raw_output="channel closed",
                )

        harness = EvaluationHarness(StubModel())
        result = harness.evaluate_task(simple_spec)
        assert result.trajectory
        first_info = result.trajectory[0].info
        assert first_info["model_confidence"] == 0.25
        assert first_info["model_reasoning"] == "API error: channel closed"
        assert first_info["model_raw_output"] == "channel closed"
        assert first_info["model_error"] == "API error: channel closed"
        harness.close()

    def test_history_configuration_controls_model_input(self, simple_spec):
        class RecordingModel(ModelInterface):
            def __init__(self):
                self.inputs = []

            @property
            def model_name(self) -> str:
                return "recorder"

            def predict(self, input: ModelInput) -> ModelOutput:
                self.inputs.append(input)
                return ModelOutput(action=2, confidence=1.0, reasoning="move", raw_output="2")

        model = RecordingModel()
        harness = EvaluationHarness(model, history_images=0, history_text=False)
        try:
            harness.evaluate_task(simple_spec)
        finally:
            harness.close()

        assert model.inputs
        assert all(not inp.prior_images for inp in model.inputs)
        assert all(inp.additional_context is None for inp in model.inputs)

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

    def test_evaluate_task_set_uses_point_scoring(self, simple_spec):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        result = harness.evaluate_task_set([simple_spec], benchmark_name="unit")
        assert isinstance(result, BenchmarkEvaluationResult)
        assert result.benchmark_name == "unit"
        assert result.num_tasks == 1
        assert result.total_available_points > 0
        assert len(result.task_results) == 1
        task_result = result.task_results[0]
        assert task_result.task_id == simple_spec.task_id
        assert task_result.available_points >= task_result.points_earned
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.save(f.name)
            with open(f.name) as fp:
                data = json.load(fp)
            assert data["benchmark_name"] == "unit"
            os.unlink(f.name)
        harness.close()

    def test_evaluate_task_dir_loads_validation_set(self):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        task_dir = str(Path(__file__).resolve().parent.parent / "mazes" / "validation_10")
        result = harness.evaluate_task_dir(task_dir=task_dir, benchmark_name="validation_10")
        assert result.benchmark_name == "validation_10"
        assert result.num_tasks == 10
        assert len(result.task_results) == 10
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
