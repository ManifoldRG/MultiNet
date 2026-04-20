"""Tests for VLM vision sanity check module.

Tests question generation and answer checking logic without requiring a VLM.
Uses a mock ask function to simulate VLM responses.
"""

import pytest
import sys
import os
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from gridworld.task_spec import TaskSpecification
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.base import GridState
from vlm_sanity_check import (
    generate_questions_for_task,
    check_answer,
    run_sanity_check,
    VisionQuestion,
)


# --- Answer checking ---

class TestCheckAnswer:
    """Test the keyword matching logic."""

    def test_exact_match(self):
        passed, matched = check_answer("I see a blue triangle", ["blue", "triangle"])
        assert passed
        assert "blue" in matched
        assert "triangle" in matched

    def test_case_insensitive(self):
        passed, matched = check_answer("BLUE TRIANGLE", ["blue", "triangle"])
        assert passed

    def test_partial_match_passes(self):
        """At least one keyword match should pass."""
        passed, matched = check_answer("I see something green", ["green", "square"])
        assert passed
        assert "green" in matched

    def test_no_match_fails(self):
        passed, matched = check_answer("I see nothing interesting", ["blue", "triangle"])
        assert not passed
        assert len(matched) == 0

    def test_empty_answer(self):
        passed, matched = check_answer("", ["blue"])
        assert not passed

    def test_keyword_in_longer_word(self):
        """Keywords can match as substrings."""
        passed, matched = check_answer("The triangle-shaped agent is blue", ["triangle"])
        assert passed


# --- Question generation ---

class TestGenerateQuestions:
    """Test question generation for different task types."""

    @pytest.fixture
    def simple_maze_spec(self):
        return TaskSpecification.from_dict({
            "task_id": "test_simple",
            "seed": 42,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [8, 8],
                "walls": [[4, 1], [4, 2], [4, 3]],
                "start": [1, 1],
                "goal": [6, 6],
            },
            "mechanisms": {"keys": [], "doors": [], "switches": [],
                           "gates": [], "blocks": [], "teleporters": [], "hazards": []},
            "rules": {"key_consumption": True, "switch_type": "toggle"},
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })

    @pytest.fixture
    def complex_spec(self):
        return TaskSpecification.from_dict({
            "task_id": "test_complex",
            "seed": 42,
            "difficulty_tier": 3,
            "maze": {
                "dimensions": [10, 10],
                "walls": [[5, 1], [5, 2]],
                "start": [1, 1],
                "goal": [8, 8],
            },
            "mechanisms": {
                "keys": [{"id": "k1", "position": [2, 3], "color": "blue"}],
                "doors": [{"id": "d1", "position": [5, 3], "requires_key": "blue"}],
                "switches": [{"id": "s1", "position": [3, 5], "controls": ["g1"]}],
                "gates": [{"id": "g1", "position": [5, 5]}],
                "blocks": [],
                "teleporters": [],
                "hazards": [{"id": "h1", "position": [7, 7], "hazard_type": "lava"}],
            },
            "rules": {"key_consumption": True, "switch_type": "toggle"},
            "goal": {"type": "reach_position", "target": [8, 8]},
            "max_steps": 100,
        })

    def test_simple_maze_questions(self, simple_maze_spec):
        """Simple maze should generate agent, goal, wall, and spatial questions."""
        state = GridState(agent_position=(1, 1), agent_direction=0)
        questions = generate_questions_for_task(simple_maze_spec, state)

        categories = [q.category for q in questions]
        assert "object_id" in categories
        assert "spatial" in categories

        # Should have at least: agent, goal, wall identification + spatial questions
        assert len(questions) >= 5

    def test_complex_task_has_more_questions(self, complex_spec):
        """Tasks with more mechanisms should generate more questions."""
        state = GridState(agent_position=(1, 1), agent_direction=0)
        questions = generate_questions_for_task(complex_spec, state)

        # Should have key, door, switch, hazard questions in addition to basics
        q_texts = " ".join(q.question.lower() for q in questions)
        assert "key" in q_texts
        assert "door" in q_texts
        assert "switch" in q_texts or "button" in q_texts
        assert "hazard" in q_texts or "lava" in q_texts

    def test_spatial_direction_question(self, simple_maze_spec):
        """Should ask about agent direction."""
        state = GridState(agent_position=(1, 1), agent_direction=0)  # facing right
        questions = generate_questions_for_task(simple_maze_spec, state)

        dir_questions = [q for q in questions if "direction" in q.question.lower() or "facing" in q.question.lower()]
        assert len(dir_questions) > 0
        # Agent faces right (dir=0), so expected keyword should be "right"
        assert "right" in dir_questions[0].expected_keywords

    def test_goal_relative_position(self, simple_maze_spec):
        """Should ask where goal is relative to agent."""
        # Agent at (1,1), goal at (6,6) → goal is below and to the right
        state = GridState(agent_position=(1, 1), agent_direction=0)
        questions = generate_questions_for_task(simple_maze_spec, state)

        rel_questions = [q for q in questions if "relative" in q.question.lower()]
        assert len(rel_questions) > 0
        # Goal is at (6,6), agent at (1,1) → right (x: 6>1) and below (y: 6>1)
        assert "right" in rel_questions[0].expected_keywords
        assert "below" in rel_questions[0].expected_keywords

    def test_no_key_question_without_keys(self, simple_maze_spec):
        """Simple maze with no keys should NOT generate key questions."""
        state = GridState(agent_position=(1, 1), agent_direction=0)
        questions = generate_questions_for_task(simple_maze_spec, state)

        key_questions = [q for q in questions if "key" in q.question.lower()]
        assert len(key_questions) == 0


# --- Mock VLM sanity check ---

class TestMockSanityCheck:
    """Test the full sanity check pipeline with mock VLM responses."""

    def test_perfect_mock_vlm(self):
        """A mock VLM that always answers correctly should get 100%."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier1" / "maze_simple_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")

        def mock_ask(image, question):
            # Return an answer that matches common keywords
            return (
                "I see a blue triangle agent facing right on a grid. "
                "There is a green goal square. There are grey walls. "
                "The grid appears to be about 8x8. "
                "The goal is below and to the right of the agent."
            )

        report = run_sanity_check(str(task_path), mock_ask, "mock_perfect", verbose=False)
        assert report.passed > 0
        assert report.object_id_score > 0

    def test_blind_mock_vlm(self):
        """A mock VLM that returns garbage should score poorly."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier1" / "maze_simple_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")

        def mock_ask(image, question):
            return "I cannot process this image."

        report = run_sanity_check(str(task_path), mock_ask, "mock_blind", verbose=False)
        assert report.failed > 0
        assert report.object_id_score < 1.0

    def test_error_handling_mock(self):
        """VLM errors should be captured gracefully."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier1" / "maze_simple_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")

        def mock_ask(image, question):
            raise ConnectionError("VLM server not available")

        report = run_sanity_check(str(task_path), mock_ask, "mock_error", verbose=False)
        # All should fail with errors
        assert report.failed == report.total_questions
        for r in report.results:
            assert r.error is not None

    def test_report_serialization(self):
        """Report should serialize to dict cleanly."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier1" / "maze_simple_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")

        def mock_ask(image, question):
            return "Blue triangle agent on a grid with green goal."

        report = run_sanity_check(str(task_path), mock_ask, "mock", verbose=False)
        d = report.to_dict()
        assert "model_name" in d
        assert "task_id" in d
        assert "results" in d
        assert isinstance(d["results"], list)

    def test_image_passed_to_vlm(self):
        """The ask function should receive a valid RGB image."""
        task_path = Path(_v1_1_dir) / "gridworld" / "tasks" / "tier2" / "single_key_001.json"
        if not task_path.exists():
            pytest.skip("Task file not found")

        received_images = []

        def mock_ask(image, question):
            received_images.append(image)
            return "blue triangle green goal red key"

        report = run_sanity_check(str(task_path), mock_ask, "mock", verbose=False)

        # All questions should have received the same image
        assert len(received_images) == report.total_questions
        for img in received_images:
            assert img.ndim == 3
            assert img.shape[2] == 3  # RGB
            assert img.dtype.name == "uint8"
            assert img.max() > 0  # Not blank
