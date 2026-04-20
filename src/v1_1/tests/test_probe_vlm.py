"""Tests for the lightweight VLM probe CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from probe_vlm import collect_probe_context, parse_action_sequence, save_probe_images


def test_parse_action_sequence_accepts_empty():
    assert parse_action_sequence("") == []
    assert parse_action_sequence(None) == []


def test_parse_action_sequence_parses_csv():
    assert parse_action_sequence("1, 2,6") == [1, 2, 6]


def test_parse_action_sequence_rejects_invalid_action():
    try:
        parse_action_sequence("7")
    except ValueError as exc:
        assert "Invalid action id" in str(exc)
    else:
        raise AssertionError("Expected invalid action sequence to raise ValueError")


def test_collect_probe_context_tracks_history():
    task_path = Path(__file__).resolve().parent.parent / "mazes" / "validation_10" / "V01_empty_room.json"
    context = collect_probe_context(
        task_path=str(task_path),
        actions=[1, 1],
        history_images=2,
        include_text_history=True,
    )

    assert context.task_id
    assert context.current_image.ndim == 3
    assert len(context.prior_images) == 2
    assert context.action_names == ["turn_right", "turn_right"]
    assert context.current_direction_name == "left"
    assert context.text_memory is not None
    assert "step 1" in context.text_memory
    assert "action=turn_right" in context.text_memory


def test_collect_probe_context_limits_history_length():
    task_path = Path(__file__).resolve().parent.parent / "mazes" / "validation_10" / "V01_empty_room.json"
    context = collect_probe_context(
        task_path=str(task_path),
        actions=[1, 1, 1],
        history_images=1,
        include_text_history=False,
    )

    assert len(context.prior_images) == 1
    assert context.text_memory is None


def test_save_probe_images_writes_current_and_prior(tmp_path):
    task_path = Path(__file__).resolve().parent.parent / "mazes" / "validation_10" / "V01_empty_room.json"
    context = collect_probe_context(
        task_path=str(task_path),
        actions=[1],
        history_images=1,
        include_text_history=False,
    )

    save_probe_images(context, str(tmp_path))

    assert (tmp_path / "current.png").exists()
    assert (tmp_path / "prior_1.png").exists()
