"""Tests for the manual chat smoke test helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_v1_1_dir = str(Path(__file__).resolve().parent.parent)
if _v1_1_dir not in sys.path:
    sys.path.insert(0, _v1_1_dir)

from chat_smoke_test import LOOK_TOKEN, build_prompt, parse_model_reply


def test_parse_model_reply_accepts_names_and_look():
    parsed = parse_model_reply(
        "move_forward\nturn_right\nLOOK",
        max_actions=3,
        allow_look=True,
    )
    assert parsed.actions == [2, 1]
    assert parsed.requested_look is True


def test_parse_model_reply_accepts_numbered_lines():
    parsed = parse_model_reply(
        "1. 2 - move forward\n2. 6",
        max_actions=2,
        allow_look=False,
    )
    assert parsed.actions == [2, 6]
    assert parsed.requested_look is False


def test_parse_model_reply_accepts_bare_numeric_reply():
    parsed = parse_model_reply("6", max_actions=1, allow_look=False)
    assert parsed.actions == [6]
    assert parsed.requested_look is False


def test_parse_model_reply_unlimited_when_budget_is_zero():
    parsed = parse_model_reply("2\n2\n2", max_actions=0, allow_look=False)
    assert parsed.actions == [2, 2, 2]


def test_parse_model_reply_rejects_unparseable_reply():
    try:
        parse_model_reply("I would probably go to the goal.", max_actions=1, allow_look=False)
    except ValueError as exc:
        assert "Could not parse any action" in str(exc)
    else:
        raise AssertionError("Expected parse failure")


def test_build_prompt_mentions_look_when_enabled():
    prompt = build_prompt(
        step_number=3,
        max_steps=20,
        action_budget=0,
        allow_look=True,
        text_history="step 1: action=turn_right",
        prior_image_count=2,
    )
    assert LOOK_TOKEN in prompt
    assert "Recent action history" in prompt
    assert "There are 2 earlier frame(s)" in prompt
    assert "Reply with as many actions as you want" in prompt
    assert "token efficiency" in prompt
    assert "agent position estimate" not in prompt
