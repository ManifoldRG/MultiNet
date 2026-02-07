"""
MiniGrid Action Space Definitions

Standard 7-action discrete space matching MiniGrid's default Actions enum.
"""

from enum import IntEnum
from typing import Dict


class MiniGridActions(IntEnum):
    """MiniGrid action space (7 discrete actions)."""
    TURN_LEFT = 0
    TURN_RIGHT = 1
    MOVE_FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5  # Interact: open door, press switch, etc.
    DONE = 6  # No-op / wait


# Human-readable action names
ACTION_NAMES: Dict[int, str] = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

# Detailed action descriptions for VLM prompts
ACTION_DESCRIPTIONS: Dict[int, str] = {
    0: "Turn left (rotate 90° counter-clockwise)",
    1: "Turn right (rotate 90° clockwise)",
    2: "Move forward (one cell in facing direction)",
    3: "Pick up (grab object in front of agent)",
    4: "Drop (release held object)",
    5: "Toggle (interact with object in front: open/close door, press switch)",
    6: "Done/Wait (no action, stay in place)",
}

# Short descriptions for compact formats
ACTION_SHORT: Dict[int, str] = {
    0: "Left",
    1: "Right",
    2: "Forward",
    3: "Pickup",
    4: "Drop",
    5: "Toggle",
    6: "Wait",
}

# Action space as dict for GenESIS format
ACTION_SPACE_DICT: Dict[int, tuple] = {
    0: ("Turn left", {0: "Rotate 90° counter-clockwise"}),
    1: ("Turn right", {1: "Rotate 90° clockwise"}),
    2: ("Move forward", {2: "Move one cell in facing direction"}),
    3: ("Pick up", {3: "Grab object directly in front"}),
    4: ("Drop", {4: "Release currently held object"}),
    5: ("Toggle/Interact", {5: "Interact with door, switch, or object in front"}),
    6: ("Done/Wait", {6: "No operation, stay in place"}),
}

# Navigation-only subset (Tier 1)
NAVIGATION_ACTIONS = {
    MiniGridActions.TURN_LEFT,
    MiniGridActions.TURN_RIGHT,
    MiniGridActions.MOVE_FORWARD,
    MiniGridActions.DONE,
}

# Full action set (Tiers 2+)
FULL_ACTIONS = set(MiniGridActions)


def action_to_name(action: int) -> str:
    """Convert action ID to human-readable name."""
    return ACTION_NAMES.get(action, f"unknown_{action}")


def name_to_action(name: str) -> int:
    """Convert action name to ID."""
    name_lower = name.lower().strip()
    for action_id, action_name in ACTION_NAMES.items():
        if action_name == name_lower:
            return action_id
    # Try partial matching
    for action_id, action_name in ACTION_NAMES.items():
        if name_lower in action_name or action_name in name_lower:
            return action_id
    raise ValueError(f"Unknown action name: {name}")


def get_valid_actions(tier: int) -> set[int]:
    """Get valid actions for a given difficulty tier."""
    if tier == 1:
        # Navigation only - no pickup, drop, or toggle needed
        return NAVIGATION_ACTIONS
    else:
        # Full action space for tiers 2+
        return FULL_ACTIONS


def format_action_space_for_prompt(tier: int = 2) -> str:
    """Format action space description for VLM prompts."""
    valid_actions = get_valid_actions(tier)
    lines = []
    for action_id in sorted(valid_actions):
        lines.append(f"  {action_id}: {ACTION_DESCRIPTIONS[action_id]}")
    return "\n".join(lines)
