"""
Natural Language Action Parser

Converts natural language commands to MiniGrid action IDs.
Uses keyword-based pattern matching with directional decomposition.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    from ..gridworld.actions import MiniGridActions
except ImportError:
    from gridworld.actions import MiniGridActions


# Direction to agent-relative action mappings
# These map compass directions to required facing direction (0=right, 1=down, 2=left, 3=up)
COMPASS_TO_FACING = {
    "north": 3, "up": 3,
    "south": 1, "down": 1,
    "east": 0, "right": 0,
    "west": 2, "left": 2,
}

# Patterns mapped to action IDs, ordered by specificity (most specific first)
ACTION_PATTERNS: list[tuple[re.Pattern, int]] = [
    # Movement
    (re.compile(r"\b(go|move|walk|step)\s+(forward|ahead|straight)\b", re.I), MiniGridActions.MOVE_FORWARD),
    (re.compile(r"\bforward\b", re.I), MiniGridActions.MOVE_FORWARD),
    (re.compile(r"\badvance\b", re.I), MiniGridActions.MOVE_FORWARD),

    # Turning
    (re.compile(r"\bturn\s+left\b", re.I), MiniGridActions.TURN_LEFT),
    (re.compile(r"\bturn\s+right\b", re.I), MiniGridActions.TURN_RIGHT),
    (re.compile(r"\brotate\s+left\b", re.I), MiniGridActions.TURN_LEFT),
    (re.compile(r"\brotate\s+right\b", re.I), MiniGridActions.TURN_RIGHT),
    (re.compile(r"\bleft\b", re.I), MiniGridActions.TURN_LEFT),
    (re.compile(r"\bright\b", re.I), MiniGridActions.TURN_RIGHT),

    # Interaction
    (re.compile(r"\bpick\s*up\b", re.I), MiniGridActions.PICKUP),
    (re.compile(r"\bgrab\b", re.I), MiniGridActions.PICKUP),
    (re.compile(r"\bcollect\b", re.I), MiniGridActions.PICKUP),
    (re.compile(r"\btake\b", re.I), MiniGridActions.PICKUP),

    (re.compile(r"\bdrop\b", re.I), MiniGridActions.DROP),
    (re.compile(r"\bput\s+down\b", re.I), MiniGridActions.DROP),
    (re.compile(r"\brelease\b", re.I), MiniGridActions.DROP),

    (re.compile(r"\btoggle\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\bopen\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\bclose\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\bpress\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\bactivate\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\bswitch\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\bunlock\b", re.I), MiniGridActions.TOGGLE),
    (re.compile(r"\binteract\b", re.I), MiniGridActions.TOGGLE),

    # Wait/done
    (re.compile(r"\bwait\b", re.I), MiniGridActions.DONE),
    (re.compile(r"\bstay\b", re.I), MiniGridActions.DONE),
    (re.compile(r"\bdo\s+nothing\b", re.I), MiniGridActions.DONE),
    (re.compile(r"\bdone\b", re.I), MiniGridActions.DONE),
    (re.compile(r"\bstop\b", re.I), MiniGridActions.DONE),

    # Push (mapped to forward, since pushing is implicit on forward into block)
    (re.compile(r"\bpush\b", re.I), MiniGridActions.MOVE_FORWARD),
]

# Compass direction patterns
COMPASS_PATTERN = re.compile(
    r"\b(?:go|move|walk|head)\s+(north|south|east|west|up|down|left|right)\b", re.I
)


class NLActionParser:
    """
    Parse natural language commands into MiniGrid action sequences.

    Supports:
    - Simple commands: "go forward", "turn left", "pick up", "toggle"
    - Directional: "move north" -> decomposed to turn sequence + forward
    - Compound: Multiple commands in one string (separated by "then" or commas)
    """

    def parse(self, command: str, agent_facing: int = 0) -> list[int]:
        """
        Parse a natural language command into a sequence of action IDs.

        Args:
            command: Natural language command string
            agent_facing: Current agent facing direction (0=right, 1=down, 2=left, 3=up)

        Returns:
            List of action IDs (usually length 1, compound commands may be longer)
        """
        command = command.strip()
        if not command:
            return [MiniGridActions.DONE]

        # Check for compound commands (split by "then", "and then", commas)
        parts = re.split(r"\bthen\b|,\s*(?:and\s+)?", command, flags=re.I)
        parts = [p.strip() for p in parts if p.strip()]

        actions = []
        for part in parts:
            parsed = self._parse_single(part, agent_facing)
            actions.extend(parsed)
            # Update facing after turns for compound commands
            for a in parsed:
                if a == MiniGridActions.TURN_LEFT:
                    agent_facing = (agent_facing + 3) % 4  # -1 mod 4
                elif a == MiniGridActions.TURN_RIGHT:
                    agent_facing = (agent_facing + 1) % 4

        return actions if actions else [MiniGridActions.DONE]

    def _parse_single(self, command: str, agent_facing: int) -> list[int]:
        """Parse a single (non-compound) command."""
        # Check for compass directions first
        compass_match = COMPASS_PATTERN.search(command)
        if compass_match:
            direction = compass_match.group(1).lower()
            target_facing = COMPASS_TO_FACING.get(direction)
            if target_facing is not None:
                return self._turn_sequence(agent_facing, target_facing) + [MiniGridActions.MOVE_FORWARD]

        # Try pattern matching
        for pattern, action_id in ACTION_PATTERNS:
            if pattern.search(command):
                return [action_id]

        # Could not parse - return wait
        return [MiniGridActions.DONE]

    def _turn_sequence(self, current_facing: int, target_facing: int) -> list[int]:
        """
        Generate turn sequence to change from current to target facing.

        Chooses the shortest rotation direction.
        """
        if current_facing == target_facing:
            return []

        # Calculate clockwise and counterclockwise distances
        cw_dist = (target_facing - current_facing) % 4
        ccw_dist = (current_facing - target_facing) % 4

        if cw_dist <= ccw_dist:
            return [MiniGridActions.TURN_RIGHT] * cw_dist
        else:
            return [MiniGridActions.TURN_LEFT] * ccw_dist
