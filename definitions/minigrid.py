"""
MiniGrid Definitions for GenESIS Framework

Provides environment descriptions, action spaces, and other metadata
for the MiniGrid/GridWorld evaluation domain.
"""

import numpy as np


class MiniGridDefinitions:
    """
    Definitions for MiniGrid gridworld environments.

    Follows the same structure as ProcGenDefinitions for consistency
    with the GenESIS evaluation framework.
    """

    # Environment descriptions by tier
    DESCRIPTIONS = {
        # Tier 1: Pure Navigation
        "tier1": {
            "navigate to the goal": [
                "Navigate through the grid to reach the goal position.",
                "Avoid obstacles and find the shortest path.",
                "The green square marks the goal location.",
            ]
        },
        "tier1_maze_simple": {
            "navigate to the goal": [
                "Navigate through an empty room to reach the goal.",
                "The green square marks the goal location.",
            ]
        },
        "tier1_maze_corridor": {
            "navigate through corridor to goal": [
                "Navigate through a corridor with walls.",
                "Find a path around obstacles to reach the goal.",
                "The green square marks the goal location.",
            ]
        },
        "tier1_maze_rooms": {
            "navigate through rooms to goal": [
                "Navigate through connected rooms.",
                "Pass through doorways to reach the goal.",
                "The green square marks the goal location.",
            ]
        },

        # Tier 2: Linear Dependencies (Keys + Doors)
        "tier2": {
            "collect key and unlock door": [
                "Collect the key to unlock the matching colored door.",
                "Navigate to the goal after opening the door.",
                "Match key colors to door colors.",
            ]
        },
        "tier2_single_key": {
            "collect key to unlock door": [
                "Find and collect the key.",
                "Use the key to unlock the matching door.",
                "Navigate through the door to reach the goal.",
            ]
        },
        "tier2_multi_key": {
            "collect keys in order": [
                "Multiple keys and doors block your path.",
                "Collect keys in the correct order to progress.",
                "Each key unlocks a door of the same color.",
            ]
        },
        "tier2_colored_doors": {
            "match keys to colored doors": [
                "Multiple colored keys and doors.",
                "Match each key to its corresponding door color.",
                "Navigate through unlocked doors to reach the goal.",
            ]
        },

        # Tier 3: Multi-Mechanism (Keys + Doors + Switches + Gates)
        "tier3": {
            "use keys switches and gates": [
                "Combine key collection with switch activation.",
                "Switches control gates that block passages.",
                "Keys unlock doors, switches open gates.",
            ]
        },
        "tier3_key_switch": {
            "use key then switch": [
                "First collect the key to unlock the door.",
                "Then activate the switch to open the gate.",
                "Navigate to the goal through opened passages.",
            ]
        },
        "tier3_gates_switches": {
            "activate switches to open gates": [
                "Multiple switches control multiple gates.",
                "Activate switches in the correct order.",
                "Navigate through opened gates to the goal.",
            ]
        },
        "tier3_complex_deps": {
            "complex mechanism dependencies": [
                "Keys, doors, switches, and gates interact.",
                "Solve the dependency chain to reach the goal.",
                "Some mechanisms may need to be activated in order.",
            ]
        },

        # Tier 4: Irreversibility (Pushable blocks, consumables)
        "tier4": {
            "push blocks and use resources wisely": [
                "Some actions cannot be undone.",
                "Pushing blocks into corners may block progress.",
                "Keys are consumed when used on doors.",
            ]
        },
        "tier4_push_block": {
            "push block to clear path": [
                "Push the block out of the way.",
                "Be careful - blocks can only be pushed, not pulled.",
                "Plan your moves to avoid getting stuck.",
            ]
        },
        "tier4_blocked_path": {
            "push blocks strategically": [
                "Multiple blocks need to be moved.",
                "Wrong moves may permanently block paths.",
                "Think ahead before pushing.",
            ]
        },
        "tier4_consumable": {
            "use limited resources wisely": [
                "Keys are consumed when used.",
                "Choose which doors to open carefully.",
                "You may not have enough keys for all doors.",
            ]
        },

        # Tier 5: Hidden Information
        "tier5": {
            "discover hidden rules": [
                "Some mechanisms have hidden effects.",
                "Experiment to discover how things work.",
                "Information must be inferred from observation.",
            ]
        },
        "tier5_hidden_switch": {
            "find the hidden switch effect": [
                "A switch controls a gate, but the connection is hidden.",
                "Try interacting to discover what controls what.",
                "Use trial and error to find the solution.",
            ]
        },
        "tier5_infer_color": {
            "infer the correct key color": [
                "The door's required key color is not visible.",
                "Try different keys to find which one works.",
                "Only one key will open the door.",
            ]
        },
        "tier5_memory": {
            "remember visited locations": [
                "Partial observability limits your view.",
                "Remember where you've been and what you've seen.",
                "Use memory to navigate efficiently.",
            ]
        },

        # Default fallback
        "default": {
            "default": [
                "Navigate the gridworld environment.",
                "Use available actions to reach your goal.",
                "Interact with objects as needed.",
            ]
        },
    }

    # Action space definitions (7 discrete actions)
    movement_actions = {
        0: "Turn left (rotate 90° counter-clockwise)",
        1: "Turn right (rotate 90° clockwise)",
        2: "Move forward (one cell in facing direction)",
    }

    interaction_actions = {
        3: "Pick up (grab object directly in front)",
        4: "Drop (release currently held object)",
        5: "Toggle (interact with door, switch, or object in front)",
        6: "Done/Wait (no operation, stay in place)",
    }

    ACTION_SPACES = {
        # Tier 1: Navigation only
        "tier1": {
            "default": {
                0: ("Movement action", movement_actions),
            }
        },
        # Tier 2+: Full action space
        "default": {
            "default": {
                0: ("Action", {**movement_actions, **interaction_actions}),
            }
        },
        "tier2": {
            "default": {
                0: ("Action", {**movement_actions, **interaction_actions}),
            }
        },
        "tier3": {
            "default": {
                0: ("Action", {**movement_actions, **interaction_actions}),
            }
        },
        "tier4": {
            "default": {
                0: ("Action", {**movement_actions, **interaction_actions}),
            }
        },
        "tier5": {
            "default": {
                0: ("Action", {**movement_actions, **interaction_actions}),
            }
        },
    }

    ACTION_EXCLUSIVENESS = {
        "default": {
            "default": True  # Only one action at a time
        }
    }

    ADDITIONAL_INSTRUCTIONS = {
        "tier1": {
            "default": "Focus on navigation - use turn_left, turn_right, and move_forward to reach the green goal square."
        },
        "tier2": {
            "default": "Collect keys (pickup action when facing key) and use them on matching colored doors (toggle action when facing door)."
        },
        "tier3": {
            "default": "Use toggle action on switches to open gates. Combine with key/door mechanics to reach the goal."
        },
        "tier4": {
            "default": "Be careful with irreversible actions. Pushing blocks into walls cannot be undone. Keys are consumed when used."
        },
        "tier5": {
            "default": "Some information is hidden. Experiment with interactions to discover how mechanisms work."
        },
        "default": {
            "default": None
        }
    }

    ACTION_DECODE_STRATEGIES = {
        "default": "single_discrete"
    }

    @staticmethod
    def get_valid_action_space(tier: int = 2) -> list[int]:
        """
        Get the valid action IDs for a given difficulty tier.

        Args:
            tier: Difficulty tier (1-5)

        Returns:
            List of valid action IDs
        """
        if tier == 1:
            # Navigation only
            return [0, 1, 2, 6]  # turn_left, turn_right, forward, wait
        else:
            # Full action space
            return list(range(7))

    @staticmethod
    def get_action_description(action_id: int) -> str:
        """
        Get human-readable description for an action.

        Args:
            action_id: Action ID (0-6)

        Returns:
            Action description string
        """
        all_actions = {
            **MiniGridDefinitions.movement_actions,
            **MiniGridDefinitions.interaction_actions
        }
        return all_actions.get(action_id, f"Unknown action {action_id}")

    @staticmethod
    def clip_action_to_valid(action: int, tier: int = 2) -> int:
        """
        Clip an action to the valid action space for a tier.

        Args:
            action: The predicted action
            tier: Difficulty tier

        Returns:
            Valid action ID (defaults to wait/done if invalid)
        """
        valid_actions = MiniGridDefinitions.get_valid_action_space(tier)
        if action in valid_actions:
            return action
        # Default to wait action
        return 6
