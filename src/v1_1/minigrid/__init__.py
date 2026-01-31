"""
MiniGrid/GridWorld Domain for MultiNet v1.1

This module provides a complete gridworld evaluation domain with:
- Task specification schema (JSON) for defining puzzles
- Task parser that creates MiniGrid environments from specs
- Backend abstraction for pluggable grid implementations
- Episode runner for trajectory collection
- Evaluation module following GenESIS patterns
"""

from .task_spec import (
    Position,
    KeySpec,
    DoorSpec,
    SwitchSpec,
    GateSpec,
    BlockSpec,
    HazardSpec,
    TeleporterSpec,
    MazeLayout,
    MechanismSet,
    Rules,
    GoalSpec,
    TaskSpecification,
)
from .task_parser import TaskParser
from .actions import MiniGridActions, ACTION_NAMES, ACTION_DESCRIPTIONS


def register_minigrid_envs():
    """
    Stub function for gymnasium plugin system compatibility.

    This local minigrid module is not the official MiniGrid package,
    but gymnasium tries to load this function from any installed 'minigrid' module.
    """
    pass


__all__ = [
    # Task specification
    "Position",
    "KeySpec",
    "DoorSpec",
    "SwitchSpec",
    "GateSpec",
    "BlockSpec",
    "HazardSpec",
    "TeleporterSpec",
    "MazeLayout",
    "MechanismSet",
    "Rules",
    "GoalSpec",
    "TaskSpecification",
    # Parser
    "TaskParser",
    # Actions
    "MiniGridActions",
    "ACTION_NAMES",
    "ACTION_DESCRIPTIONS",
    # Gymnasium compatibility
    "register_minigrid_envs",
]
