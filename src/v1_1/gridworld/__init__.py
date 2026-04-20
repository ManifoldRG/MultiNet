"""
MiniGrid/GridWorld Domain for MultiNet v1.1

This module provides a complete gridworld evaluation domain with:
- Task specification schema (JSON) for defining puzzles
- Task parser that creates MiniGrid environments from specs
- Backend abstraction for pluggable grid implementations
- Episode runner for trajectory collection
- Evaluation module following GenESIS patterns
"""

from .bootstrap import disable_gymnasium_env_plugins

disable_gymnasium_env_plugins()

from .task_spec import (
    Position,
    KeySpec,
    DoorSpec,
    SwitchSpec,
    GateSpec,
    BlockSpec,
    HazardSpec,
    TeleporterSpec,
    DependencyStep,
    DependencyChain,
    Distractor,
    MazeLayout,
    MechanismSet,
    Rules,
    GoalSpec,
    TaskSpecification,
)
from .task_parser import TaskParser
from .actions import MiniGridActions, ACTION_NAMES, ACTION_DESCRIPTIONS


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
    "DependencyStep",
    "DependencyChain",
    "Distractor",
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
]
