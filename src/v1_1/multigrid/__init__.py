# multigrid/__init__.py

"""
MultiGrid: Topology-Agnostic Gridworld Environments

Provides gridworld environments with pluggable tiling systems:
- Square: Traditional 4-connected grid (up/down/left/right)
- Hexagonal: 6-connected pointy-top hexagons
- Triangle: 3-connected triangles within hexagons

Usage:
    from multigrid.env import MultiGridEnv, TilingRegistry

    # Create environment with triangle tiling
    env = MultiGridEnv(task_spec=spec, tiling="triangle")
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from .core import Cell, TilingGraph
from .base import Tiling
from .tilings import SquareTiling, HexTiling, TriangleTiling
from .env import MultiGridEnv, TilingRegistry
from .agent import AgentState, Action
from .world import WorldState, execute_action
from .goals import (
    Goal,
    ReachPositionGoal,
    ReachCanonicalPositionGoal,
    CollectAllGoal,
    PushBlockToGoal,
    SurviveStepsGoal,
    CompositeGoal,
    AnyGoal,
    create_goal_from_spec,
)
from .rendering import render_multigrid, MinimalRenderer

__all__ = [
    # Core
    'Cell',
    'TilingGraph',
    'Tiling',
    # Tilings
    'SquareTiling',
    'HexTiling',
    'TriangleTiling',
    # Environment
    'MultiGridEnv',
    'TilingRegistry',
    # Agent
    'AgentState',
    'Action',
    # World
    'WorldState',
    'execute_action',
    # Goals
    'Goal',
    'ReachPositionGoal',
    'ReachCanonicalPositionGoal',
    'CollectAllGoal',
    'PushBlockToGoal',
    'SurviveStepsGoal',
    'CompositeGoal',
    'AnyGoal',
    'create_goal_from_spec',
    # Rendering
    'render_multigrid',
    'MinimalRenderer',
]
