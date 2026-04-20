"""
Abstract Base Class for Grid Backends

Defines the interface that all grid environment backends must implement.
This allows swapping between MiniGrid (gymnasium) and custom MultiGrid implementations.

BACKEND ABSTRACTION LAYER
=========================

This module provides a pluggable backend system for gridworld environments.
Any grid implementation (MiniGrid, custom MultiGrid with square/hex/triangle tilings,
or future backends) can be used with the same runner and evaluation pipeline.

Architecture:
    TaskSpecification (JSON)
           │
           ▼
    ┌─────────────────────┐
    │  AbstractGridBackend │ ◄── This interface
    └─────────┬───────────┘
         ┌────┴────┐
         ▼         ▼
    ┌─────────┐ ┌─────────────┐
    │MiniGrid │ │ MultiGrid   │
    │Backend  │ │ Backend     │
    │(MVP)    │ │(Custom)     │
    └─────────┘ └─────────────┘

Usage:
    # Option 1: Use MiniGridBackend (gymnasium-based, recommended for MVP)
    from gridworld.backends import MiniGridBackend
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(task_spec)
    obs, state, info = backend.reset(seed=42)
    obs, reward, terminated, truncated, state, info = backend.step(action)

    # Option 2: Use MultiGridBackend (custom tilings: square, hex, triangle)
    from gridworld.backends import MultiGridBackend
    backend = MultiGridBackend(tiling="triangle", render_mode="rgb_array")
    backend.configure(task_spec)
    # ... same interface as above

Implementing a New Backend:
    1. Create a new class that inherits from AbstractGridBackend
    2. Implement all abstract methods (see docstrings below)
    3. The backend must:
       - Accept TaskSpecification objects via configure()
       - Return consistent GridState objects from reset() and step()
       - Provide RGB observations via render()
       - Support the 7-action MiniGrid action space (0-6)

GridState:
    The GridState dataclass provides a backend-agnostic snapshot of environment
    state for evaluation and comparison. All backends must populate this correctly.

Action Space:
    All backends use the standard 7-action discrete space:
    0: turn_left, 1: turn_right, 2: forward, 3: pickup, 4: drop, 5: toggle, 6: done/wait

FEATURE COMPARISON
==================

The two backends have different feature support. Choose based on your needs:

    Feature              | MiniGridBackend | MultiGridBackend
    ---------------------|-----------------|------------------
    Tilings:             |                 |
      Square grid        | ✓               | ✓
      Hexagonal grid     | ✗               | ✓
      Triangle grid      | ✗               | ✓
      3-4-6-4            | ✗               | ✓
      4-8-8              | ✗               | ✓
    Objects:             |                 |
      Walls              | ✓               | ✓
      Movable/Blocks     | ✓               | ✓
      Keys               | ✓               | ✓
      Doors              | ✓               | ✓
      Switches           | ✓               | ✓
      Gates              | ✓               | ✓
      Hazards (Lava)     | ✓               | ✓
      Teleporters        | ✓               | ✓
      Zones (targets)    | ✗               | ✓
    Features:            |                 |
      Partial obs (cone) | ✓               | ✓
      Fog of war         | ✓               | ✓
      Mature/tested      | ✓               | ✗ (newer)

    Recommendation:
    - Use MiniGridBackend for standard square grid tasks (more mature)
    - Use MultiGridBackend for exotic tilings (hex/triangle) or zones

See Also:
    - minigrid_backend.py: MiniGrid (gymnasium) implementation
    - multigrid_backend.py: Custom MultiGrid implementation with exotic tilings
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np

from ..task_spec import TaskSpecification, Position


@dataclass
class GridState:
    """
    Represents the current state of a grid environment.

    This is a backend-agnostic representation of the environment state
    that can be used for evaluation and comparison.
    """
    # Agent state
    agent_position: tuple[int, int]
    agent_direction: int  # 0=right, 1=down, 2=left, 3=up
    agent_carrying: Optional[str] = None  # ID or color of carried object

    # Environment state
    step_count: int = 0
    max_steps: int = 100
    terminated: bool = False
    truncated: bool = False
    reward: float = 0.0

    # Mechanism states
    open_doors: set[str] = field(default_factory=set)  # IDs of open doors
    collected_keys: set[str] = field(default_factory=set)  # IDs of collected keys
    active_switches: set[str] = field(default_factory=set)  # IDs of active switches
    open_gates: set[str] = field(default_factory=set)  # IDs of open gates
    block_positions: dict[str, tuple[int, int]] = field(default_factory=dict)  # block_id -> position
    teleporter_cooldowns: dict[str, int] = field(default_factory=dict)  # teleporter_id -> cooldown

    # Goal state
    goal_reached: bool = False

    # Observability state
    observability_mode: str = "full"  # "full", "view_cone", "fog_of_war"
    visible_cells: set[tuple[int, int]] = field(default_factory=set)  # Currently visible cells
    explored_cells: set[tuple[int, int]] = field(default_factory=set)  # All ever-seen cells (fog_of_war)

    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization."""
        return {
            "agent_position": list(self.agent_position),
            "agent_direction": self.agent_direction,
            "agent_carrying": self.agent_carrying,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reward": self.reward,
            "open_doors": list(self.open_doors),
            "collected_keys": list(self.collected_keys),
            "active_switches": list(self.active_switches),
            "open_gates": list(self.open_gates),
            "block_positions": {k: list(v) for k, v in self.block_positions.items()},
            "teleporter_cooldowns": self.teleporter_cooldowns,
            "goal_reached": self.goal_reached,
            "observability_mode": self.observability_mode,
            "visible_cells": [list(c) for c in self.visible_cells],
            "explored_cells": [list(c) for c in self.explored_cells],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridState":
        """Create state from dictionary."""
        return cls(
            agent_position=tuple(d["agent_position"]),
            agent_direction=d["agent_direction"],
            agent_carrying=d.get("agent_carrying"),
            step_count=d.get("step_count", 0),
            max_steps=d.get("max_steps", 100),
            terminated=d.get("terminated", False),
            truncated=d.get("truncated", False),
            reward=d.get("reward", 0.0),
            open_doors=set(d.get("open_doors", [])),
            collected_keys=set(d.get("collected_keys", [])),
            active_switches=set(d.get("active_switches", [])),
            open_gates=set(d.get("open_gates", [])),
            block_positions={k: tuple(v) for k, v in d.get("block_positions", {}).items()},
            teleporter_cooldowns=d.get("teleporter_cooldowns", {}),
            goal_reached=d.get("goal_reached", False),
            observability_mode=d.get("observability_mode", "full"),
            visible_cells={tuple(c) for c in d.get("visible_cells", [])},
            explored_cells={tuple(c) for c in d.get("explored_cells", [])},
        )


class AbstractGridBackend(ABC):
    """
    Abstract interface for grid environment backends.

    Implementations provide the actual environment logic while
    maintaining a consistent interface for the runner and evaluation.
    """

    def __init__(self):
        self.task_spec: Optional[TaskSpecification] = None
        self._configured = False

    @abstractmethod
    def configure(self, task_spec: TaskSpecification) -> None:
        """
        Configure the backend with a task specification.

        Args:
            task_spec: The task specification defining the puzzle
        """
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            observation: The initial observation (RGB image)
            state: The initial GridState
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        """
        Execute one action in the environment.

        Args:
            action: The action to execute (0-6 for MiniGrid actions)

        Returns:
            observation: The new observation (RGB image)
            reward: The reward for this step
            terminated: Whether the episode ended (goal reached or failed)
            truncated: Whether the episode was cut short (max steps)
            state: The new GridState
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """
        Render the current environment state.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        pass

    @abstractmethod
    def get_mission_text(self) -> str:
        """
        Get the mission/goal description text.

        Returns:
            Human-readable mission description
        """
        pass

    @abstractmethod
    def get_state(self) -> GridState:
        """
        Get the current environment state.

        Returns:
            Current GridState
        """
        pass

    @property
    def is_configured(self) -> bool:
        """Whether the backend has been configured with a task spec."""
        return self._configured

    @property
    def action_space_size(self) -> int:
        """Size of the action space (7 for MiniGrid)."""
        return 7

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of observations (H, W, C)."""
        return (64, 64, 3)  # Default, can be overridden

    def close(self) -> None:
        """Clean up resources."""
        pass
