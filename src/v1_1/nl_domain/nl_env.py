"""
Natural Language GridWorld Environment

Wraps any AbstractGridBackend with a text-based action space.
Accepts NL commands, parses them to discrete actions, and executes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from ..gridworld.backends.base import AbstractGridBackend, GridState
    from ..gridworld.backends.minigrid_backend import MiniGridBackend
    from ..gridworld.task_spec import TaskSpecification
except ImportError:
    from gridworld.backends.base import AbstractGridBackend, GridState
    from gridworld.backends.minigrid_backend import MiniGridBackend
    from gridworld.task_spec import TaskSpecification
from .nl_action_parser import NLActionParser


class NLGridWorldEnv(gym.Env):
    """
    Natural Language GridWorld environment.

    Wraps an AbstractGridBackend and accepts text action commands.
    Parses NL commands to discrete MiniGrid actions and executes them.

    Usage:
        env = NLGridWorldEnv(task_spec)
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step("go forward")
        obs, reward, terminated, truncated, info = env.step("turn left then move forward")
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
    }

    def __init__(
        self,
        task_spec: TaskSpecification,
        backend: Optional[AbstractGridBackend] = None,
        render_mode: str = "rgb_array",
    ):
        super().__init__()

        self.task_spec = task_spec
        self.backend = backend or MiniGridBackend(render_mode=render_mode)
        self.parser = NLActionParser()
        self.render_mode = render_mode

        # Text action space
        self.action_space = spaces.Text(min_length=1, max_length=256)

        # Observation space (RGB image)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )

        # State tracking
        self._state: Optional[GridState] = None
        self._obs: Optional[np.ndarray] = None
        self._nl_history: list[str] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        self.backend.configure(self.task_spec)
        obs, state, info = self.backend.reset(seed=seed)
        self._state = state
        self._obs = obs
        self._nl_history = []

        info["state"] = state
        info["mission"] = self.backend.get_mission_text()
        return obs, info

    def step(self, nl_command: str) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute a natural language command.

        The command is parsed into one or more discrete actions,
        which are executed sequentially. The observation and reward
        from the final action are returned.

        Args:
            nl_command: Natural language action command

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        self._nl_history.append(nl_command)

        # Parse NL command to action sequence
        agent_facing = self._state.agent_direction
        actions = self.parser.parse(nl_command, agent_facing)

        # Execute all parsed actions
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self._obs
        info = {}

        for action in actions:
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, state, info = self.backend.step(action)
            self._state = state
            self._obs = obs
            total_reward += reward

        info["state"] = self._state
        info["parsed_actions"] = actions
        info["nl_history"] = self._nl_history.copy()

        return obs, total_reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render current state."""
        return self.backend.render()

    def get_state(self) -> GridState:
        """Get current grid state."""
        return self._state

    def get_nl_history(self) -> list[str]:
        """Get history of NL commands issued."""
        return self._nl_history.copy()

    def close(self):
        """Clean up resources."""
        self.backend.close()
