# multigrid/env.py

import json
import numpy as np
from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
from .agent import Action
from .world import WorldState, execute_action
from .base import Tiling
from .tilings import SquareTiling, HexTiling, TriangleTiling
from .rendering import render_multigrid


class TilingRegistry:
    """Registry for tiling types."""
    _types = {
        "square": SquareTiling,
        "hex": HexTiling,
        "triangle": TriangleTiling
    }

    @classmethod
    def get(cls, name: str) -> Tiling:
        """Get tiling instance by name."""
        if name not in cls._types:
            raise ValueError(f"Unknown tiling type: {name}")
        return cls._types[name]()


class MultiGridEnv(gym.Env):
    """
    MultiGrid environment with arbitrary tiling support.

    Fully compatible with gymnasium.Env for RL library compatibility.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "state_dict"],
        "render_fps": 10,
    }

    def __init__(
        self,
        task_spec: Union[dict, str],           # Task spec dict or path to JSON
        tiling: Union[str, Tiling] = "square", # Tiling type or instance
        render_mode: Optional[str] = None,
        render_style: str = "minimal",         # "minimal" or "sprite"
        partial_obs: bool = False,             # Partial observability
        obs_radius: int = 3,                   # Vision radius if partial_obs
    ):
        super().__init__()

        # Load task spec
        if isinstance(task_spec, str):
            with open(task_spec) as f:
                task_spec = json.load(f)
        self.task_spec = task_spec

        # Initialize tiling
        if isinstance(tiling, str):
            self.tiling = TilingRegistry.get(tiling)
        else:
            self.tiling = tiling

        self.render_mode = render_mode
        self.render_style = render_style
        self.partial_obs = partial_obs
        self.obs_radius = obs_radius

        # Define Gymnasium action space
        self.action_space = spaces.Discrete(len(Action))

        # Define Gymnasium observation space (RGB image)
        # Simplified: 64x64 RGB for now
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(64, 64, 3),
            dtype=np.uint8
        )

        # State tracking
        self.state: Optional[WorldState] = None
        self.steps: int = 0
        self.renderer = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        # Use task spec seed if not overridden
        actual_seed = seed if seed is not None else self.task_spec.get("seed", 0)

        # Generate world from task spec
        self.state = WorldState.from_task_spec(
            self.task_spec,
            self.tiling,
            seed=actual_seed
        )
        self.steps = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        assert self.state is not None, "Call reset() before step()"

        # Execute action
        self.state, done, action_info = execute_action(
            self.state,
            Action(action),
            self.tiling
        )
        self.steps += 1

        # Compute reward
        reward = self._compute_reward(done, action_info)

        # Check termination conditions
        terminated = done  # Goal achieved
        truncated = self.steps >= self.task_spec["limits"]["max_steps"]

        obs = self._get_obs()
        info = self._get_info()
        info.update(action_info)

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()
            return None
        elif self.render_mode == "state_dict":
            return self.get_state_dict()

    def get_state_dict(self) -> dict:
        """Export full state as structured dict for cross-domain verification."""
        return {
            "agent": {
                "cell_id": self.state.agent.cell_id,
                "facing": self.state.agent.facing,
                "facing_direction": self.state.agent.get_facing_direction(self.tiling),
                "holding": self.state.agent.holding.id if self.state.agent.holding else None,
                "position_canonical": self.tiling.cell_to_canonical(self.state.agent.cell_id)
            },
            "objects": {
                obj.id: {
                    "type": obj.obj_type,
                    "cell_id": obj.cell_id,
                    "position_canonical": self.tiling.cell_to_canonical(obj.cell_id) if obj.cell_id else None,
                    "color": obj.color
                }
                for obj in self.state.objects.values()
            },
            "step": self.steps,
            "goal_achieved": self.state.check_goal()
        }

    def _get_obs(self) -> np.ndarray:
        """Get observation based on observability mode."""
        if self.state is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        # Get goal cell ID for rendering if goal is position-based
        goal_cell_id = None
        if self.state.goal is not None:
            # Check if goal has a target_cell_id (ReachPositionGoal or ReachCanonicalPositionGoal)
            if hasattr(self.state.goal, 'target_cell_id'):
                goal_cell_id = self.state.goal.target_cell_id

        # Render observation at 64x64 for VLM input
        return render_multigrid(
            self.state,
            self.tiling,
            width=64,
            height=64,
            goal_cell_id=goal_cell_id
        )

    def _get_info(self) -> dict:
        """Get info dict."""
        return {
            "step": self.steps,
            "agent_cell": self.state.agent.cell_id
        }

    def _compute_reward(self, done: bool, action_info: dict) -> float:
        """Compute reward signal."""
        if done:
            return 1.0  # Goal achieved
        elif action_info.get("invalid_action"):
            return -0.01  # Small penalty for invalid actions
        else:
            return 0.0  # Neutral

    def _render_frame(self) -> np.ndarray:
        """Render frame to RGB array."""
        if self.state is None:
            return np.zeros((640, 640, 3), dtype=np.uint8)

        # Get goal cell ID for rendering if goal is position-based
        goal_cell_id = None
        if self.state.goal is not None:
            if hasattr(self.state.goal, 'target_cell_id'):
                goal_cell_id = self.state.goal.target_cell_id

        # Render at higher resolution for human viewing
        return render_multigrid(
            self.state,
            self.tiling,
            width=640,
            height=640,
            goal_cell_id=goal_cell_id
        )

    def _render_human(self):
        """Render for human viewing."""
        if self.state is None:
            print("No state to render")
            return

        # Print state info
        print(f"Step {self.steps}, Agent at {self.state.agent.cell_id}, Facing: {self.state.agent.facing}")

        # Try to display image if PIL is available
        try:
            from PIL import Image
            frame = self._render_frame()
            img = Image.fromarray(frame)
            img.show()
        except ImportError:
            print("PIL not available for image display")
