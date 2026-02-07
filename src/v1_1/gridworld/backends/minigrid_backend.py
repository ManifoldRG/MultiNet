"""
MiniGrid Backend Implementation

Wraps the gymnasium MiniGrid environment with the AbstractGridBackend interface.
"""

from typing import Optional

import numpy as np

from ..task_spec import TaskSpecification
from ..task_parser import TaskParser
from ..custom_env import CustomMiniGridEnv
from .base import AbstractGridBackend, GridState


class MiniGridBackend(AbstractGridBackend):
    """
    Backend implementation using gymnasium's MiniGrid package.

    This is the MVP backend that wraps MiniGrid environments and
    provides the standard AbstractGridBackend interface.
    """

    def __init__(self, render_mode: Optional[str] = "rgb_array"):
        """
        Initialize the MiniGrid backend.

        Args:
            render_mode: Rendering mode ("human", "rgb_array", or None)
        """
        super().__init__()
        self.render_mode = render_mode
        self.parser = TaskParser(render_mode=render_mode)
        self.env: Optional[CustomMiniGridEnv] = None
        self._last_obs = None

    def configure(self, task_spec: TaskSpecification) -> None:
        """
        Configure the backend with a task specification.

        Args:
            task_spec: The task specification defining the puzzle
        """
        self.task_spec = task_spec
        self._configured = True
        # Environment will be created on reset

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]:
        """
        Reset the environment to initial state.

        This method creates a fresh environment from the configured task specification.
        It leverages the TaskParser to handle environment creation and grid population.

        IMPORTANT DESIGN NOTE - Why we don't call env.reset() here:
        The TaskParser.parse() method internally calls env.reset() to initialize the
        grid structure, then populates it with task-specific objects. If we were to
        call reset() again here, it would wipe out all the carefully placed objects
        (keys, doors, switches, etc.) and leave us with an empty grid!

        This is a deliberate architectural choice:
        - TaskParser handles: environment creation + reset + population
        - Backend reset() handles: triggering parser + extracting observations/state

        Args:
            seed: Random seed for reproducibility. Passed through to the parser
                  to ensure deterministic environment initialization.

        Returns:
            observation: The initial RGB observation (image array)
            state: The initial GridState containing agent position, mechanism states, etc.
            info: Additional information dictionary (currently empty, for future use)

        Raises:
            RuntimeError: If configure() has not been called before reset()
        """
        if not self._configured:
            raise RuntimeError("Backend must be configured before reset")

        # Create fresh environment from task spec
        # CRITICAL: parser.parse() internally calls env.reset() and populates the grid.
        # We must NOT call reset() again here or it will wipe out all objects!
        self.env = self.parser.parse(self.task_spec, seed=seed)

        # Generate observation (env is already reset and populated by parser)
        obs = self.env.gen_obs()
        info = {}

        # Get RGB observation
        # MiniGrid supports two rendering modes: direct RGB or symbolic observation
        if self.render_mode == "rgb_array":
            # Use environment's built-in renderer for high-quality RGB output
            rgb_obs = self.env.render()
        else:
            # Convert symbolic observation to RGB
            rgb_obs = self._obs_to_rgb(obs)

        # Cache observation for later render() calls
        self._last_obs = rgb_obs

        # Extract backend-agnostic GridState for evaluation
        state = self._get_grid_state()

        # Include partial observation data in info
        obs_mode = self.task_spec.rules.observability if self.task_spec else "full"
        if obs_mode != "full":
            info["partial_obs"] = obs  # The MiniGrid symbolic partial observation

        return rgb_obs, state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        """
        Execute one action in the environment.

        Args:
            action: The action to execute (0-6 for MiniGrid actions)

        Returns:
            observation: The new observation (RGB image)
            reward: The reward for this step
            terminated: Whether the episode ended
            truncated: Whether the episode was cut short
            state: The new GridState
            info: Additional information dictionary
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update fog-of-war explored cells after movement
        obs_mode = self.task_spec.rules.observability if self.task_spec else "full"
        if obs_mode in ("view_cone", "fog_of_war"):
            self.env.update_explored()

        # Get RGB observation
        if self.render_mode == "rgb_array":
            rgb_obs = self.env.render()
        else:
            rgb_obs = self._obs_to_rgb(obs)

        self._last_obs = rgb_obs
        state = self._get_grid_state()
        state.terminated = terminated
        state.truncated = truncated
        state.reward = reward
        state.goal_reached = terminated and reward > 0

        return rgb_obs, reward, terminated, truncated, state, info

    def render(self) -> np.ndarray:
        """
        Render the current environment state.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.render_mode == "rgb_array":
            return self.env.render()
        elif self._last_obs is not None:
            return self._last_obs
        else:
            # Return placeholder
            return np.zeros((64, 64, 3), dtype=np.uint8)

    def get_mission_text(self) -> str:
        """
        Get the mission/goal description text.

        Returns:
            Human-readable mission description
        """
        if self.env is not None:
            return self.env.mission
        elif self.task_spec is not None:
            return self.task_spec.get_mission_text()
        return "Navigate to the goal"

    def get_state(self) -> GridState:
        """
        Get the current environment state.

        Returns:
            Current GridState
        """
        return self._get_grid_state()

    def _get_grid_state(self) -> GridState:
        """
        Extract GridState from current environment state.

        This method creates a backend-agnostic representation of the current
        environment state by inspecting the CustomMiniGridEnv and extracting
        all relevant information into a standardized GridState object.

        The GridState abstraction allows evaluation code to work with any backend
        (MiniGrid, MultiGrid, or future implementations) without backend-specific
        knowledge.

        State Extraction Process:
        1. Agent state: position, direction, held object
        2. Mechanism states: switches (active/inactive), gates (open/closed)
        3. Block positions: locate all blocks by grid scan
        4. Goal state: check if agent reached goal position

        Performance Note:
        Block position tracking requires a full grid scan (O(width * height) per block).
        This is acceptable for small grids (8x8 to 32x32) but could be optimized
        for larger environments by maintaining a position cache.

        Returns:
            GridState object with current environment state, or a default empty
            state if the environment is not initialized.
        """
        # Return empty state if environment not initialized
        if self.env is None:
            return GridState(
                agent_position=(0, 0),
                agent_direction=0,
            )

        # Extract agent carrying information
        # The agent can carry keys or other objects. We extract the color for keys,
        # or a string representation for other object types.
        carrying = None
        if self.env.carrying is not None:
            # Try to get color attribute (for keys), fall back to string representation
            carrying = getattr(self.env.carrying, "color", str(self.env.carrying))

        # Initialize mechanism state tracking containers
        open_doors = set()  # Currently unused but reserved for future door state tracking
        collected_keys = set()  # Currently unused but reserved for key collection tracking
        active_switches = set()  # IDs of switches that are currently activated
        open_gates = set()  # IDs of gates that are currently open (passable)
        block_positions = {}  # Maps block_id -> (x, y) position

        # Track switch states
        # Switches can be toggled on/off to control gates
        for switch_id, switch in self.env.switches.items():
            if switch.is_active:
                active_switches.add(switch_id)

        # Track gate states
        # Gates can be open (passable) or closed (blocking)
        for gate_id, gate in self.env.gates.items():
            if gate.is_open:
                open_gates.add(gate_id)

        # Track block positions
        # Blocks can be pushed around, so we need to locate them in the grid.
        # This requires scanning the entire grid for each block.
        # TODO: Consider maintaining a position cache to avoid O(N*W*H) complexity
        for block_id, block in self.env.blocks.items():
            # Find block position by scanning grid
            found = False
            for x in range(self.env.width):
                for y in range(self.env.height):
                    cell = self.env.grid.get(x, y)
                    if cell is block:
                        block_positions[block_id] = (x, y)
                        found = True
                        break  # Exit inner loop
                if found:
                    break  # Exit outer loop

        # Track teleporter cooldown states
        teleporter_cooldowns = {}
        for tp_id, tp in self.env.teleporters.items():
            teleporter_cooldowns[tp_id] = tp.cooldown

        # Check if goal has been reached
        # Goal is reached when agent position matches goal position from task spec
        goal_reached = False
        if self.task_spec is not None:
            goal_pos = self.task_spec.maze.goal.to_tuple()
            goal_reached = self.env.agent_pos == goal_pos

        # Get observability info
        obs_mode = self.task_spec.rules.observability if self.task_spec else "full"
        visible_cells = set()
        explored_cells = set()
        if obs_mode != "full":
            visible_cells = self.env.get_visible_cells()
            explored_cells = set(self.env.explored_cells)

        # Construct and return the GridState
        return GridState(
            agent_position=self.env.agent_pos,
            agent_direction=self.env.agent_dir,
            agent_carrying=carrying,
            step_count=self.env.step_count,
            max_steps=self.env.max_steps,
            open_doors=open_doors,
            collected_keys=collected_keys,
            active_switches=active_switches,
            open_gates=open_gates,
            block_positions=block_positions,
            teleporter_cooldowns=teleporter_cooldowns,
            goal_reached=goal_reached,
            observability_mode=obs_mode,
            visible_cells=visible_cells,
            explored_cells=explored_cells,
        )

    def _obs_to_rgb(self, obs: dict) -> np.ndarray:
        """
        Convert MiniGrid observation to RGB image.

        Args:
            obs: MiniGrid observation dict

        Returns:
            RGB image array
        """
        if isinstance(obs, dict) and "image" in obs:
            # Symbolic observation - need to render
            return self.env.render() if self.env else np.zeros((64, 64, 3), dtype=np.uint8)
        elif isinstance(obs, np.ndarray):
            if obs.shape[-1] == 3:
                return obs.astype(np.uint8)
            else:
                # Symbolic grid observation
                return self.env.render() if self.env else np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            return self.env.render() if self.env else np.zeros((64, 64, 3), dtype=np.uint8)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of rendered observations."""
        if self.env is not None:
            img = self.env.render()
            return img.shape
        return (64, 64, 3)

    def close(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
            self.env = None
