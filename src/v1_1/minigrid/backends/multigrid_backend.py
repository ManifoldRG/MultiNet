# minigrid/backends/multigrid_backend.py

"""
MultiGrid Backend Implementation

Adapter for the custom MultiGrid system (src/v1_1/multigrid/) that implements
the AbstractGridBackend interface. This allows evaluation of custom tilings
(square, hex, triangle) using the same pipeline as MiniGrid.

Usage:
    from minigrid.backends import MultiGridBackend

    # Use with triangle tiling
    backend = MultiGridBackend(tiling="triangle", render_mode="rgb_array")
    backend.configure(task_spec)
    obs, state, info = backend.reset(seed=42)
    obs, reward, terminated, truncated, state, info = backend.step(action)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .base import AbstractGridBackend, GridState
from ..task_spec import TaskSpecification

# Add parent directory to path for multigrid imports
_multigrid_path = Path(__file__).parent.parent.parent / "multigrid"
if str(_multigrid_path.parent) not in sys.path:
    sys.path.insert(0, str(_multigrid_path.parent))


class MultiGridBackend(AbstractGridBackend):
    """
    Backend adapter for the custom MultiGrid system.

    Supports exotic tilings: square, hex, triangle.

    Args:
        tiling: Tiling type ("square", "hex", "triangle")
        render_mode: Render mode ("rgb_array" or "human")
        render_width: Width of rendered image (default 640)
        render_height: Height of rendered image (default 640)
    """

    def __init__(
        self,
        tiling: str = "square",
        render_mode: str = "rgb_array",
        render_width: int = 640,
        render_height: int = 640,
    ):
        super().__init__()
        self.tiling_type = tiling
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height

        # Will be initialized on configure()
        self.env = None
        self._step_count = 0
        self._max_steps = 100

    def configure(self, task_spec: TaskSpecification) -> None:
        """
        Configure the backend with a task specification.

        Converts the TaskSpecification to the multigrid format and creates
        the environment.

        Args:
            task_spec: The task specification defining the puzzle
        """
        self.task_spec = task_spec

        # Convert TaskSpecification to multigrid task_spec dict
        multigrid_spec = self._convert_task_spec(task_spec)

        # Import and create MultiGridEnv
        from multigrid.env import MultiGridEnv

        self.env = MultiGridEnv(
            task_spec=multigrid_spec,
            tiling=self.tiling_type,
            render_mode=self.render_mode,
        )

        self._max_steps = task_spec.max_steps
        self._configured = True

    def _convert_task_spec(self, spec: TaskSpecification) -> dict:
        """
        Convert TaskSpecification to multigrid task_spec dict format.

        This method bridges the gap between the standard MiniGrid TaskSpecification
        format (used for consistency across backends) and the MultiGrid-specific
        format required by the custom MultiGrid environment.

        Key Differences Between Formats:
        1. Coordinate System:
           - MiniGrid: Integer grid coordinates (e.g., x=3, y=5)
           - MultiGrid: Normalized [0,1] coordinates (e.g., x=0.375, y=0.625)

        2. Object Representation:
           - MiniGrid: Separate mechanism types (keys, doors, blocks)
           - MultiGrid: Unified "objects" list with type field

        3. Tiling Support:
           - MiniGrid: Implicit square tiling
           - MultiGrid: Explicit tiling type (square, hex, triangle)

        Translation Strategy:
        - Keys → "movable" objects (can be picked up)
        - Doors → "wall" objects with color (blocking barriers)
        - Blocks → "movable" objects (pushable)
        - Switches/Gates → Not yet implemented in MultiGrid backend
        - Positions → Normalized by dividing by grid dimensions

        Note on Coordinate Normalization:
        MultiGrid uses normalized [0,1] coordinates to support different tilings
        uniformly. For example, in an 8x8 grid, position (4, 4) becomes (0.5, 0.5).
        This allows the same task to be rendered on square, hex, or triangle grids.

        Args:
            spec: TaskSpecification from the minigrid module (standard format)

        Returns:
            Dictionary in multigrid format ready for MultiGridEnv initialization

        Limitations:
            - Switches and gates are not yet supported (MultiGrid enhancement needed)
            - Teleporters not implemented
            - Hazards not implemented
            - All objects except goal are treated as "movable" or "wall"
        """
        # Build walls list from maze layout
        # Walls are kept in absolute coordinates as MultiGrid handles them specially
        walls = [[w.x, w.y] for w in spec.maze.walls]

        # Build scene objects list
        # All interactive objects are collected here with unified format
        objects = []

        # Add keys as movable objects
        # Keys can be picked up and carried by the agent
        for key in spec.mechanisms.keys:
            objects.append({
                "id": key.id,
                "type": "movable",
                "color": key.color,
                # Normalize position to [0,1] range for MultiGrid
                "position": {"x": key.position.x / spec.maze.dimensions[0],
                            "y": key.position.y / spec.maze.dimensions[1]}
            })

        # Add doors as walls (or special handling)
        # Doors are treated as colored walls in the current MultiGrid implementation
        # TODO: Enhance MultiGrid to support door unlocking mechanics
        for door in spec.mechanisms.doors:
            objects.append({
                "id": door.id,
                "type": "wall",  # Doors are blocking barriers
                "color": door.requires_key,  # Color indicates which key unlocks it
                "position": {"x": door.position.x / spec.maze.dimensions[0],
                            "y": door.position.y / spec.maze.dimensions[1]}
            })

        # Add blocks as movable objects
        # Blocks can be pushed by the agent (Sokoban-style)
        for block in spec.mechanisms.blocks:
            objects.append({
                "id": block.id,
                "type": "movable",
                "color": "grey",  # Default block color
                "position": {"x": block.position.x / spec.maze.dimensions[0],
                            "y": block.position.y / spec.maze.dimensions[1]}
            })

        # Build goal specification
        # MultiGrid supports multiple goal types with different win conditions
        goal_spec = {}
        if spec.goal:
            if spec.goal.goal_type == "reach_position":
                # Win by reaching a specific position
                goal_spec = {
                    "type": "reach_position",
                    "target": {
                        "x": spec.goal.target.x / spec.maze.dimensions[0],
                        "y": spec.goal.target.y / spec.maze.dimensions[1]
                    }
                }
            elif spec.goal.goal_type == "collect_all":
                # Win by collecting all specified objects
                goal_spec = {
                    "type": "collect_all",
                    "target_ids": spec.goal.target_ids
                }
            elif spec.goal.goal_type == "push_block_to":
                # Win by pushing blocks to target positions (Sokoban-style)
                goal_spec = {
                    "type": "push_block_to",
                    "target_ids": spec.goal.target_ids,
                    "target_positions": [
                        {"x": p.x / spec.maze.dimensions[0],
                         "y": p.y / spec.maze.dimensions[1]}
                        for p in spec.goal.target_positions
                    ] if spec.goal.target_positions else []
                }

        # Construct complete MultiGrid task specification
        return {
            "task_id": spec.task_id,
            "seed": spec.seed,
            "tiling": {
                "type": self.tiling_type,  # square, hex, or triangle
                "grid_size": {
                    "width": spec.maze.dimensions[0],
                    "height": spec.maze.dimensions[1]
                }
            },
            "scene": {
                "agent": {
                    "position": {
                        # Agent start position in normalized coordinates
                        "x": spec.maze.start.x / spec.maze.dimensions[0],
                        "y": spec.maze.start.y / spec.maze.dimensions[1]
                    },
                    "facing": 0  # Default direction (right)
                },
                "objects": objects,
                "walls": walls
            },
            "goal": goal_spec,
            "limits": {
                "max_steps": spec.max_steps
            }
        }

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
        if not self._configured or self.env is None:
            raise RuntimeError("Backend must be configured before reset")

        obs, info = self.env.reset(seed=seed)
        self._step_count = 0

        state = self._build_grid_state()

        return obs, state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]:
        """
        Execute one action in the environment.

        This method provides the bridge between the standard MiniGrid action space
        (used for consistency across backends) and the MultiGrid-specific action
        indices. The mapping ensures that the same agent policy can work with both
        backends without modification.

        Action Space Translation:
        MiniGrid uses a 7-action discrete space (0-6), while MultiGrid has a
        different internal action enumeration. This method translates between them:

        MiniGrid Action → MultiGrid Action
        0: turn_left     → 2: TURN_LEFT
        1: turn_right    → 3: TURN_RIGHT
        2: forward       → 0: FORWARD
        3: pickup        → 4: PICKUP
        4: drop          → 5: DROP
        5: toggle        → 6: PUSH (closest equivalent for switch/door interaction)
        6: done/wait     → 7: WAIT

        Note on "toggle" vs "PUSH":
        MiniGrid's "toggle" action is used for switches, doors, and other interactive
        objects. MultiGrid's closest equivalent is "PUSH", which can interact with
        objects in front of the agent. This mapping may need refinement as MultiGrid
        adds more interaction mechanics.

        Design Rationale:
        The action mapping allows evaluation code to use standard MiniGrid action
        indices regardless of backend. This is critical for:
        - Running the same agent policy on different backends
        - Comparing results across backends
        - Using pre-trained models that expect MiniGrid actions

        Args:
            action: The action to execute (0-6, standard MiniGrid action space)

        Returns:
            observation: RGB image of the new state
            reward: Reward for this step
            terminated: Whether the episode ended (goal reached or failure)
            truncated: Whether the episode was cut short (max steps reached)
            state: GridState representing the new environment state
            info: Additional information dictionary from the environment

        Raises:
            RuntimeError: If the backend has not been configured or reset
        """
        if not self._configured or self.env is None:
            raise RuntimeError("Backend must be configured before step")

        # Map MiniGrid action to MultiGrid action
        # This translation ensures compatibility between backends
        action_map = {
            0: 2,  # turn_left -> TURN_LEFT
            1: 3,  # turn_right -> TURN_RIGHT
            2: 0,  # forward -> FORWARD
            3: 4,  # pickup -> PICKUP
            4: 5,  # drop -> DROP
            5: 6,  # toggle -> PUSH (closest equivalent)
            6: 7,  # done -> WAIT
        }

        # Get MultiGrid action index, default to WAIT if action invalid
        multigrid_action = action_map.get(action, 7)

        # Execute action in MultiGrid environment
        obs, reward, terminated, truncated, info = self.env.step(multigrid_action)

        # Track step count (MultiGrid doesn't track this internally)
        self._step_count += 1

        # Build GridState for backend-agnostic representation
        state = self._build_grid_state()
        # Update state with step results
        state.terminated = terminated
        state.truncated = truncated
        state.reward = reward
        state.step_count = self._step_count

        return obs, reward, terminated, truncated, state, info

    def render(self) -> np.ndarray:
        """
        Render the current environment state.

        Returns:
            RGB image array of shape (H, W, 3)
        """
        if self.env is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

        return self.env.render()

    def get_mission_text(self) -> str:
        """
        Get the mission/goal description text.

        Returns:
            Human-readable mission description
        """
        if self.task_spec is None:
            return "No mission"

        # Use task description or generate from goal
        if self.task_spec.description:
            return self.task_spec.description

        if self.task_spec.goal:
            goal_type = self.task_spec.goal.goal_type
            if goal_type == "reach_position":
                return f"Navigate to position ({self.task_spec.goal.target.x}, {self.task_spec.goal.target.y})"
            elif goal_type == "collect_all":
                return f"Collect all items: {', '.join(self.task_spec.goal.target_ids)}"
            elif goal_type == "push_block_to":
                return "Push blocks to target positions"

        return "Complete the task"

    def get_state(self) -> GridState:
        """
        Get the current environment state.

        Returns:
            Current GridState
        """
        return self._build_grid_state()

    def _build_grid_state(self) -> GridState:
        """
        Build a GridState from the current MultiGrid state.

        Returns:
            GridState representing current environment
        """
        if self.env is None or self.env.state is None:
            return GridState(
                agent_position=(0, 0),
                agent_direction=0,
                step_count=self._step_count,
                max_steps=self._max_steps,
            )

        state = self.env.state
        tiling = self.env.tiling

        # Get agent position in grid coordinates
        agent_pos = tiling.cell_to_canonical(state.agent.cell_id)
        grid_pos = (
            int(agent_pos[0] * self.task_spec.maze.dimensions[0]),
            int(agent_pos[1] * self.task_spec.maze.dimensions[1])
        )

        # Get carrying object
        carrying = None
        if state.agent.holding is not None:
            carrying = state.agent.holding.id

        # Build block positions
        block_positions = {}
        for obj_id, obj in state.objects.items():
            if obj.obj_type == "movable" and obj.cell_id is not None:
                pos = tiling.cell_to_canonical(obj.cell_id)
                block_positions[obj_id] = (
                    int(pos[0] * self.task_spec.maze.dimensions[0]),
                    int(pos[1] * self.task_spec.maze.dimensions[1])
                )

        return GridState(
            agent_position=grid_pos,
            agent_direction=state.agent.facing,
            agent_carrying=carrying,
            step_count=self._step_count,
            max_steps=self._max_steps,
            block_positions=block_positions,
            goal_reached=state.check_goal(),
        )

    def close(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            # MultiGridEnv doesn't have explicit close
            self.env = None
        self._configured = False

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of observations (H, W, C)."""
        return (64, 64, 3)
