# gridworld/backends/multigrid_backend.py

"""
MultiGrid Backend Implementation

Adapter for the custom MultiGrid system (src/v1_1/multigrid/) that implements
the AbstractGridBackend interface. This allows evaluation of custom tilings
(square, hex, triangle) using the same pipeline as MiniGrid.

Usage:
    from gridworld.backends import MultiGridBackend

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

        # Extract observability settings from task_spec
        obs_mode = task_spec.rules.observability if task_spec.rules else "full"
        view_size = task_spec.rules.view_size if task_spec.rules else 7
        partial = obs_mode != "full"

        # Import and create MultiGridEnv
        from multigrid.env import MultiGridEnv

        self.env = MultiGridEnv(
            task_spec=multigrid_spec,
            tiling=self.tiling_type,
            render_mode=self.render_mode,
            partial_obs=partial,
            obs_radius=view_size // 2,
            observability_mode=obs_mode,
        )

        self._max_steps = task_spec.max_steps
        self._configured = True

    def _convert_task_spec(self, spec: TaskSpecification) -> dict:
        """
        Convert TaskSpecification to multigrid task_spec dict format.

        This method bridges the gap between the standard MiniGrid TaskSpecification
        format (used for consistency across backends) and the MultiGrid-specific
        format required by the custom MultiGrid environment.

        This preserves the canonical TaskSpecification semantics by emitting the
        corresponding native MultiGrid object types rather than degrading them.

        Args:
            spec: TaskSpecification from the minigrid module (standard format)

        Returns:
            Dictionary in multigrid format ready for MultiGridEnv initialization

        Limitations:
            - Border cells are represented as explicit wall objects so square-grid
              semantics match the MiniGrid backend.
        """
        width, height = spec.maze.dimensions

        def canonical_pos(x: int, y: int) -> dict:
            return {
                "x": (x + 0.5) / width,
                "y": (y + 0.5) / height,
            }

        # Build scene objects list
        objects = []

        wall_positions = {(w.x, w.y) for w in spec.maze.walls}
        for x in range(width):
            wall_positions.add((x, 0))
            wall_positions.add((x, height - 1))
        for y in range(height):
            wall_positions.add((0, y))
            wall_positions.add((width - 1, y))

        for x, y in sorted(wall_positions):
            objects.append({
                "id": f"wall_{x}_{y}",
                "type": "wall",
                "color": "grey",
                "position": canonical_pos(x, y),
            })

        for key in spec.mechanisms.keys:
            objects.append({
                "id": key.id,
                "type": "key",
                "color": key.color,
                "position": canonical_pos(key.position.x, key.position.y),
            })

        for door in spec.mechanisms.doors:
            objects.append({
                "id": door.id,
                "type": "door",
                "color": door.requires_key,
                "position": canonical_pos(door.position.x, door.position.y),
                "is_locked": door.initial_state == "locked",
            })

        for switch in spec.mechanisms.switches:
            objects.append({
                "id": switch.id,
                "type": "switch",
                "color": "yellow",
                "position": canonical_pos(switch.position.x, switch.position.y),
                "controls": switch.controls,
                "switch_type": switch.switch_type,
                "initial_state": switch.initial_state == "on",
            })

        for gate in spec.mechanisms.gates:
            controlled_by = [
                switch.id for switch in spec.mechanisms.switches if gate.id in switch.controls
            ]
            objects.append({
                "id": gate.id,
                "type": "gate",
                "color": "grey",
                "position": canonical_pos(gate.position.x, gate.position.y),
                "is_open": gate.initial_state == "open",
                "controlled_by": controlled_by,
            })

        for block in spec.mechanisms.blocks:
            objects.append({
                "id": block.id,
                "type": "movable",
                "color": block.color,
                "position": canonical_pos(block.position.x, block.position.y),
            })

        for hazard in spec.mechanisms.hazards:
            objects.append({
                "id": hazard.id,
                "type": "hazard",
                "color": "red",
                "position": canonical_pos(hazard.position.x, hazard.position.y),
                "hazard_type": hazard.hazard_type,
            })

        for teleporter in spec.mechanisms.teleporters:
            a_id = f"{teleporter.id}_a"
            b_id = f"{teleporter.id}_b"
            objects.append({
                "id": a_id,
                "type": "teleporter",
                "color": "purple",
                "position": canonical_pos(teleporter.position_a.x, teleporter.position_a.y),
                "linked_to": b_id,
            })
            objects.append({
                "id": b_id,
                "type": "teleporter",
                "color": "purple",
                "position": canonical_pos(teleporter.position_b.x, teleporter.position_b.y),
                "linked_to": a_id if teleporter.bidirectional else None,
            })

        goal_spec = {}
        if spec.goal:
            if spec.goal.goal_type == "reach_position":
                goal_target = spec.goal.target or spec.maze.goal
                goal_spec = {
                    "type": "reach_position",
                    "target": {
                        "x": (goal_target.x + 0.5) / width,
                        "y": (goal_target.y + 0.5) / height,
                    }
                }
            elif spec.goal.goal_type == "collect_all":
                goal_spec = {
                    "type": "collect_all",
                    "target_ids": spec.goal.target_ids
                }
            elif spec.goal.goal_type == "push_block_to":
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
                        "x": (spec.maze.start.x + 0.5) / width,
                        "y": (spec.maze.start.y + 0.5) / height,
                    },
                    "facing": 0  # Default direction (right)
                },
                "objects": objects,
            },
            "goal": goal_spec,
            "rules": {
                "key_consumption": spec.rules.key_consumption,
                "switch_type": spec.rules.switch_type,
            },
            "limits": {
                "max_steps": spec.max_steps
            },
            "metadata": spec.metadata or {},
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
            5: 6,  # toggle -> TOGGLE
            6: 8,  # done -> WAIT
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

        block_ids = {block.id for block in self.task_spec.mechanisms.blocks}
        key_ids = {key.id for key in self.task_spec.mechanisms.keys}

        # Build block positions
        block_positions = {}
        for obj_id, obj in state.objects.items():
            if obj_id in block_ids and obj.obj_type == "movable" and obj.cell_id is not None:
                pos = tiling.cell_to_canonical(obj.cell_id)
                block_positions[obj_id] = (
                    int(pos[0] * self.task_spec.maze.dimensions[0]),
                    int(pos[1] * self.task_spec.maze.dimensions[1])
                )

        # Convert visibility sets from cell_id strings to (x,y) grid coords
        obs_mode = getattr(state, 'observability_mode', 'full')
        visible_xy = set()
        explored_xy = set()

        if obs_mode != "full":
            dims = self.task_spec.maze.dimensions
            for cell_id in state.visible_cells:
                pos = tiling.cell_to_canonical(cell_id)
                visible_xy.add((int(pos[0] * dims[0]), int(pos[1] * dims[1])))
            for cell_id in state.explored_cells:
                pos = tiling.cell_to_canonical(cell_id)
                explored_xy.add((int(pos[0] * dims[0]), int(pos[1] * dims[1])))

        open_doors = {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "door" and getattr(obj, "is_open", False)
        }
        active_switches = {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "switch" and getattr(obj, "is_active", False)
        }
        open_gates = {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "gate" and getattr(obj, "is_open", False)
        }
        collected_keys = key_ids - {
            obj.id for obj in state.objects.values()
            if obj.obj_type == "key"
        }
        teleporter_cooldowns = {
            obj.id: getattr(obj, "current_cooldown", 0)
            for obj in state.objects.values()
            if obj.obj_type == "teleporter"
        }

        return GridState(
            agent_position=grid_pos,
            agent_direction=state.agent.facing,
            agent_carrying=carrying,
            step_count=self._step_count,
            max_steps=self._max_steps,
            open_doors=open_doors,
            collected_keys=collected_keys,
            active_switches=active_switches,
            open_gates=open_gates,
            block_positions=block_positions,
            teleporter_cooldowns=teleporter_cooldowns,
            goal_reached=state.check_goal(),
            observability_mode=obs_mode,
            visible_cells=visible_xy,
            explored_cells=explored_xy,
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
