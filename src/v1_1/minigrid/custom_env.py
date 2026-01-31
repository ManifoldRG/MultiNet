"""
Custom MiniGrid Environment

A configurable MiniGrid environment that can be populated from TaskSpecification.
Supports all mechanism types: keys, doors, switches, gates, blocks, hazards.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

# Import from gymnasium's minigrid package via helper (avoids naming conflict)
from ._minigrid_pkg import (
    mg_Grid as Grid,
    mg_MissionSpace as MissionSpace,
    mg_WorldObj as WorldObj,
    mg_Key as Key,
    mg_Door as Door,
    mg_Goal as Goal,
    mg_Wall as Wall,
    mg_Lava as Lava,
    mg_Box as Box,
    mg_Ball as Ball,
    mg_MiniGridEnv as MiniGridEnv,
)

from .task_spec import TaskSpecification, Position


# Color mapping for MiniGrid
MINIGRID_COLORS = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "purple": "purple",
    "grey": "grey",
    "gray": "grey",
}


class Switch(Ball):
    """
    Switch object that can control gates.
    Rendered as a ball with special interaction behavior.
    """

    def __init__(self, color: str = "yellow", switch_id: str = "", controls: list[str] = None):
        super().__init__(color)
        self.switch_id = switch_id
        self.controls = controls or []
        self.is_active = False

    def can_pickup(self):
        return False

    def toggle(self, env, pos):
        """Toggle the switch state and update controlled gates."""
        self.is_active = not self.is_active
        # Gate toggling is handled by the environment
        return True


class Gate(Door):
    """
    Gate object controlled by switches.
    When closed, blocks movement like a wall. When open, passable.
    Extends Door for proper rendering.
    """

    def __init__(self, color: str = "grey", gate_id: str = "", is_open: bool = False):
        # Initialize as unlocked door
        super().__init__(color, is_locked=False)
        self.gate_id = gate_id
        self.is_open = is_open

    def can_overlap(self):
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # Gates can only be toggled by switches, not directly
        return False


class PushableBlock(Box):
    """
    A block that can be pushed by the agent.
    Extends Box to leverage existing rendering.
    """

    def __init__(self, color: str = "grey", block_id: str = ""):
        super().__init__(color)
        self.block_id = block_id
        self.pushable = True

    def can_pickup(self):
        return False


class CustomMiniGridEnv(MiniGridEnv):
    """
    Custom MiniGrid environment that can be configured from a TaskSpecification.

    This environment supports:
    - Arbitrary maze layouts
    - Keys and colored doors
    - Switches and gates
    - Pushable blocks
    - Hazards (lava)
    - Custom goal conditions
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        max_steps: int = 100,
        agent_start_pos: Optional[tuple[int, int]] = None,
        agent_start_dir: int = 0,
        goal_pos: Optional[tuple[int, int]] = None,
        mission_text: str = "Navigate to the goal",
        render_mode: Optional[str] = None,
        task_spec: Optional[TaskSpecification] = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self._custom_mission_text = mission_text  # Store our custom mission text
        self.task_spec = task_spec

        # Mechanism tracking
        self.switches: dict[str, Switch] = {}
        self.gates: dict[str, Gate] = {}
        self.blocks: dict[str, PushableBlock] = {}
        self.switch_gate_map: dict[str, list[str]] = {}  # switch_id -> [gate_ids]

        # Mission space for the environment - the func returns our custom text
        mission_space = MissionSpace(mission_func=lambda: mission_text)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

        # After super().__init__, self.mission is set by the parent class
        # We can update it to our custom text if needed
        self.mission = mission_text

    def _gen_grid(self, width: int, height: int):
        """Generate the grid. Called by reset()."""
        # Create empty grid
        self.grid = Grid(width, height)

        # Add border walls
        self.grid.wall_rect(0, 0, width, height)

        # If we have a task spec, it will be populated after _gen_grid by the parser
        # For now, set basic start/goal if provided

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            # Default: place agent at (1, 1)
            self.agent_pos = (1, 1)
            self.agent_dir = 0

        if self.goal_pos is not None:
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

    def place_wall(self, x: int, y: int):
        """Place a wall at the given position."""
        self.grid.set(x, y, Wall())

    def place_key(self, x: int, y: int, color: str):
        """Place a key at the given position."""
        color = MINIGRID_COLORS.get(color, color)
        self.put_obj(Key(color), x, y)

    def place_door(self, x: int, y: int, color: str, is_locked: bool = True):
        """Place a door at the given position."""
        color = MINIGRID_COLORS.get(color, color)
        door = Door(color, is_locked=is_locked)
        self.grid.set(x, y, door)

    def place_switch(self, x: int, y: int, switch_id: str, controls: list[str], color: str = "yellow"):
        """Place a switch at the given position."""
        switch = Switch(color=color, switch_id=switch_id, controls=controls)
        self.switches[switch_id] = switch
        self.switch_gate_map[switch_id] = controls
        self.put_obj(switch, x, y)

    def place_gate(self, x: int, y: int, gate_id: str, is_open: bool = False, color: str = "grey"):
        """Place a gate at the given position."""
        gate = Gate(color=color, gate_id=gate_id, is_open=is_open)
        self.gates[gate_id] = gate
        self.grid.set(x, y, gate)

    def place_block(self, x: int, y: int, block_id: str, color: str = "grey"):
        """Place a pushable block at the given position."""
        block = PushableBlock(color=color, block_id=block_id)
        self.blocks[block_id] = block
        self.put_obj(block, x, y)

    def place_hazard(self, x: int, y: int, hazard_type: str = "lava"):
        """Place a hazard at the given position."""
        # All hazards use Lava for now
        self.grid.set(x, y, Lava())

    def place_goal(self, x: int, y: int):
        """Place the goal at the given position."""
        self.put_obj(Goal(), x, y)

    def set_agent_position(self, x: int, y: int, direction: int = 0):
        """Set the agent's starting position and direction."""
        self.agent_pos = (x, y)
        self.agent_dir = direction

    def toggle_gate(self, gate_id: str):
        """Toggle a gate's open/closed state."""
        if gate_id in self.gates:
            gate = self.gates[gate_id]
            gate.is_open = not gate.is_open

    def step(self, action: int):
        """Execute one step in the environment with custom mechanics."""
        # Get the position in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        # Handle key consumption when unlocking doors
        if action == self.actions.toggle and isinstance(fwd_cell, Door) and not isinstance(fwd_cell, Gate):
            if fwd_cell.is_locked and self.carrying is not None:
                if isinstance(self.carrying, Key) and self.carrying.color == fwd_cell.color:
                    # Key matches - unlock the door
                    fwd_cell.is_locked = False
                    fwd_cell.is_open = True

                    # Check if key should be consumed
                    if self.task_spec and self.task_spec.rules.key_consumption:
                        self.carrying = None  # Consume the key

                    # Return after handling
                    self.step_count += 1
                    truncated = self.step_count >= self.max_steps
                    obs = self.gen_obs()
                    return obs, 0, False, truncated, {}

        # Handle switch interaction
        if action == self.actions.toggle and isinstance(fwd_cell, Switch):
            # Toggle the switch
            fwd_cell.is_active = not fwd_cell.is_active
            # Toggle all controlled gates
            for gate_id in fwd_cell.controls:
                self.toggle_gate(gate_id)

        # Handle block pushing
        if action == self.actions.forward and isinstance(fwd_cell, PushableBlock):
            # Calculate position behind the block
            dir_vec = self.dir_vec
            behind_block_pos = (fwd_pos[0] + dir_vec[0], fwd_pos[1] + dir_vec[1])

            # Check if we can push the block
            behind_cell = self.grid.get(*behind_block_pos)
            if behind_cell is None or behind_cell.can_overlap():
                # Push the block
                self.grid.set(*fwd_pos, None)
                self.grid.set(*behind_block_pos, fwd_cell)
                # Agent moves forward
                self.agent_pos = fwd_pos

                # Check step count and return
                self.step_count += 1

                if self.step_count >= self.max_steps:
                    truncated = True
                else:
                    truncated = False

                # Check if goal reached
                terminated = False
                reward = 0
                if self.goal_pos and self.agent_pos == self.goal_pos:
                    terminated = True
                    reward = 1 - 0.9 * (self.step_count / self.max_steps)
                elif isinstance(self.grid.get(*self.agent_pos), Goal):
                    terminated = True
                    reward = 1 - 0.9 * (self.step_count / self.max_steps)

                obs = self.gen_obs()
                return obs, reward, terminated, truncated, {}

        # Handle gate blocking
        if action == self.actions.forward and isinstance(fwd_cell, Gate) and not fwd_cell.is_open:
            # Can't move through closed gate
            self.step_count += 1
            if self.step_count >= self.max_steps:
                truncated = True
            else:
                truncated = False
            obs = self.gen_obs()
            return obs, 0, False, truncated, {}

        # Default behavior
        return super().step(action)

    def get_mission_text(self) -> str:
        """Return the mission text."""
        return self._custom_mission_text
