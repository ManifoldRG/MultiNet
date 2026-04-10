"""
Custom MiniGrid Environment

A configurable MiniGrid environment that can be populated from TaskSpecification.
Supports all mechanism types: keys, doors, switches, gates, blocks, hazards.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

from .bootstrap import disable_gymnasium_env_plugins

disable_gymnasium_env_plugins()

# Import from gymnasium's minigrid package (no naming conflict after rename to gridworld/)
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Key, Door, Goal, Wall, Lava, Box, Ball
from minigrid.minigrid_env import MiniGridEnv

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

    def __init__(
        self,
        color: str = "yellow",
        switch_id: str = "",
        controls: list[str] = None,
        switch_type: str = "toggle",
        initial_state: str = "off",
    ):
        super().__init__(color)
        self.switch_id = switch_id
        self.controls = controls or []
        self.switch_type = switch_type
        self.is_active = initial_state == "on"
        self.used = self.is_active and switch_type == "one_shot"

    def can_pickup(self):
        return False

    def can_overlap(self):
        return self.switch_type == "hold"

    def activate(self):
        """Apply switch-type-specific activation semantics."""
        if self.switch_type == "one_shot":
            if self.used:
                return False
            self.used = True
            self.is_active = True
            return True
        if self.switch_type == "hold":
            if not self.is_active:
                self.is_active = True
                return True
            return False
        self.is_active = not self.is_active
        return True

    def deactivate(self):
        """Deactivate hold-type switches when the agent leaves the tile."""
        if self.switch_type == "hold" and self.is_active:
            self.is_active = False
            return True
        return False


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


class TeleporterObj(Ball):
    """
    Teleporter endpoint object.
    When the agent steps on it, they are teleported to the partner endpoint.
    Rendered as a ball with special portal appearance.
    """

    def __init__(self, color: str = "purple", teleporter_id: str = "",
                 partner: "TeleporterObj | None" = None, cooldown_max: int = 1):
        super().__init__(color)
        self.teleporter_id = teleporter_id
        self.partner: TeleporterObj | None = partner
        self.cooldown = 0
        self.cooldown_max = cooldown_max

    def can_overlap(self):
        return True

    def can_pickup(self):
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
        see_through_walls: bool = True,
        agent_view_size: int = 7,
        highlight: bool = True,
        agent_pov: bool = False,
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
        self.teleporters: dict[str, TeleporterObj] = {}
        self.switch_gate_map: dict[str, list[str]] = {}  # switch_id -> [gate_ids]
        self.gate_initial_state: dict[str, bool] = {}

        # Fog of war tracking: set of (x, y) cells the agent has visited/seen
        self.explored_cells: set[tuple[int, int]] = set()

        # Mission space for the environment - the func returns our custom text
        mission_space = MissionSpace(mission_func=lambda: mission_text)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agent_view_size=agent_view_size,
            highlight=highlight,
            agent_pov=agent_pov,
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

        # Reset fog-of-war tracking
        self.explored_cells = set()

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

    def place_switch(
        self,
        x: int,
        y: int,
        switch_id: str,
        controls: list[str],
        switch_type: str = "toggle",
        initial_state: str = "off",
        color: str = "yellow",
    ):
        """Place a switch at the given position."""
        switch = Switch(
            color=color,
            switch_id=switch_id,
            controls=controls,
            switch_type=switch_type,
            initial_state=initial_state,
        )
        self.switches[switch_id] = switch
        self.switch_gate_map[switch_id] = controls
        self.put_obj(switch, x, y)
        self._refresh_gates()

    def place_gate(self, x: int, y: int, gate_id: str, is_open: bool = False, color: str = "grey"):
        """Place a gate at the given position."""
        gate = Gate(color=color, gate_id=gate_id, is_open=is_open)
        self.gates[gate_id] = gate
        self.gate_initial_state[gate_id] = is_open
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

    def place_teleporter(self, teleporter_id: str, x_a: int, y_a: int,
                         x_b: int, y_b: int, bidirectional: bool = True,
                         color: str = "purple"):
        """Place a teleporter pair at the given positions."""
        tp_a = TeleporterObj(color=color, teleporter_id=f"{teleporter_id}_a")
        tp_b = TeleporterObj(color=color, teleporter_id=f"{teleporter_id}_b")
        tp_a.partner = tp_b
        if bidirectional:
            tp_b.partner = tp_a
        self.teleporters[f"{teleporter_id}_a"] = tp_a
        self.teleporters[f"{teleporter_id}_b"] = tp_b
        self.put_obj(tp_a, x_a, y_a)
        self.put_obj(tp_b, x_b, y_b)

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

    def _refresh_gates(self):
        """Recompute gate states from initial configuration and switch activity."""
        for gate_id, gate in self.gates.items():
            is_open = self.gate_initial_state.get(gate_id, False)
            for switch_id, controls in self.switch_gate_map.items():
                switch = self.switches.get(switch_id)
                if switch is not None and gate_id in controls and switch.is_active:
                    is_open = True
            gate.is_open = is_open

    def _update_hold_switches(self):
        """Keep hold-type switches active only while the agent stands on them."""
        changed = False
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Switch) and cell.switch_type == "hold":
                    if (x, y) == self.agent_pos:
                        changed = cell.activate() or changed
                    else:
                        changed = cell.deactivate() or changed
        if changed:
            self._refresh_gates()

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

        # Handle gate toggle attempt (gates can only be opened by switches, not directly)
        if action == self.actions.toggle and isinstance(fwd_cell, Gate):
            # No-op: gates are not directly toggleable
            self.step_count += 1
            truncated = self.step_count >= self.max_steps
            obs = self.gen_obs()
            return obs, 0, False, truncated, {}

        # Handle switch interaction
        if action == self.actions.toggle and isinstance(fwd_cell, Switch):
            if not fwd_cell.activate():
                self.step_count += 1
                truncated = self.step_count >= self.max_steps
                obs = self.gen_obs()
                return obs, 0, False, truncated, {"invalid_action": True}
            self._refresh_gates()
            # Return after handling (don't fall through to super which would re-toggle)
            self.step_count += 1
            truncated = self.step_count >= self.max_steps
            obs = self.gen_obs()
            return obs, 0, False, truncated, {}

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
        obs, reward, terminated, truncated, info = super().step(action)
        if action == self.actions.forward:
            self._update_hold_switches()

        # Tick teleporter cooldowns
        for tp in self.teleporters.values():
            if tp.cooldown > 0:
                tp.cooldown -= 1

        # Check if agent landed on a teleporter after moving forward
        if action == self.actions.forward:
            cell = self.grid.get(*self.agent_pos)
            if isinstance(cell, TeleporterObj) and cell.partner is not None and cell.cooldown == 0:
                # Find partner position
                for x in range(self.width):
                    for y in range(self.height):
                        if self.grid.get(x, y) is cell.partner:
                            self.agent_pos = (x, y)
                            # Set cooldown on destination to prevent immediate bounce-back
                            cell.partner.cooldown = cell.partner.cooldown_max
                            # Regenerate observation after teleport
                            obs = self.gen_obs()
                            break
                    else:
                        continue
                    break

        return obs, reward, terminated, truncated, info

    def get_mission_text(self) -> str:
        """Return the mission text."""
        return self._custom_mission_text

    def get_visible_cells(self) -> set[tuple[int, int]]:
        """Get the set of (x, y) cells currently visible to the agent via view cone.

        Uses the same coordinate mapping as MiniGrid's get_frame highlight logic:
        the vis_mask from gen_obs_grid is in rotated agent-relative space, and we
        map back to absolute grid coordinates using dir_vec / right_vec.
        """
        _, vis_mask = self.gen_obs_grid()
        visible = set()

        # MiniGrid coordinate mapping: agent is at bottom-center of rotated view
        f_vec = self.dir_vec
        r_vec = np.array((-f_vec[1], f_vec[0]))
        top_left = (
            np.array(self.agent_pos)
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        for vis_i in range(self.agent_view_size):
            for vis_j in range(self.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue
                abs_pos = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                abs_x, abs_y = int(abs_pos[0]), int(abs_pos[1])
                if 0 <= abs_x < self.width and 0 <= abs_y < self.height:
                    visible.add((abs_x, abs_y))
        return visible

    def update_explored(self):
        """Update fog-of-war: add currently visible cells to explored set."""
        self.explored_cells |= self.get_visible_cells()
