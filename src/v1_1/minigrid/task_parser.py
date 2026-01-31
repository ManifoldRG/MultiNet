"""
Task Parser for MiniGrid Domain

Parses TaskSpecification JSON files and creates configured MiniGrid environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import gymnasium as gym

from .task_spec import TaskSpecification, Position
from .custom_env import CustomMiniGridEnv


class TaskParser:
    """
    Parse TaskSpecification and create configured MiniGrid environments.

    Usage:
        parser = TaskParser()
        env = parser.parse(task_spec)
        # or
        env = parser.parse_file("path/to/task.json")
    """

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the parser.

        Args:
            render_mode: Rendering mode for created environments ("human", "rgb_array", None)
        """
        self.render_mode = render_mode

    def parse(self, spec: TaskSpecification, seed: Optional[int] = None) -> CustomMiniGridEnv:
        """
        Create a configured MiniGrid environment from a TaskSpecification.

        This is the core parsing method that transforms a declarative JSON-based
        TaskSpecification into a fully configured, runnable MiniGrid environment.

        The parsing process follows three stages:
        1. Validation: Ensures the spec is internally consistent (bounds checking,
           dependency validation, etc.)
        2. Environment Creation: Instantiates a CustomMiniGridEnv with basic parameters
           and calls reset() to initialize the grid with border walls
        3. Grid Population: Adds all task-specific elements (walls, keys, doors,
           switches, gates, blocks, hazards) to the grid

        Note on reset behavior: The environment's reset() method is called internally
        to initialize the grid structure. The parser then populates the grid with
        task-specific objects. This two-phase approach ensures proper initialization
        order while avoiding state corruption.

        Args:
            spec: The task specification to parse. Must contain valid maze dimensions,
                  start/goal positions, and mechanism definitions.
            seed: Optional seed override for environment initialization. If None,
                  uses spec.seed. This enables running the same task with different
                  random seeds for evaluation.

        Returns:
            Configured CustomMiniGridEnv ready for use. The environment is already
            reset and populated with all objects from the specification.

        Raises:
            ValueError: If the task specification fails validation. Error message
                        includes all validation failures concatenated.
        """
        # Validate specification to catch errors early
        # This checks bounds, dependency consistency (e.g., doors have matching keys),
        # and other constraints defined in TaskSpecification.validate()
        is_valid, errors = spec.validate()
        if not is_valid:
            raise ValueError(f"Invalid task specification: {'; '.join(errors)}")

        width, height = spec.maze.dimensions

        # Use provided seed or fall back to spec seed
        # This allows the same task to be evaluated with different random seeds
        actual_seed = seed if seed is not None else spec.seed

        # Create the base environment with core parameters
        # The CustomMiniGridEnv is initialized but not yet populated with task objects
        env = CustomMiniGridEnv(
            width=width,
            height=height,
            max_steps=spec.max_steps,
            agent_start_pos=spec.maze.start.to_tuple(),
            agent_start_dir=0,  # Default facing right (standard MiniGrid convention)
            goal_pos=spec.maze.goal.to_tuple(),
            mission_text=spec.get_mission_text(),
            render_mode=self.render_mode,
            task_spec=spec,
        )

        # Reset to initialize the grid structure
        # CRITICAL: This call initializes the grid with border walls and sets up
        # the base environment state. We MUST call reset() before populate_grid()
        # to ensure the grid exists and is properly initialized.
        env.reset(seed=actual_seed)

        # Now populate the grid with task-specific elements
        # This adds all interactive objects (keys, doors, switches, etc.) to the grid
        # The order of placement matters for certain objects (e.g., gates before switches)
        self._populate_grid(env, spec)

        return env

    def parse_file(self, path: Union[str, Path]) -> CustomMiniGridEnv:
        """
        Create a configured MiniGrid environment from a JSON file.

        Args:
            path: Path to the JSON task specification file

        Returns:
            Configured CustomMiniGridEnv ready for use
        """
        spec = TaskSpecification.from_json(str(path))
        return self.parse(spec)

    def parse_dict(self, data: dict) -> CustomMiniGridEnv:
        """
        Create a configured MiniGrid environment from a dictionary.

        Args:
            data: Dictionary containing task specification

        Returns:
            Configured CustomMiniGridEnv ready for use
        """
        spec = TaskSpecification.from_dict(data)
        return self.parse(spec)

    def _populate_grid(self, env: CustomMiniGridEnv, spec: TaskSpecification):
        """
        Populate the environment grid with walls and mechanisms.

        This method is called after environment reset to add all task-specific
        elements to the grid. The placement order is carefully designed to handle
        dependencies between objects and ensure proper initialization.

        Placement Strategy:
        1. Clear interior cells (preserves border walls from reset)
        2. Add static elements: walls, goal
        3. Add collectible items: keys
        4. Add barriers: doors
        5. Add control mechanisms: gates first (so switches can reference them),
           then switches
        6. Add movable objects: blocks
        7. Add hazards: lava/pits/spikes
        8. Finalize: Set agent position (overwrites any objects at start)

        Design Rationale:
        - Gates before switches: Switches store references to gates, so gates
          must exist in env.gates dict before switch placement
        - Agent position last: Ensures the agent always starts at the correct
          position even if other objects were accidentally placed there
        - Border walls preserved: The 1-pixel border is created by reset() and
          should never be modified

        Args:
            env: The CustomMiniGridEnv to populate (must already be reset)
            spec: The task specification containing all object definitions
        """
        # Clear existing grid (except border walls)
        # Border walls at x=0, x=width-1, y=0, y=height-1 are preserved
        width, height = spec.maze.dimensions
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                env.grid.set(x, y, None)

        # Place interior walls
        # Border positions are skipped since reset() already placed walls there
        for wall_pos in spec.maze.walls:
            x, y = wall_pos.x, wall_pos.y
            # Skip border positions (already have walls from reset)
            if 0 < x < width - 1 and 0 < y < height - 1:
                env.place_wall(x, y)

        # Place goal marker
        # The goal position is typically the win condition for navigation tasks
        env.place_goal(spec.maze.goal.x, spec.maze.goal.y)

        # Place keys
        # Keys are collectible items that can unlock doors of matching color
        for key in spec.mechanisms.keys:
            env.place_key(key.position.x, key.position.y, key.color)

        # Place doors
        # Doors can be locked (requiring a matching key) or initially open
        for door in spec.mechanisms.doors:
            is_locked = door.initial_state == "locked"
            env.place_door(door.position.x, door.position.y, door.requires_key, is_locked)

        # Place gates BEFORE switches
        # CRITICAL: Gates must be registered in env.gates before switches are placed,
        # because switches store references to gate IDs and need to validate them
        for gate in spec.mechanisms.gates:
            is_open = gate.initial_state == "open"
            env.place_gate(gate.position.x, gate.position.y, gate.id, is_open)

        # Place switches
        # Switches control gates. When toggled, they change the state of all
        # gates in their controls list
        for switch in spec.mechanisms.switches:
            env.place_switch(
                switch.position.x,
                switch.position.y,
                switch.id,
                switch.controls,  # List of gate IDs this switch controls
            )

        # Place blocks
        # Blocks are pushable objects (Sokoban-style) that can be moved by the agent
        for block in spec.mechanisms.blocks:
            env.place_block(block.position.x, block.position.y, block.id, block.color)

        # Place hazards
        # Hazards (lava, pits, spikes) typically end the episode if touched
        for hazard in spec.mechanisms.hazards:
            env.place_hazard(hazard.position.x, hazard.position.y, hazard.hazard_type)

        # Set agent position (overwrite anything at start position)
        # This is done last to ensure the agent always spawns at the correct location,
        # even if the task specification accidentally placed another object there
        env.set_agent_position(spec.maze.start.x, spec.maze.start.y)


def load_task_from_file(path: Union[str, Path], render_mode: Optional[str] = None) -> CustomMiniGridEnv:
    """
    Convenience function to load a task from a JSON file.

    Args:
        path: Path to the JSON task specification file
        render_mode: Rendering mode for the environment

    Returns:
        Configured CustomMiniGridEnv ready for use
    """
    parser = TaskParser(render_mode=render_mode)
    return parser.parse_file(path)


def load_task_from_dict(data: dict, render_mode: Optional[str] = None) -> CustomMiniGridEnv:
    """
    Convenience function to load a task from a dictionary.

    Args:
        data: Dictionary containing task specification
        render_mode: Rendering mode for the environment

    Returns:
        Configured CustomMiniGridEnv ready for use
    """
    parser = TaskParser(render_mode=render_mode)
    return parser.parse_dict(data)
