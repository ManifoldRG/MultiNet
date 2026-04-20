"""
GridWorld Domain Adapter

Maps canonical task specs to MiniGrid/MultiGrid environments.
Handles coordinate normalization between [0,1] canonical space
and integer grid coordinates.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .canonical_task_spec import CanonicalTaskSpec, CanonicalGoal, CanonicalObject
from .domain_adapter import DomainAdapter

try:
    from ..gridworld.backends.base import AbstractGridBackend, GridState
    from ..gridworld.backends.minigrid_backend import MiniGridBackend
    from ..gridworld.task_spec import (
        TaskSpecification, MazeLayout, MechanismSet, Rules, GoalSpec, Position,
        KeySpec, DoorSpec, SwitchSpec, GateSpec, BlockSpec, TeleporterSpec, HazardSpec,
    )
except ImportError:
    from gridworld.backends.base import AbstractGridBackend, GridState
    from gridworld.backends.minigrid_backend import MiniGridBackend
    from gridworld.task_spec import (
        TaskSpecification, MazeLayout, MechanismSet, Rules, GoalSpec, Position,
        KeySpec, DoorSpec, SwitchSpec, GateSpec, BlockSpec, TeleporterSpec, HazardSpec,
    )


# Mapping from canonical object types to MiniGrid mechanism types
CANONICAL_TO_MECHANISM = {
    "barrier": "wall",
    "collectible": "key",
    "interactive": "switch",
    "hazard": "hazard",
    "portal": "teleporter",
    "door": "door",
    "gate": "gate",
    "block": "block",
}


class GridWorldDomainAdapter(DomainAdapter):
    """
    Domain adapter for MiniGrid/MultiGrid gridworld environments.

    Converts between canonical [0,1] coordinates and integer grid positions.
    """

    def __init__(
        self,
        backend: Optional[AbstractGridBackend] = None,
        render_mode: str = "rgb_array",
    ):
        self.backend = backend or MiniGridBackend(render_mode=render_mode)
        self._task_spec: Optional[TaskSpecification] = None
        self._state: Optional[GridState] = None
        self._obs: Optional[np.ndarray] = None

    @property
    def domain_name(self) -> str:
        return "gridworld"

    @property
    def action_type(self) -> str:
        return "discrete"

    def from_canonical(self, spec: CanonicalTaskSpec) -> TaskSpecification:
        """Convert canonical spec to MiniGrid TaskSpecification."""
        # Determine grid dimensions from domain_config or default
        grid_w = spec.domain_config.get("grid_width", 10)
        grid_h = spec.domain_config.get("grid_height", 10)

        def denorm(pos: tuple[float, ...]) -> Position:
            """Convert normalized [0,1] to grid coordinates."""
            x = max(1, min(grid_w - 2, int(pos[0] * (grid_w - 1))))
            y = max(1, min(grid_h - 2, int(pos[1] * (grid_h - 1))))
            return Position(x, y)

        # Build mechanisms from canonical objects
        keys = []
        doors = []
        switches = []
        gates = []
        blocks = []
        teleporters = []
        hazards = []
        walls = []

        for obj in spec.objects:
            pos = denorm(obj.position)
            props = obj.properties

            if obj.obj_type == "barrier":
                walls.append(pos)
            elif obj.obj_type == "collectible":
                keys.append(KeySpec(
                    id=obj.id,
                    position=pos,
                    color=props.get("color", "yellow"),
                ))
            elif obj.obj_type == "door":
                doors.append(DoorSpec(
                    id=obj.id,
                    position=pos,
                    requires_key=props.get("requires_key", "yellow"),
                    initial_state=props.get("initial_state", "locked"),
                ))
            elif obj.obj_type == "interactive" and props.get("subtype") == "gate":
                gates.append(GateSpec(
                    id=obj.id,
                    position=pos,
                    initial_state=props.get("initial_state", "closed"),
                ))
            elif obj.obj_type == "interactive":
                switches.append(SwitchSpec(
                    id=obj.id,
                    position=pos,
                    controls=props.get("controls", []),
                    switch_type=props.get("switch_type", "toggle"),
                ))
            elif obj.obj_type == "block":
                blocks.append(BlockSpec(
                    id=obj.id,
                    position=pos,
                    color=props.get("color", "grey"),
                ))
            elif obj.obj_type == "hazard":
                hazards.append(HazardSpec(
                    id=obj.id,
                    position=pos,
                    hazard_type=props.get("hazard_type", "lava"),
                ))
            elif obj.obj_type == "portal":
                # Portals need paired positions
                pos_b = props.get("position_b")
                if pos_b:
                    teleporters.append(TeleporterSpec(
                        id=obj.id,
                        position_a=pos,
                        position_b=denorm(tuple(pos_b)),
                        bidirectional=props.get("bidirectional", True),
                    ))

        # Build goal
        goal_target = denorm(spec.goal.target) if spec.goal.target else None
        goal = GoalSpec(
            goal_type={
                "reach": "reach_position",
                "collect": "collect_all",
                "arrange": "push_block_to",
                "survive": "survive_steps",
            }.get(spec.goal.goal_type, "reach_position"),
            target=goal_target,
            target_ids=spec.goal.target_ids,
        )

        start = denorm(spec.agent_start)
        goal_pos = goal_target or Position(grid_w - 2, grid_h - 2)

        task_spec = TaskSpecification(
            task_id=spec.task_id,
            seed=spec.seed,
            difficulty_tier=spec.difficulty,
            maze=MazeLayout(
                dimensions=(grid_w, grid_h),
                walls=walls,
                start=start,
                goal=goal_pos,
            ),
            mechanisms=MechanismSet(
                keys=keys,
                doors=doors,
                switches=switches,
                gates=gates,
                blocks=blocks,
                teleporters=teleporters,
                hazards=hazards,
            ),
            rules=Rules(),
            goal=goal,
            max_steps=spec.max_steps,
            description=spec.description,
        )

        self._task_spec = task_spec
        return task_spec

    def to_canonical(self, domain_spec: TaskSpecification) -> CanonicalTaskSpec:
        """Convert MiniGrid TaskSpecification to canonical spec."""
        grid_w, grid_h = domain_spec.maze.dimensions

        def norm(pos: Position) -> tuple[float, float]:
            """Convert grid coordinates to normalized [0,1]."""
            return (pos.x / (grid_w - 1), pos.y / (grid_h - 1))

        objects = []

        # Convert walls
        for wall in domain_spec.maze.walls:
            objects.append(CanonicalObject(
                id=f"wall_{wall.x}_{wall.y}",
                obj_type="barrier",
                position=norm(wall),
            ))

        # Convert keys
        for key in domain_spec.mechanisms.keys:
            objects.append(CanonicalObject(
                id=key.id,
                obj_type="collectible",
                position=norm(key.position),
                properties={"color": key.color},
            ))

        # Convert doors
        for door in domain_spec.mechanisms.doors:
            objects.append(CanonicalObject(
                id=door.id,
                obj_type="door",
                position=norm(door.position),
                properties={"requires_key": door.requires_key, "initial_state": door.initial_state},
            ))

        # Convert switches
        for switch in domain_spec.mechanisms.switches:
            objects.append(CanonicalObject(
                id=switch.id,
                obj_type="interactive",
                position=norm(switch.position),
                properties={"controls": switch.controls, "switch_type": switch.switch_type},
            ))

        # Convert gates
        for gate in domain_spec.mechanisms.gates:
            objects.append(CanonicalObject(
                id=gate.id,
                obj_type="interactive",
                position=norm(gate.position),
                properties={"subtype": "gate", "initial_state": gate.initial_state},
            ))

        # Convert blocks
        for block in domain_spec.mechanisms.blocks:
            objects.append(CanonicalObject(
                id=block.id,
                obj_type="block",
                position=norm(block.position),
                properties={"color": block.color},
            ))

        # Convert hazards
        for hazard in domain_spec.mechanisms.hazards:
            objects.append(CanonicalObject(
                id=hazard.id,
                obj_type="hazard",
                position=norm(hazard.position),
                properties={"hazard_type": hazard.hazard_type},
            ))

        # Convert teleporters
        for tp in domain_spec.mechanisms.teleporters:
            objects.append(CanonicalObject(
                id=tp.id,
                obj_type="portal",
                position=norm(tp.position_a),
                properties={
                    "position_b": list(norm(tp.position_b)),
                    "bidirectional": tp.bidirectional,
                },
            ))

        # Convert goal
        goal_type_map = {
            "reach_position": "reach",
            "collect_all": "collect",
            "push_block_to": "arrange",
            "survive_steps": "survive",
        }
        canonical_goal = CanonicalGoal(
            goal_type=goal_type_map.get(domain_spec.goal.goal_type, "reach"),
            target=norm(domain_spec.goal.target) if domain_spec.goal.target else None,
            target_ids=domain_spec.goal.target_ids,
        )

        return CanonicalTaskSpec(
            task_id=domain_spec.task_id,
            seed=domain_spec.seed,
            difficulty=domain_spec.difficulty_tier,
            dimensions=(1.0, 1.0),
            agent_start=norm(domain_spec.maze.start),
            goal=canonical_goal,
            objects=objects,
            max_steps=domain_spec.max_steps,
            description=domain_spec.description,
            domain_config={"grid_width": grid_w, "grid_height": grid_h},
        )

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Reset environment."""
        if self._task_spec is None:
            raise RuntimeError("Call from_canonical() before reset()")
        self.backend.configure(self._task_spec)
        obs, state, info = self.backend.reset(seed=seed)
        self._state = state
        self._obs = obs
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute discrete action."""
        obs, reward, terminated, truncated, state, info = self.backend.step(action)
        self._state = state
        self._obs = obs
        return obs, reward, terminated, truncated, info

    def check_success(self) -> bool:
        """Check if goal was reached."""
        if self._state is None:
            return False
        return self._state.goal_reached

    def render(self) -> Optional[np.ndarray]:
        return self.backend.render()

    def close(self) -> None:
        self.backend.close()
