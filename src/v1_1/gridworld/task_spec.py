"""
Task Specification Schema for MiniGrid Domain

Defines the complete JSON schema for gridworld puzzles, matching the PDF specification.
Supports tiers 1-5: Navigation, Linear Dependencies, Multi-Mechanism, Irreversibility, Hidden Info.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Any
import json


@dataclass
class Position:
    """2D grid position."""
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    @classmethod
    def from_list(cls, coords: list[int]) -> "Position":
        return cls(x=coords[0], y=coords[1])

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(x=d["x"], y=d["y"])


@dataclass
class KeySpec:
    """Key object specification."""
    id: str
    position: Position
    color: str  # "red", "blue", "green", "yellow", "purple", "grey"

    @classmethod
    def from_dict(cls, d: dict) -> "KeySpec":
        return cls(
            id=d["id"],
            position=Position.from_list(d["position"]) if isinstance(d["position"], list) else Position.from_dict(d["position"]),
            color=d["color"]
        )


@dataclass
class DoorSpec:
    """Door object specification."""
    id: str
    position: Position
    requires_key: str  # color that unlocks this door
    initial_state: Literal["locked", "open"] = "locked"

    @classmethod
    def from_dict(cls, d: dict) -> "DoorSpec":
        return cls(
            id=d["id"],
            position=Position.from_list(d["position"]) if isinstance(d["position"], list) else Position.from_dict(d["position"]),
            requires_key=d["requires_key"],
            initial_state=d.get("initial_state", "locked")
        )


@dataclass
class SwitchSpec:
    """Switch/button specification for controlling gates."""
    id: str
    position: Position
    controls: list[str]  # list of gate IDs this switch controls
    switch_type: Literal["toggle", "hold", "one_shot"] = "toggle"
    initial_state: Literal["on", "off"] = "off"

    @classmethod
    def from_dict(cls, d: dict) -> "SwitchSpec":
        return cls(
            id=d["id"],
            position=Position.from_list(d["position"]) if isinstance(d["position"], list) else Position.from_dict(d["position"]),
            controls=d["controls"],
            switch_type=d.get("switch_type", "toggle"),
            initial_state=d.get("initial_state", "off")
        )


@dataclass
class GateSpec:
    """Gate specification (controlled by switches)."""
    id: str
    position: Position
    initial_state: Literal["open", "closed"] = "closed"

    @classmethod
    def from_dict(cls, d: dict) -> "GateSpec":
        return cls(
            id=d["id"],
            position=Position.from_list(d["position"]) if isinstance(d["position"], list) else Position.from_dict(d["position"]),
            initial_state=d.get("initial_state", "closed")
        )


@dataclass
class BlockSpec:
    """Pushable block specification (for Sokoban-style puzzles)."""
    id: str
    position: Position
    pushable: bool = True
    color: str = "grey"

    @classmethod
    def from_dict(cls, d: dict) -> "BlockSpec":
        return cls(
            id=d["id"],
            position=Position.from_list(d["position"]) if isinstance(d["position"], list) else Position.from_dict(d["position"]),
            pushable=d.get("pushable", True),
            color=d.get("color", "grey")
        )


@dataclass
class TeleporterSpec:
    """Teleporter pair specification."""
    id: str
    position_a: Position
    position_b: Position
    bidirectional: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "TeleporterSpec":
        return cls(
            id=d["id"],
            position_a=Position.from_list(d["position_a"]) if isinstance(d["position_a"], list) else Position.from_dict(d["position_a"]),
            position_b=Position.from_list(d["position_b"]) if isinstance(d["position_b"], list) else Position.from_dict(d["position_b"]),
            bidirectional=d.get("bidirectional", True)
        )


@dataclass
class HazardSpec:
    """Hazard/lava specification."""
    id: str
    position: Position
    hazard_type: Literal["lava", "pit", "spike"] = "lava"

    @classmethod
    def from_dict(cls, d: dict) -> "HazardSpec":
        return cls(
            id=d["id"],
            position=Position.from_list(d["position"]) if isinstance(d["position"], list) else Position.from_dict(d["position"]),
            hazard_type=d.get("hazard_type", "lava")
        )


@dataclass
class MazeLayout:
    """Maze geometry and structure."""
    dimensions: tuple[int, int]  # (width, height)
    walls: list[Position]
    start: Position
    goal: Position
    floor: Optional[list[Position]] = None  # If not specified, all non-wall cells are floor

    @classmethod
    def from_dict(cls, d: dict) -> "MazeLayout":
        dims = tuple(d["dimensions"])
        walls = [Position.from_list(w) if isinstance(w, list) else Position.from_dict(w) for w in d.get("walls", [])]
        start = Position.from_list(d["start"]) if isinstance(d["start"], list) else Position.from_dict(d["start"])
        goal = Position.from_list(d["goal"]) if isinstance(d["goal"], list) else Position.from_dict(d["goal"])
        floor = None
        if "floor" in d and d["floor"]:
            floor = [Position.from_list(f) if isinstance(f, list) else Position.from_dict(f) for f in d["floor"]]
        return cls(dimensions=dims, walls=walls, start=start, goal=goal, floor=floor)


@dataclass
class MechanismSet:
    """Collection of all interactive mechanisms in the puzzle."""
    keys: list[KeySpec] = field(default_factory=list)
    doors: list[DoorSpec] = field(default_factory=list)
    switches: list[SwitchSpec] = field(default_factory=list)
    gates: list[GateSpec] = field(default_factory=list)
    blocks: list[BlockSpec] = field(default_factory=list)
    teleporters: list[TeleporterSpec] = field(default_factory=list)
    hazards: list[HazardSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MechanismSet":
        return cls(
            keys=[KeySpec.from_dict(k) for k in d.get("keys", [])],
            doors=[DoorSpec.from_dict(door) for door in d.get("doors", [])],
            switches=[SwitchSpec.from_dict(s) for s in d.get("switches", [])],
            gates=[GateSpec.from_dict(g) for g in d.get("gates", [])],
            blocks=[BlockSpec.from_dict(b) for b in d.get("blocks", [])],
            teleporters=[TeleporterSpec.from_dict(t) for t in d.get("teleporters", [])],
            hazards=[HazardSpec.from_dict(h) for h in d.get("hazards", [])],
        )


@dataclass
class Rules:
    """Puzzle rule configuration."""
    key_consumption: bool = True  # Keys are consumed when used
    switch_type: Literal["toggle", "hold", "one_shot"] = "toggle"  # Default switch behavior
    hidden_mechanisms: list[str] = field(default_factory=list)  # IDs of mechanisms not visible initially
    observability: Literal["full", "view_cone", "fog_of_war"] = "full"
    view_size: int = 7  # Agent view cone size (must be odd, >= 3). Only used when observability != "full"

    @classmethod
    def from_dict(cls, d: dict) -> "Rules":
        return cls(
            key_consumption=d.get("key_consumption", True),
            switch_type=d.get("switch_type", "toggle"),
            hidden_mechanisms=d.get("hidden_mechanisms", []),
            observability=d.get("observability", "full"),
            view_size=d.get("view_size", 7),
        )


@dataclass
class GoalSpec:
    """Goal/win condition specification."""
    goal_type: Literal["reach_position", "collect_all", "push_block_to", "survive_steps"] = "reach_position"
    target: Optional[Position] = None  # For reach_position
    target_ids: list[str] = field(default_factory=list)  # For collect_all or push_block_to
    target_positions: list[Position] = field(default_factory=list)  # For push_block_to
    auxiliary_conditions: list[str] = field(default_factory=list)  # Additional requirements

    @classmethod
    def from_dict(cls, d: dict) -> "GoalSpec":
        target = None
        if "target" in d and d["target"]:
            target = Position.from_list(d["target"]) if isinstance(d["target"], list) else Position.from_dict(d["target"])
        target_positions = []
        if "target_positions" in d:
            target_positions = [
                Position.from_list(p) if isinstance(p, list) else Position.from_dict(p)
                for p in d["target_positions"]
            ]
        return cls(
            goal_type=d.get("type", d.get("goal_type", "reach_position")),
            target=target,
            target_ids=d.get("target_ids", []),
            target_positions=target_positions,
            auxiliary_conditions=d.get("auxiliary_conditions", [])
        )


@dataclass
class TaskSpecification:
    """Complete task specification for a gridworld puzzle."""
    task_id: str
    seed: int
    difficulty_tier: int  # 1-5
    maze: MazeLayout
    mechanisms: MechanismSet
    rules: Rules
    goal: GoalSpec
    max_steps: int
    version: str = "1.0"
    description: str = ""  # Human-readable task description

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpecification":
        """Parse from dictionary (e.g., loaded JSON)."""
        # Handle nested TaskSpecification key if present
        if "TaskSpecification" in d:
            d = d["TaskSpecification"]

        # Parse maze layout
        maze_data = d.get("maze", {})
        if "layout" in maze_data:
            # Nested layout format from PDF spec
            layout = maze_data["layout"]
            maze_layout = MazeLayout(
                dimensions=tuple(maze_data["dimensions"]),
                walls=[Position.from_list(w) if isinstance(w, list) else Position.from_dict(w) for w in layout.get("walls", [])],
                start=Position.from_list(layout["start"]) if isinstance(layout["start"], list) else Position.from_dict(layout["start"]),
                goal=Position.from_list(layout["goal"]) if isinstance(layout["goal"], list) else Position.from_dict(layout["goal"]),
                floor=[Position.from_list(f) if isinstance(f, list) else Position.from_dict(f) for f in layout.get("floor", [])] if layout.get("floor") else None
            )
            # Mechanisms may be under maze
            mechanisms_data = maze_data.get("mechanisms", d.get("mechanisms", {}))
        else:
            # Flat format
            maze_layout = MazeLayout.from_dict(maze_data) if maze_data else MazeLayout(
                dimensions=(8, 8),
                walls=[],
                start=Position(1, 1),
                goal=Position(6, 6)
            )
            mechanisms_data = d.get("mechanisms", {})

        mechanisms = MechanismSet.from_dict(mechanisms_data)
        rules = Rules.from_dict(d.get("rules", {}))
        goal = GoalSpec.from_dict(d.get("goal", {}))

        return cls(
            task_id=d.get("task_id", "unknown"),
            seed=d.get("seed", 42),
            difficulty_tier=d.get("difficulty_tier", 1),
            maze=maze_layout,
            mechanisms=mechanisms,
            rules=rules,
            goal=goal,
            max_steps=d.get("max_steps", 100),
            version=d.get("version", "1.0"),
            description=d.get("description", "")
        )

    @classmethod
    def from_json(cls, path: str) -> "TaskSpecification":
        """Load task specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        def pos_to_list(p: Position) -> list[int]:
            return [p.x, p.y]

        return {
            "task_id": self.task_id,
            "version": self.version,
            "seed": self.seed,
            "difficulty_tier": self.difficulty_tier,
            "description": self.description,
            "maze": {
                "dimensions": list(self.maze.dimensions),
                "walls": [pos_to_list(w) for w in self.maze.walls],
                "start": pos_to_list(self.maze.start),
                "goal": pos_to_list(self.maze.goal),
                "floor": [pos_to_list(f) for f in self.maze.floor] if self.maze.floor else None
            },
            "mechanisms": {
                "keys": [{"id": k.id, "position": pos_to_list(k.position), "color": k.color} for k in self.mechanisms.keys],
                "doors": [{"id": d.id, "position": pos_to_list(d.position), "requires_key": d.requires_key, "initial_state": d.initial_state} for d in self.mechanisms.doors],
                "switches": [{"id": s.id, "position": pos_to_list(s.position), "controls": s.controls, "switch_type": s.switch_type, "initial_state": s.initial_state} for s in self.mechanisms.switches],
                "gates": [{"id": g.id, "position": pos_to_list(g.position), "initial_state": g.initial_state} for g in self.mechanisms.gates],
                "blocks": [{"id": b.id, "position": pos_to_list(b.position), "pushable": b.pushable, "color": b.color} for b in self.mechanisms.blocks],
                "teleporters": [{"id": t.id, "position_a": pos_to_list(t.position_a), "position_b": pos_to_list(t.position_b), "bidirectional": t.bidirectional} for t in self.mechanisms.teleporters],
                "hazards": [{"id": h.id, "position": pos_to_list(h.position), "hazard_type": h.hazard_type} for h in self.mechanisms.hazards],
            },
            "rules": {
                "key_consumption": self.rules.key_consumption,
                "switch_type": self.rules.switch_type,
                "hidden_mechanisms": self.rules.hidden_mechanisms,
                "observability": self.rules.observability,
                "view_size": self.rules.view_size,
            },
            "goal": {
                "type": self.goal.goal_type,
                "target": pos_to_list(self.goal.target) if self.goal.target else None,
                "target_ids": self.goal.target_ids,
                "target_positions": [pos_to_list(p) for p in self.goal.target_positions],
                "auxiliary_conditions": self.goal.auxiliary_conditions
            },
            "max_steps": self.max_steps
        }

    def to_json(self, path: str) -> None:
        """Save task specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the task specification for consistency.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        width, height = self.maze.dimensions

        # Check dimensions
        if width < 3 or height < 3:
            errors.append(f"Maze dimensions too small: {width}x{height}, minimum is 3x3")

        # Check start position
        if not (0 <= self.maze.start.x < width and 0 <= self.maze.start.y < height):
            errors.append(f"Start position {self.maze.start.to_tuple()} out of bounds")

        # Check goal position
        if not (0 <= self.maze.goal.x < width and 0 <= self.maze.goal.y < height):
            errors.append(f"Goal position {self.maze.goal.to_tuple()} out of bounds")

        # Check that start and goal are not walls
        wall_positions = {w.to_tuple() for w in self.maze.walls}
        if self.maze.start.to_tuple() in wall_positions:
            errors.append("Start position is a wall")
        if self.maze.goal.to_tuple() in wall_positions:
            errors.append("Goal position is a wall")

        # Check all mechanism positions are in bounds and not walls
        def check_position(pos: Position, name: str):
            if not (0 <= pos.x < width and 0 <= pos.y < height):
                errors.append(f"{name} position {pos.to_tuple()} out of bounds")
            elif pos.to_tuple() in wall_positions:
                errors.append(f"{name} position {pos.to_tuple()} is a wall")

        for key in self.mechanisms.keys:
            check_position(key.position, f"Key {key.id}")

        for door in self.mechanisms.doors:
            check_position(door.position, f"Door {door.id}")

        for switch in self.mechanisms.switches:
            check_position(switch.position, f"Switch {switch.id}")

        for gate in self.mechanisms.gates:
            check_position(gate.position, f"Gate {gate.id}")

        for block in self.mechanisms.blocks:
            check_position(block.position, f"Block {block.id}")

        for hazard in self.mechanisms.hazards:
            check_position(hazard.position, f"Hazard {hazard.id}")

        for teleporter in self.mechanisms.teleporters:
            check_position(teleporter.position_a, f"Teleporter {teleporter.id} endpoint A")
            check_position(teleporter.position_b, f"Teleporter {teleporter.id} endpoint B")

        # Check door-key color consistency
        key_colors = {k.color for k in self.mechanisms.keys}
        for door in self.mechanisms.doors:
            if door.requires_key not in key_colors:
                errors.append(f"Door {door.id} requires color '{door.requires_key}' but no key of that color exists")

        # Check switch-gate consistency
        gate_ids = {g.id for g in self.mechanisms.gates}
        for switch in self.mechanisms.switches:
            for controlled_id in switch.controls:
                if controlled_id not in gate_ids:
                    errors.append(f"Switch {switch.id} controls non-existent gate '{controlled_id}'")

        # Check difficulty tier
        if not 1 <= self.difficulty_tier <= 5:
            errors.append(f"Invalid difficulty tier: {self.difficulty_tier}, must be 1-5")

        # Check max_steps
        if self.max_steps < 1:
            errors.append(f"Invalid max_steps: {self.max_steps}, must be positive")

        return len(errors) == 0, errors

    def get_mission_text(self) -> str:
        """Generate a human-readable mission description."""
        if self.description:
            return self.description

        parts = []

        # Goal description
        if self.goal.goal_type == "reach_position":
            parts.append("Navigate to the goal")
        elif self.goal.goal_type == "collect_all":
            parts.append("Collect all required items")
        elif self.goal.goal_type == "push_block_to":
            parts.append("Push the block to the target position")
        elif self.goal.goal_type == "survive_steps":
            parts.append(f"Survive for {self.max_steps} steps")

        # Mechanism hints
        if self.mechanisms.keys:
            parts.append(f"Keys: {len(self.mechanisms.keys)}")
        if self.mechanisms.doors:
            parts.append(f"Locked doors: {len(self.mechanisms.doors)}")
        if self.mechanisms.switches:
            parts.append(f"Switches: {len(self.mechanisms.switches)}")
        if self.mechanisms.blocks:
            parts.append(f"Pushable blocks: {len(self.mechanisms.blocks)}")
        if self.mechanisms.hazards:
            parts.append("Avoid hazards")

        return ". ".join(parts) + "."
