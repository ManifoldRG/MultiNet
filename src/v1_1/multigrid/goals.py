# multigrid/goals.py

"""
Goal System for MultiGrid Environments

Provides goal predicates that can be checked against world state to determine
if an episode has been successfully completed.

Supported goal types:
- reach_position: Agent must reach a specific cell
- collect_all: Agent must collect all specified objects
- push_block_to: Agent must push block(s) to target position(s)
- survive_steps: Agent must survive for N steps (always returns False until truncation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .world import WorldState
    from .base import Tiling


class Goal(ABC):
    """Abstract base class for goal predicates."""

    @abstractmethod
    def check(self, state: "WorldState") -> bool:
        """
        Check if the goal condition is satisfied.

        Args:
            state: Current world state

        Returns:
            True if goal is achieved, False otherwise
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the goal."""
        pass


class ReachPositionGoal(Goal):
    """Goal: Agent must reach a specific cell."""

    def __init__(self, target_cell_id: str):
        """
        Args:
            target_cell_id: The cell ID the agent must reach
        """
        self.target_cell_id = target_cell_id

    def check(self, state: "WorldState") -> bool:
        return state.agent.cell_id == self.target_cell_id

    def get_description(self) -> str:
        return f"Reach position {self.target_cell_id}"


class ReachCanonicalPositionGoal(Goal):
    """Goal: Agent must reach a cell at canonical coordinates (uses nearest cell)."""

    def __init__(self, x: float, y: float, tiling: "Tiling"):
        """
        Args:
            x: Target x coordinate (normalized 0-1)
            y: Target y coordinate (normalized 0-1)
            tiling: Tiling to convert coordinates to cell ID
        """
        self.x = x
        self.y = y
        self.tiling = tiling
        self._target_cell_id: Optional[str] = None

    @property
    def target_cell_id(self) -> str:
        if self._target_cell_id is None:
            self._target_cell_id = self.tiling.canonical_to_cell(self.x, self.y)
        return self._target_cell_id

    def check(self, state: "WorldState") -> bool:
        return state.agent.cell_id == self.target_cell_id

    def get_description(self) -> str:
        return f"Reach position ({self.x:.2f}, {self.y:.2f})"


class CollectAllGoal(Goal):
    """Goal: Agent must collect all specified objects."""

    def __init__(self, object_ids: list[str]):
        """
        Args:
            object_ids: List of object IDs that must be collected
        """
        self.object_ids = set(object_ids)
        self.collected: set[str] = set()

    def check(self, state: "WorldState") -> bool:
        # Check which objects are no longer in the world (collected)
        remaining_objects = set(state.objects.keys())
        collected = self.object_ids - remaining_objects

        # Also check if agent is holding any target objects
        if state.agent.holding and state.agent.holding.id in self.object_ids:
            collected.add(state.agent.holding.id)

        return collected == self.object_ids

    def get_description(self) -> str:
        return f"Collect all items: {', '.join(self.object_ids)}"


class PushBlockToGoal(Goal):
    """Goal: Push specified block(s) to target position(s)."""

    def __init__(self, block_targets: dict[str, str]):
        """
        Args:
            block_targets: Mapping of block_id -> target_cell_id
        """
        self.block_targets = block_targets

    def check(self, state: "WorldState") -> bool:
        for block_id, target_cell in self.block_targets.items():
            if block_id not in state.objects:
                return False  # Block doesn't exist
            if state.objects[block_id].cell_id != target_cell:
                return False  # Block not at target
        return True

    def get_description(self) -> str:
        targets = [f"{bid} to {cell}" for bid, cell in self.block_targets.items()]
        return f"Push blocks: {', '.join(targets)}"


class SurviveStepsGoal(Goal):
    """Goal: Survive for N steps (never returns True from check, relies on truncation)."""

    def __init__(self, steps: int):
        """
        Args:
            steps: Number of steps to survive
        """
        self.steps = steps

    def check(self, state: "WorldState") -> bool:
        # This goal is achieved via truncation, not termination
        return False

    def get_description(self) -> str:
        return f"Survive for {self.steps} steps"


class ObjectInZoneGoal(Goal):
    """Goal: A specified object must be inside a zone's covered_cells for N consecutive steps."""

    def __init__(self, object_id: str, zone_id: str, consecutive_steps: int = 1):
        self.object_id = object_id
        self.zone_id = zone_id
        self.consecutive_steps = consecutive_steps
        self._steps_in_zone = 0

    def check(self, state: "WorldState") -> bool:
        obj = state.objects.get(self.object_id)
        zone = state.objects.get(self.zone_id)
        if obj and zone and obj.cell_id in zone.covered_cells:
            self._steps_in_zone += 1
        else:
            self._steps_in_zone = 0
        return self._steps_in_zone >= self.consecutive_steps

    def get_description(self) -> str:
        desc = f"Object {self.object_id} in zone {self.zone_id}"
        if self.consecutive_steps > 1:
            desc += f" for {self.consecutive_steps} consecutive steps"
        return desc


class CompositeGoal(Goal):
    """Goal: All sub-goals must be achieved (AND logic)."""

    def __init__(self, goals: list[Goal]):
        """
        Args:
            goals: List of goals that must all be satisfied
        """
        self.goals = goals

    def check(self, state: "WorldState") -> bool:
        return all(goal.check(state) for goal in self.goals)

    def get_description(self) -> str:
        descs = [goal.get_description() for goal in self.goals]
        return " AND ".join(descs)


class AnyGoal(Goal):
    """Goal: Any one sub-goal must be achieved (OR logic)."""

    def __init__(self, goals: list[Goal]):
        """
        Args:
            goals: List of goals where any one is sufficient
        """
        self.goals = goals

    def check(self, state: "WorldState") -> bool:
        return any(goal.check(state) for goal in self.goals)

    def get_description(self) -> str:
        descs = [goal.get_description() for goal in self.goals]
        return " OR ".join(descs)


def create_goal_from_spec(goal_spec: dict, tiling: "Tiling") -> Goal:
    """
    Create a Goal object from a goal specification dictionary.

    Args:
        goal_spec: Dictionary containing goal specification
            - type: Goal type ("reach_position", "collect_all", "push_block_to", "survive_steps")
            - target: Target position for reach_position (dict with x, y)
            - target_ids: List of object IDs for collect_all
            - block_targets: Dict of block_id -> target position for push_block_to
            - auxiliary_conditions: Additional goals to AND together

        tiling: Tiling instance for coordinate conversion

    Returns:
        Goal object
    """
    goal_type = goal_spec.get("type", "reach_position")
    goals = []

    if goal_type == "reach_position":
        target = goal_spec.get("target")
        if target:
            if isinstance(target, dict):
                # Canonical coordinates
                goals.append(ReachCanonicalPositionGoal(target["x"], target["y"], tiling))
            elif isinstance(target, str):
                # Cell ID
                goals.append(ReachPositionGoal(target))
            elif isinstance(target, (list, tuple)) and len(target) == 2:
                # [x, y] format - treat as canonical coordinates
                goals.append(ReachCanonicalPositionGoal(float(target[0]), float(target[1]), tiling))

    elif goal_type == "collect_all":
        target_ids = goal_spec.get("target_ids", [])
        if target_ids:
            goals.append(CollectAllGoal(target_ids))

    elif goal_type == "push_block_to":
        # Build block_targets mapping
        target_ids = goal_spec.get("target_ids", [])
        target_positions = goal_spec.get("target_positions", [])

        if target_ids and target_positions:
            block_targets = {}
            for block_id, target_pos in zip(target_ids, target_positions):
                if isinstance(target_pos, dict):
                    target_cell = tiling.canonical_to_cell(target_pos["x"], target_pos["y"])
                elif isinstance(target_pos, (list, tuple)) and len(target_pos) == 2:
                    target_cell = tiling.canonical_to_cell(float(target_pos[0]), float(target_pos[1]))
                else:
                    target_cell = str(target_pos)
                block_targets[block_id] = target_cell
            goals.append(PushBlockToGoal(block_targets))

    elif goal_type == "object_in_zone":
        goals.append(ObjectInZoneGoal(
            goal_spec["object_id"],
            goal_spec["zone_id"],
            goal_spec.get("consecutive_steps", 1),
        ))

    elif goal_type == "survive_steps":
        steps = goal_spec.get("steps", goal_spec.get("max_steps", 100))
        goals.append(SurviveStepsGoal(steps))

    # Handle auxiliary conditions
    auxiliary = goal_spec.get("auxiliary_conditions", [])
    for aux in auxiliary:
        if isinstance(aux, dict):
            aux_goal = create_goal_from_spec(aux, tiling)
            goals.append(aux_goal)
        elif isinstance(aux, str):
            # Simple string conditions (could be expanded)
            pass

    if len(goals) == 0:
        # Default: reach position (0.9, 0.9) - bottom-right
        return ReachCanonicalPositionGoal(0.9, 0.9, tiling)
    elif len(goals) == 1:
        return goals[0]
    else:
        return CompositeGoal(goals)
