"""12-dimension scoring for gridworld tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

from .task_spec import TaskSpecification
from .task_validator import DifficultyReport, TaskValidator


DIMENSION_NAMES = [
    "optimal_path_length",
    "search_space_size",
    "backtracking_required",
    "fragility",
    "dependency_depth",
    "dependency_variety",
    "distractor_count",
    "distractor_quality",
    "grid_size",
    "wall_density",
    "partial_observability",
    "irreversibility",
]


@dataclass
class ScoredDifficulty:
    """Full 12-dimension score report."""
    dimensions: list[float]
    dimension_names: list[str] = field(default_factory=lambda: DIMENSION_NAMES.copy())
    composite: float = 0.0
    weights: list[float] = field(default_factory=lambda: [1.0] * len(DIMENSION_NAMES))

    def to_dict(self) -> dict:
        return {
            "dimensions": self.dimensions,
            "dimension_names": self.dimension_names,
            "composite": self.composite,
            "weights": self.weights,
        }


def _count_backtracking(solution: list[tuple[int, int]] | None) -> float:
    if not solution:
        return 0.0
    seen = set()
    revisits = 0
    for pos in solution:
        if pos in seen:
            revisits += 1
        seen.add(pos)
    return float(revisits)


def _dependency_variety(spec: TaskSpecification) -> float:
    if spec.dependency_chain is not None:
        return float(len({step.type for step in spec.dependency_chain.sequence}))

    variety = 0
    if spec.mechanisms.keys and spec.mechanisms.doors:
        variety += 1
    if spec.mechanisms.switches and spec.mechanisms.gates:
        variety += 1
    if spec.mechanisms.blocks:
        variety += 1
    if spec.mechanisms.teleporters:
        variety += 1
    if spec.mechanisms.hazards:
        variety += 1
    return float(variety)


def _distractor_quality(spec: TaskSpecification) -> float:
    if not spec.distractors:
        return 0.0
    weights = {
        "wrong_color_key": 1.0,
        "inactive_switch": 2.0,
        "decoy_door": 2.0,
        "distractor_chain": 3.0,
    }
    return float(sum(weights.get(d.type, 1.0) for d in spec.distractors))


def _partial_observability(spec: TaskSpecification) -> float:
    mapping = {"full": 0.0, "view_cone": 1.0, "fog_of_war": 2.0}
    return mapping.get(spec.rules.observability, 0.0)


def _irreversibility(spec: TaskSpecification) -> float:
    score = 0.0
    if spec.rules.key_consumption:
        score += float(len(spec.mechanisms.doors))
    score += float(sum(1 for switch in spec.mechanisms.switches if switch.switch_type == "one_shot"))
    score += float(sum(1 for tp in spec.mechanisms.teleporters if not tp.bidirectional))
    return score


def compute_12d_score(
    spec: TaskSpecification,
    solver_output: DifficultyReport | None = None,
    weights: list[float] | None = None,
) -> ScoredDifficulty:
    """Compute the 12-dimension score from a task spec and solver output."""
    validator = TaskValidator(spec)
    is_beatable, solution, message = validator.validate()
    if solver_output is None:
        from .task_validator import compute_difficulty

        solver_output = compute_difficulty(spec)

    fragility = validator.compute_fragility()
    fragility_value = 0.0 if fragility.min_steps_to_break == -1 else 1.0 / fragility.min_steps_to_break

    width, height = spec.maze.dimensions
    grid_size = float(width * height)
    wall_density = float(len(spec.maze.walls) / grid_size) if grid_size else 0.0

    dimensions = [
        float(solver_output.optimal_steps),
        float(solver_output.states_explored),
        float(solver_output.backtrack_count if hasattr(solver_output, "backtrack_count") else _count_backtracking(solution)),
        fragility_value,
        float(spec.dependency_chain.depth if spec.dependency_chain is not None else solver_output.dependency_depth),
        _dependency_variety(spec),
        float(len(spec.distractors or [])),
        _distractor_quality(spec),
        grid_size,
        wall_density,
        _partial_observability(spec),
        _irreversibility(spec),
    ]

    weight_vector = weights or [1.0] * len(DIMENSION_NAMES)
    composite = float(sum(d * w for d, w in zip(dimensions, weight_vector)))
    return ScoredDifficulty(
        dimensions=dimensions,
        composite=composite,
        weights=weight_vector,
    )
