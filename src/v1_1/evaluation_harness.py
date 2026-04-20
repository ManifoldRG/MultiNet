"""
Evaluation Harness for MultiNet v1.1

Wraps GridRunner + ModelInterface to evaluate models on MiniGrid tasks.
Handles conversion between GridRunner's callback interface and ModelInterface.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from .model_interface import ModelInterface, ModelInput, ModelOutput
    from .gridworld.runner.grid_runner import GridRunner, EpisodeResult
    from .gridworld.backends.base import AbstractGridBackend, GridState
    from .gridworld.backends.minigrid_backend import MiniGridBackend
    from .gridworld.task_spec import TaskSpecification
    from .gridworld.actions import ACTION_NAMES, ACTION_DESCRIPTIONS
    from .gridworld.task_validator import compute_difficulty
    from .gridworld.scoring import compute_12d_score
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput
    from gridworld.runner.grid_runner import GridRunner, EpisodeResult
    from gridworld.backends.base import AbstractGridBackend, GridState
    from gridworld.backends.minigrid_backend import MiniGridBackend
    from gridworld.task_spec import TaskSpecification
    from gridworld.actions import ACTION_NAMES, ACTION_DESCRIPTIONS
    from gridworld.task_validator import compute_difficulty
    from gridworld.scoring import compute_12d_score


def _json_default(value):
    """Convert NumPy scalars to native Python types for JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


@dataclass
class TierMetrics:
    """Aggregate metrics for a tier of tasks."""
    tier: int
    num_tasks: int
    num_success: int
    success_rate: float
    avg_steps: float
    avg_reward: float
    results: list[EpisodeResult] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "num_tasks": self.num_tasks,
            "num_success": self.num_success,
            "success_rate": self.success_rate,
            "avg_steps": self.avg_steps,
            "avg_reward": self.avg_reward,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result across all tiers."""
    model_name: str
    tier_metrics: dict[int, TierMetrics]
    overall_success_rate: float
    overall_avg_steps: float
    overall_avg_reward: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "tier_metrics": {k: v.to_dict() for k, v in self.tier_metrics.items()},
            "overall_success_rate": self.overall_success_rate,
            "overall_avg_steps": self.overall_avg_steps,
            "overall_avg_reward": self.overall_avg_reward,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)


@dataclass
class TaskBenchmarkResult:
    """Per-task benchmark metrics with point-based scoring."""
    task_id: str
    success: bool
    steps_taken: int
    optimal_steps: int
    optimality_ratio: float | None
    available_points: float
    points_earned: float
    composite_score: float
    difficulty_dimensions: list[float]
    episode: EpisodeResult = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "steps_taken": self.steps_taken,
            "optimal_steps": self.optimal_steps,
            "optimality_ratio": self.optimality_ratio,
            "available_points": self.available_points,
            "points_earned": self.points_earned,
            "composite_score": self.composite_score,
            "difficulty_dimensions": self.difficulty_dimensions,
            "episode": self.episode.to_dict(),
        }


@dataclass
class BenchmarkEvaluationResult:
    """Aggregate metrics for a named benchmark set such as validation_10."""
    benchmark_name: str
    model_name: str
    num_tasks: int
    num_success: int
    success_rate: float
    total_available_points: float
    total_points_earned: float
    point_rate: float
    avg_optimality_ratio: float
    task_results: list[TaskBenchmarkResult] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "num_tasks": self.num_tasks,
            "num_success": self.num_success,
            "success_rate": self.success_rate,
            "total_available_points": self.total_available_points,
            "total_points_earned": self.total_points_earned,
            "point_rate": self.point_rate,
            "avg_optimality_ratio": self.avg_optimality_ratio,
            "task_results": [result.to_dict() for result in self.task_results],
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)


class EvaluationHarness:
    """
    Evaluation harness that bridges ModelInterface with GridRunner.

    Usage:
        harness = EvaluationHarness(model)
        result = harness.evaluate_task(task_spec, seed=42)
        tier_result = harness.evaluate_tier(tier=1, task_dir="gridworld/tasks")
        full_result = harness.evaluate_all(task_dir="gridworld/tasks")
    """

    def __init__(
        self,
        model: ModelInterface,
        backend: Optional[AbstractGridBackend] = None,
        render_mode: str = "rgb_array",
        history_images: int = 2,
        history_text: bool = True,
        history_text_window: int = 3,
    ):
        self.model = model
        self.history_images = history_images
        self.history_text = history_text
        self.history_text_window = history_text_window
        self.runner = GridRunner(
            backend=backend or MiniGridBackend(render_mode=render_mode),
            render_mode=render_mode,
        )

    def _make_policy_fn(self):
        """Create a policy function bridging GridRunner to ModelInterface."""
        step_counter = [0]
        recent_observations: list[np.ndarray] = []
        recent_summaries: list[str] = []
        previous_action = [None]
        previous_state = [None]

        def classify_action_result(prev_state: GridState, curr_state: GridState, action: int) -> str:
            action_name = ACTION_NAMES.get(action, str(action))
            if action in (0, 1):
                if curr_state.agent_direction != prev_state.agent_direction:
                    return f"ok: {action_name} changed facing direction"
                return f"error: {action_name} had no effect"

            if action == 2:
                if curr_state.agent_position != prev_state.agent_position:
                    return "ok: move_forward changed position"
                blocker = describe_forward_blocker(prev_state)
                if blocker:
                    return f"error: cannot move into {blocker}"
                return "error: move_forward did not change position"

            if action == 3:
                if curr_state.agent_carrying != prev_state.agent_carrying:
                    return f"ok: pickup now carrying {curr_state.agent_carrying}"
                return "error: pickup had no effect"

            if action == 5:
                if (
                    curr_state.open_doors != prev_state.open_doors
                    or curr_state.open_gates != prev_state.open_gates
                    or curr_state.active_switches != prev_state.active_switches
                ):
                    return "ok: toggle changed environment state"
                return "error: toggle had no effect"

            if action == 4:
                if curr_state.agent_carrying != prev_state.agent_carrying:
                    return "ok: drop changed carrying state"
                return "error: drop had no effect"

            return f"ok: {action_name}"

        def describe_forward_blocker(prev_state: GridState) -> str | None:
            if self.runner.backend.task_spec is None:
                return None
            x, y = prev_state.agent_position
            direction = prev_state.agent_direction
            dx, dy = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}.get(direction, (0, 0))
            target = (x + dx, y + dy)
            spec = self.runner.backend.task_spec
            width, height = spec.maze.dimensions
            if not (0 <= target[0] < width and 0 <= target[1] < height):
                return "boundary"
            wall_positions = {(wall.x, wall.y) for wall in spec.maze.walls}
            if target in wall_positions or target[0] in {0, width - 1} or target[1] in {0, height - 1}:
                return "wall"
            for door in spec.mechanisms.doors:
                if door.position.to_tuple() == target and door.id not in prev_state.open_doors:
                    return f"{door.requires_key} door"
            for gate in spec.mechanisms.gates:
                if gate.position.to_tuple() == target and gate.id not in prev_state.open_gates:
                    return "closed gate"
            for block_id, pos in prev_state.block_positions.items():
                if tuple(pos) == target:
                    return f"block {block_id}"
            return None

        def policy_fn(obs: np.ndarray, state: GridState, mission: str):
            step_counter[0] += 1
            if previous_action[0] is not None and previous_state[0] is not None:
                result_line = (
                    f"step {step_counter[0] - 1}: action={ACTION_NAMES.get(previous_action[0], previous_action[0])}, "
                    f"result={classify_action_result(previous_state[0], state, previous_action[0])}, "
                    f"position={state.agent_position}, agent_direction={state.agent_direction}"
                )
                if not recent_summaries or recent_summaries[-1] != result_line:
                    recent_summaries.append(result_line)
            prior_images = []
            if self.history_images > 0:
                prior_images = [frame.copy() for frame in recent_observations[-self.history_images:]]
            additional_context = None
            if self.history_text and recent_summaries:
                additional_context = "Recent steps:\n" + "\n".join(
                    recent_summaries[-self.history_text_window:]
                )
            model_input = ModelInput(
                image=obs if isinstance(obs, np.ndarray) and obs.ndim == 3 else
                      obs["image"] if isinstance(obs, dict) and "image" in obs else
                      np.zeros((64, 64, 3), dtype=np.uint8),
                text_prompt=mission,
                action_space=ACTION_NAMES,
                step_number=step_counter[0],
                max_steps=state.max_steps,
                additional_context=additional_context,
                prior_images=prior_images,
            )
            output = self.model.predict(model_input)
            policy_info = {
                "model_confidence": output.confidence,
                "model_reasoning": output.reasoning,
                "model_raw_output": output.raw_output,
            }
            if output.reasoning and output.reasoning.startswith("API error:"):
                policy_info["model_error"] = output.reasoning
            recent_observations.append(model_input.image.copy())
            previous_action[0] = output.action
            previous_state[0] = GridState.from_dict(state.to_dict())
            return output.action, policy_info

        return policy_fn

    def evaluate_task(
        self,
        task_spec: TaskSpecification,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> EpisodeResult:
        """
        Evaluate the model on a single task.

        Args:
            task_spec: Task to evaluate
            seed: Random seed override
            verbose: Print step-by-step info

        Returns:
            EpisodeResult with trajectory and metrics
        """
        policy_fn = self._make_policy_fn()
        return self.runner.run_episode(
            task_spec=task_spec,
            policy_fn=policy_fn,
            seed=seed,
            verbose=verbose,
        )

    def evaluate_tier(
        self,
        tier: int,
        task_dir: str = "gridworld/tasks",
        verbose: bool = False,
    ) -> TierMetrics:
        """
        Evaluate the model on all tasks in a tier.

        Args:
            tier: Difficulty tier (1-5)
            task_dir: Base directory containing tier subdirectories
            verbose: Print progress

        Returns:
            TierMetrics with aggregate results
        """
        tier_path = Path(task_dir) / f"tier{tier}"
        if not tier_path.exists():
            raise FileNotFoundError(f"Tier directory not found: {tier_path}")

        task_files = sorted(tier_path.glob("*.json"))
        if not task_files:
            raise FileNotFoundError(f"No task files found in {tier_path}")

        results = []
        for task_file in task_files:
            spec = TaskSpecification.from_json(str(task_file))
            if verbose:
                print(f"  Evaluating {spec.task_id}...")
            result = self.evaluate_task(spec, verbose=verbose)
            results.append(result)

        return self._compute_tier_metrics(tier, results)

    def evaluate_all(
        self,
        task_dir: str = "gridworld/tasks",
        tiers: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate the model on all tiers.

        Args:
            task_dir: Base directory containing tier subdirectories
            tiers: List of tiers to evaluate (default: 1-5)
            verbose: Print progress

        Returns:
            EvaluationResult with per-tier and overall metrics
        """
        if tiers is None:
            tiers = [1, 2, 3, 4, 5]

        tier_metrics = {}
        all_results = []

        for tier in tiers:
            tier_path = Path(task_dir) / f"tier{tier}"
            if not tier_path.exists():
                if verbose:
                    print(f"Skipping tier {tier} (directory not found)")
                continue

            if verbose:
                print(f"\n=== Tier {tier} ===")

            metrics = self.evaluate_tier(tier, task_dir, verbose=verbose)
            tier_metrics[tier] = metrics
            all_results.extend(metrics.results)

        # Compute overall metrics
        if all_results:
            overall_success = sum(1 for r in all_results if r.success) / len(all_results)
            overall_steps = sum(r.steps_taken for r in all_results) / len(all_results)
            overall_reward = sum(r.total_reward for r in all_results) / len(all_results)
        else:
            overall_success = 0.0
            overall_steps = 0.0
            overall_reward = 0.0

        return EvaluationResult(
            model_name=self.model.model_name,
            tier_metrics=tier_metrics,
            overall_success_rate=overall_success,
            overall_avg_steps=overall_steps,
            overall_avg_reward=overall_reward,
        )

    def evaluate_task_set(
        self,
        task_specs: list[TaskSpecification],
        benchmark_name: str = "custom",
        verbose: bool = False,
    ) -> BenchmarkEvaluationResult:
        """
        Evaluate a named benchmark set and compute point-based metrics.

        Point earning uses the authored difficulty composite as the available
        budget and scales it by efficiency on successful runs.
        """
        task_results: list[TaskBenchmarkResult] = []

        for spec in task_specs:
            episode = self.evaluate_task(spec, seed=spec.seed, verbose=verbose)
            difficulty = compute_difficulty(spec)
            score = compute_12d_score(spec, solver_output=difficulty)

            optimal_steps = difficulty.optimal_steps
            optimality_ratio = None
            if episode.success and optimal_steps > 0:
                optimality_ratio = episode.steps_taken / optimal_steps

            efficiency = 0.0
            if episode.success and optimal_steps > 0 and episode.steps_taken > 0:
                efficiency = min(1.0, optimal_steps / episode.steps_taken)

            available_points = score.composite
            points_earned = available_points * efficiency

            task_results.append(TaskBenchmarkResult(
                task_id=spec.task_id,
                success=episode.success,
                steps_taken=episode.steps_taken,
                optimal_steps=optimal_steps,
                optimality_ratio=optimality_ratio,
                available_points=available_points,
                points_earned=points_earned,
                composite_score=score.composite,
                difficulty_dimensions=score.dimensions,
                episode=episode,
            ))

        num_tasks = len(task_results)
        num_success = sum(1 for result in task_results if result.success)
        total_available_points = sum(result.available_points for result in task_results)
        total_points_earned = sum(result.points_earned for result in task_results)
        optimality_values = [result.optimality_ratio for result in task_results if result.optimality_ratio is not None]

        return BenchmarkEvaluationResult(
            benchmark_name=benchmark_name,
            model_name=self.model.model_name,
            num_tasks=num_tasks,
            num_success=num_success,
            success_rate=(num_success / num_tasks) if num_tasks else 0.0,
            total_available_points=total_available_points,
            total_points_earned=total_points_earned,
            point_rate=(total_points_earned / total_available_points) if total_available_points else 0.0,
            avg_optimality_ratio=(sum(optimality_values) / len(optimality_values)) if optimality_values else 0.0,
            task_results=task_results,
        )

    def evaluate_task_dir(
        self,
        task_dir: str,
        benchmark_name: str | None = None,
        verbose: bool = False,
    ) -> BenchmarkEvaluationResult:
        """Evaluate every JSON task file in a directory as a named benchmark set."""
        task_path = Path(task_dir)
        task_files = sorted(task_path.glob("*.json"))
        if not task_files:
            raise FileNotFoundError(f"No task files found in {task_path}")

        specs = [TaskSpecification.from_json(str(task_file)) for task_file in task_files]
        return self.evaluate_task_set(
            specs,
            benchmark_name=benchmark_name or task_path.name,
            verbose=verbose,
        )

    def _compute_tier_metrics(self, tier: int, results: list[EpisodeResult]) -> TierMetrics:
        """Compute aggregate metrics for a set of episode results."""
        num_tasks = len(results)
        num_success = sum(1 for r in results if r.success)
        success_rate = num_success / num_tasks if num_tasks > 0 else 0.0
        avg_steps = sum(r.steps_taken for r in results) / num_tasks if num_tasks > 0 else 0.0
        avg_reward = sum(r.total_reward for r in results) / num_tasks if num_tasks > 0 else 0.0

        return TierMetrics(
            tier=tier,
            num_tasks=num_tasks,
            num_success=num_success,
            success_rate=success_rate,
            avg_steps=avg_steps,
            avg_reward=avg_reward,
            results=results,
        )

    def close(self):
        """Clean up resources."""
        self.model.teardown()
        self.runner.close()
