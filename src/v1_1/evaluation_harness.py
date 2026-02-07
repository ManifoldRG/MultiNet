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
except ImportError:
    from model_interface import ModelInterface, ModelInput, ModelOutput
    from gridworld.runner.grid_runner import GridRunner, EpisodeResult
    from gridworld.backends.base import AbstractGridBackend, GridState
    from gridworld.backends.minigrid_backend import MiniGridBackend
    from gridworld.task_spec import TaskSpecification
    from gridworld.actions import ACTION_NAMES, ACTION_DESCRIPTIONS


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
            json.dump(self.to_dict(), f, indent=2)


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
    ):
        self.model = model
        self.runner = GridRunner(
            backend=backend or MiniGridBackend(render_mode=render_mode),
            render_mode=render_mode,
        )

    def _make_policy_fn(self):
        """Create a policy function bridging GridRunner to ModelInterface."""
        step_counter = [0]

        def policy_fn(obs: np.ndarray, state: GridState, mission: str) -> int:
            step_counter[0] += 1
            model_input = ModelInput(
                image=obs if isinstance(obs, np.ndarray) and obs.ndim == 3 else
                      obs["image"] if isinstance(obs, dict) and "image" in obs else
                      np.zeros((64, 64, 3), dtype=np.uint8),
                text_prompt=mission,
                action_space=ACTION_NAMES,
                step_number=step_counter[0],
                max_steps=state.max_steps,
            )
            output = self.model.predict(model_input)
            return output.action

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
