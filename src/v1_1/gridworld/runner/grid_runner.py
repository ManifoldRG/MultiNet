"""
Grid Runner for Episode Execution

Executes episodes in MiniGrid environments and collects trajectories
for evaluation with VLM/VLA models.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from pathlib import Path
import json
import numpy as np

from ..backends.base import AbstractGridBackend, GridState
from ..backends.minigrid_backend import MiniGridBackend
from ..task_spec import TaskSpecification
from ..actions import ACTION_NAMES


@dataclass
class Trajectory:
    """
    A single step in an episode trajectory.
    """
    step: int
    observation: np.ndarray  # RGB image
    action: int
    action_name: str
    reward: float
    state: GridState
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary (without image for serialization)."""
        return {
            "step": self.step,
            "action": self.action,
            "action_name": self.action_name,
            "reward": self.reward,
            "state": self.state.to_dict(),
            "info": self.info,
        }


@dataclass
class EpisodeResult:
    """
    Result of running an episode.
    """
    task_id: str
    success: bool
    total_reward: float
    steps_taken: int
    max_steps: int
    terminated: bool
    truncated: bool
    trajectory: list[Trajectory]
    final_state: GridState
    seed: int
    mission: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "total_reward": self.total_reward,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "trajectory": [t.to_dict() for t in self.trajectory],
            "final_state": self.final_state.to_dict(),
            "seed": self.seed,
            "mission": self.mission,
        }

    def save(self, path: str) -> None:
        """Save episode result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EpisodeResult":
        """Load episode result from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        # Note: observations not included in saved trajectories
        trajectory = [
            Trajectory(
                step=t["step"],
                observation=np.zeros((64, 64, 3), dtype=np.uint8),  # Placeholder
                action=t["action"],
                action_name=t["action_name"],
                reward=t["reward"],
                state=GridState.from_dict(t["state"]),
                info=t.get("info", {}),
            )
            for t in data["trajectory"]
        ]
        return cls(
            task_id=data["task_id"],
            success=data["success"],
            total_reward=data["total_reward"],
            steps_taken=data["steps_taken"],
            max_steps=data["max_steps"],
            terminated=data["terminated"],
            truncated=data["truncated"],
            trajectory=trajectory,
            final_state=GridState.from_dict(data["final_state"]),
            seed=data["seed"],
            mission=data["mission"],
        )


class GridRunner:
    """
    Episode runner for MiniGrid environments.

    Executes episodes using either:
    - A policy function (for VLM/VLA evaluation)
    - Random actions (for baseline)
    - Expert demonstrations (if available)
    """

    def __init__(
        self,
        backend: Optional[AbstractGridBackend] = None,
        render_mode: str = "rgb_array",
    ):
        """
        Initialize the runner.

        Args:
            backend: Grid backend to use (defaults to MiniGridBackend)
            render_mode: Rendering mode for observations
        """
        self.backend = backend or MiniGridBackend(render_mode=render_mode)
        self.render_mode = render_mode

    def run_episode(
        self,
        task_spec: TaskSpecification,
        policy_fn: Optional[Callable[[np.ndarray, GridState, str], Any]] = None,
        seed: Optional[int] = None,
        record_trajectory: bool = True,
        verbose: bool = False,
    ) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            task_spec: Task specification defining the puzzle
            policy_fn: Function that takes (observation, state, mission) and returns action.
                       If None, uses random policy.
            seed: Random seed (uses task_spec.seed if not provided)
            record_trajectory: Whether to record full trajectory
            verbose: Print step information

        Returns:
            EpisodeResult with episode outcomes and trajectory
        """
        # Configure backend
        self.backend.configure(task_spec)

        # Reset environment
        seed = seed or task_spec.seed
        obs, state, info = self.backend.reset(seed=seed)
        mission = self.backend.get_mission_text()

        # Initialize tracking
        trajectory = []
        total_reward = 0.0
        step = 0
        terminated = False
        truncated = False

        # Seed random number generator for deterministic random policy
        rng = np.random.RandomState(seed)

        if verbose:
            print(f"Starting episode: {task_spec.task_id}")
            print(f"Mission: {mission}")

        while not terminated and not truncated:
            # Get action from policy or random
            policy_info = {}
            if policy_fn is not None:
                policy_result = policy_fn(obs, state, mission)
                if isinstance(policy_result, tuple):
                    action = int(policy_result[0])
                    if len(policy_result) > 1 and isinstance(policy_result[1], dict):
                        policy_info = policy_result[1]
                else:
                    action = int(policy_result)
            else:
                # Random policy with explicit seed
                action = rng.randint(0, 7)

            # Execute action
            next_obs, reward, terminated, truncated, next_state, info = self.backend.step(action)
            if policy_info:
                info = {**info, **policy_info}
            total_reward += reward
            step += 1

            if verbose:
                action_name = ACTION_NAMES.get(action, f"action_{action}")
                print(f"  Step {step}: {action_name} -> reward={reward:.3f}, done={terminated or truncated}")

            # Record trajectory
            if record_trajectory:
                trajectory.append(Trajectory(
                    step=step,
                    observation=obs.copy(),
                    action=action,
                    action_name=ACTION_NAMES.get(action, f"action_{action}"),
                    reward=reward,
                    state=state,
                    info=info,
                ))

            # Update for next iteration
            obs = next_obs
            state = next_state

        # Determine success
        success = terminated and total_reward > 0

        if verbose:
            print(f"Episode complete: success={success}, steps={step}, reward={total_reward:.3f}")

        return EpisodeResult(
            task_id=task_spec.task_id,
            success=success,
            total_reward=total_reward,
            steps_taken=step,
            max_steps=task_spec.max_steps,
            terminated=terminated,
            truncated=truncated,
            trajectory=trajectory,
            final_state=state,
            seed=seed,
            mission=mission,
        )

    def run_batch(
        self,
        task_specs: list[TaskSpecification],
        policy_fn: Optional[Callable[[np.ndarray, GridState, str], int]] = None,
        verbose: bool = False,
    ) -> list[EpisodeResult]:
        """
        Run multiple episodes.

        Args:
            task_specs: List of task specifications
            policy_fn: Policy function (see run_episode)
            verbose: Print progress

        Returns:
            List of EpisodeResults
        """
        results = []
        for i, spec in enumerate(task_specs):
            if verbose:
                print(f"\n=== Task {i+1}/{len(task_specs)}: {spec.task_id} ===")
            result = self.run_episode(spec, policy_fn, verbose=verbose)
            results.append(result)
        return results

    def collect_demonstrations(
        self,
        task_spec: TaskSpecification,
        actions: list[int],
        seed: Optional[int] = None,
    ) -> EpisodeResult:
        """
        Execute a fixed sequence of actions to collect a demonstration.

        Args:
            task_spec: Task specification
            actions: List of actions to execute
            seed: Random seed

        Returns:
            EpisodeResult with the demonstration trajectory
        """
        def demo_policy(obs, state, mission, action_idx=[0]):
            if action_idx[0] < len(actions):
                action = actions[action_idx[0]]
                action_idx[0] += 1
                return action
            return 6  # Wait if no more actions

        return self.run_episode(task_spec, policy_fn=demo_policy, seed=seed)

    def generate_observation_dataset(
        self,
        task_specs: list[TaskSpecification],
        policy_fn: Optional[Callable] = None,
        output_dir: str = "observations",
        save_images: bool = True,
    ) -> list[dict]:
        """
        Generate a dataset of observations for evaluation.

        Args:
            task_specs: List of task specifications
            policy_fn: Policy to use (random if None)
            output_dir: Directory to save images
            save_images: Whether to save observation images

        Returns:
            List of observation records with metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        records = []
        for spec in task_specs:
            result = self.run_episode(spec, policy_fn, record_trajectory=True)

            for traj in result.trajectory:
                record = {
                    "task_id": spec.task_id,
                    "step": traj.step,
                    "action": traj.action,
                    "action_name": traj.action_name,
                    "reward": traj.reward,
                    "mission": result.mission,
                    "tier": spec.difficulty_tier,
                    "agent_position": list(traj.state.agent_position),
                    "agent_direction": traj.state.agent_direction,
                }

                if save_images:
                    img_name = f"{spec.task_id}_step{traj.step:04d}.npy"
                    img_path = output_path / img_name
                    np.save(img_path, traj.observation)
                    record["image_path"] = str(img_path)

                records.append(record)

        return records

    def close(self):
        """Clean up resources."""
        self.backend.close()
