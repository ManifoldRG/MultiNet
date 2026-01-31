"""
MiniGrid DataLoader for GenESIS Evaluation

Provides PyTorch Dataset and DataLoader for MiniGrid gridworld tasks.
"""

from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import sys

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "v1_1"))

from definitions.minigrid import MiniGridDefinitions


class MiniGridDataset(Dataset):
    """
    PyTorch Dataset for MiniGrid gridworld tasks.

    Loads task specifications and generates observations on-the-fly
    by running episodes with the MiniGrid backend.
    """

    def __init__(
        self,
        task_files: List[str],
        dataset_name: str = "minigrid",
        by_episode: bool = False,
        max_steps_per_episode: Optional[int] = None,
        render_mode: str = "rgb_array",
    ):
        """
        Initialize the MiniGrid dataset.

        Args:
            task_files: List of paths to task JSON files
            dataset_name: Name for this dataset (e.g., "tier1", "tier2")
            by_episode: If True, each item is a full episode; if False, each item is a step
            max_steps_per_episode: Optional limit on steps per episode
            render_mode: Rendering mode for observations
        """
        self.task_files = task_files
        self.dataset_name = dataset_name
        self.by_episode = by_episode
        self.max_steps_per_episode = max_steps_per_episode
        self.render_mode = render_mode

        self._action_stats = None
        self._episodes_cache = {}
        self._step_index = []  # (task_idx, step_idx) for step-level access

        # Pre-compute step index if needed
        if not by_episode:
            self._build_step_index()

    def _build_step_index(self):
        """Build index mapping flat indices to (task, step) pairs."""
        for task_idx, task_file in enumerate(self.task_files):
            # Load task to get max_steps
            spec = self._load_task_spec(task_file)
            max_steps = spec.get("max_steps", 100)
            if self.max_steps_per_episode:
                max_steps = min(max_steps, self.max_steps_per_episode)

            for step_idx in range(max_steps):
                self._step_index.append((task_idx, step_idx))

    def _load_task_spec(self, path: str) -> dict:
        """Load task specification from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        if "TaskSpecification" in data:
            return data["TaskSpecification"]
        return data

    def _generate_episode(self, task_idx: int) -> List[Dict[str, Any]]:
        """
        Generate episode data by running the task.

        Args:
            task_idx: Index of the task file

        Returns:
            List of step data dictionaries
        """
        if task_idx in self._episodes_cache:
            return self._episodes_cache[task_idx]

        # Import here to avoid circular imports
        from v1_1.minigrid.task_spec import TaskSpecification
        from v1_1.minigrid.backends.minigrid_backend import MiniGridBackend

        # Load task specification
        spec_dict = self._load_task_spec(self.task_files[task_idx])
        spec = TaskSpecification.from_dict(spec_dict)

        # Create backend and run episode with random policy
        backend = MiniGridBackend(render_mode=self.render_mode)
        backend.configure(spec)

        obs, state, info = backend.reset(seed=spec.seed)
        mission = backend.get_mission_text()

        episode_data = []
        step = 0
        terminated = False
        truncated = False

        max_steps = spec.max_steps
        if self.max_steps_per_episode:
            max_steps = min(max_steps, self.max_steps_per_episode)

        while not terminated and not truncated and step < max_steps:
            # Random action for data generation
            action = np.random.randint(0, 7)

            # Get observation before action
            rgb_obs = backend.render()

            # Execute action
            next_obs, reward, terminated, truncated, next_state, _ = backend.step(action)

            # Determine tier/env name for text observation
            tier_name = f"tier{spec.difficulty_tier}"
            env_names = list(MiniGridDefinitions.DESCRIPTIONS.get(tier_name, {}).keys())
            text_obs = env_names[0] if env_names else "navigate to the goal"

            # Store step data
            step_data = {
                "text_observation": text_obs,
                "image_observation": rgb_obs.astype(np.uint8),
                "action": np.array([action], dtype=np.int64),
                "reward": reward,
                "is_last": terminated or truncated,
                "mission": mission,
                "task_id": spec.task_id,
                "tier": spec.difficulty_tier,
                "agent_position": list(state.agent_position),
                "agent_direction": state.agent_direction,
            }

            episode_data.append(step_data)
            obs = next_obs
            state = next_state
            step += 1

        backend.close()

        # Cache the episode
        self._episodes_cache[task_idx] = episode_data

        # Update action stats
        if self._action_stats is None and episode_data:
            self._action_stats = {
                "size": episode_data[0]["action"].shape,
                "min": 0,
                "max": 6,
                "mean": 3.0,
            }

        return episode_data

    @property
    def action_stats(self):
        """Get action space statistics."""
        if self._action_stats is None:
            self._action_stats = {
                "size": (1,),  # Single discrete action
                "min": 0,
                "max": 6,
                "mean": 3.0,
            }
        return self._action_stats

    def __len__(self) -> int:
        if self.by_episode:
            return len(self.task_files)
        return len(self._step_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.by_episode:
            # Return full episode
            episode = self._generate_episode(idx)
            return self._process_episode(episode)
        else:
            # Return single step
            task_idx, step_idx = self._step_index[idx]
            episode = self._generate_episode(task_idx)
            if step_idx < len(episode):
                return episode[step_idx]
            else:
                # Return last step if index is beyond episode length
                return episode[-1]

    def _process_episode(self, episode: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process episode into batched format.

        Args:
            episode: List of step dictionaries

        Returns:
            Dictionary with lists of values per key
        """
        result = defaultdict(list)
        for step in episode:
            for key, value in step.items():
                result[key].append(value)
        return dict(result)


class MiniGridPrecomputedDataset(Dataset):
    """
    Dataset for pre-generated MiniGrid observations.

    Uses saved numpy arrays and metadata instead of running episodes live.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "minigrid",
        by_episode: bool = False,
    ):
        """
        Initialize from pre-computed data directory.

        Args:
            data_dir: Directory containing observation files and metadata
            dataset_name: Name for this dataset
            by_episode: If True, group by episode
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.by_episode = by_episode

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"samples": []}

        self.samples = self.metadata.get("samples", [])
        self._action_stats = {
            "size": (1,),
            "min": 0,
            "max": 6,
            "mean": 3.0,
        }

    @property
    def action_stats(self):
        return self._action_stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load observation image
        img_path = self.data_dir / sample.get("image_path", f"obs_{idx}.npy")
        if img_path.exists():
            image_obs = np.load(img_path)
        else:
            image_obs = np.zeros((64, 64, 3), dtype=np.uint8)

        return {
            "text_observation": sample.get("mission", "navigate to the goal"),
            "image_observation": image_obs,
            "action": np.array([sample.get("action", 0)], dtype=np.int64),
            "reward": sample.get("reward", 0.0),
            "is_last": sample.get("is_last", False),
            "task_id": sample.get("task_id", "unknown"),
            "tier": sample.get("tier", 1),
        }


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Custom collate function for DataLoader."""
    result = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    return dict(result)


def get_minigrid_dataloader(
    task_files: List[str],
    batch_size: int,
    dataset_name: str = "minigrid",
    num_workers: int = 0,
    by_episode: bool = False,
) -> tuple:
    """
    Create MiniGrid dataset and dataloader.

    Args:
        task_files: List of task JSON file paths
        batch_size: Batch size
        dataset_name: Dataset name
        num_workers: Number of data loading workers
        by_episode: Whether to load by episode

    Returns:
        Tuple of (dataset, dataloader)
    """
    dataset = MiniGridDataset(
        task_files=task_files,
        dataset_name=dataset_name,
        by_episode=by_episode,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    return dataset, dataloader


def get_minigrid_precomputed_dataloader(
    data_dir: str,
    batch_size: int,
    dataset_name: str = "minigrid",
    num_workers: int = 0,
) -> tuple:
    """
    Create dataloader from pre-computed observations.

    Args:
        data_dir: Directory with saved observations
        batch_size: Batch size
        dataset_name: Dataset name
        num_workers: Number of workers

    Returns:
        Tuple of (dataset, dataloader)
    """
    dataset = MiniGridPrecomputedDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    return dataset, dataloader
