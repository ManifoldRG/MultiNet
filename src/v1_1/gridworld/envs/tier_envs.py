"""
Pre-configured Environments by Difficulty Tier

Provides factory functions to create environments for each tier.
Also supports loading standard MiniGrid environments as fallback.
"""

from pathlib import Path
from typing import Optional, List, Dict
import json
import glob

from ..task_spec import TaskSpecification
from ..task_parser import TaskParser, load_task_from_file
from ..backends.minigrid_backend import MiniGridBackend


# Base path for task files
TASKS_DIR = Path(__file__).parent.parent / "tasks"


def _load_tasks_from_dir(tier_dir: Path) -> List[TaskSpecification]:
    """Load all task specifications from a tier directory."""
    tasks = []
    if tier_dir.exists():
        for json_file in sorted(tier_dir.glob("*.json")):
            try:
                spec = TaskSpecification.from_json(str(json_file))
                tasks.append(spec)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    return tasks


def get_tier1_envs(render_mode: str = "rgb_array") -> List[tuple]:
    """
    Get Tier 1 (Navigation) environments.

    Returns:
        List of (task_spec, env) tuples
    """
    tier_dir = TASKS_DIR / "tier1"
    tasks = _load_tasks_from_dir(tier_dir)

    parser = TaskParser(render_mode=render_mode)
    envs = []
    for task in tasks:
        try:
            env = parser.parse(task)
            envs.append((task, env))
        except Exception as e:
            print(f"Warning: Failed to create env for {task.task_id}: {e}")

    return envs


def get_tier2_envs(render_mode: str = "rgb_array") -> List[tuple]:
    """
    Get Tier 2 (Linear Dependencies - Keys/Doors) environments.

    Returns:
        List of (task_spec, env) tuples
    """
    tier_dir = TASKS_DIR / "tier2"
    tasks = _load_tasks_from_dir(tier_dir)

    parser = TaskParser(render_mode=render_mode)
    envs = []
    for task in tasks:
        try:
            env = parser.parse(task)
            envs.append((task, env))
        except Exception as e:
            print(f"Warning: Failed to create env for {task.task_id}: {e}")

    return envs


def get_tier3_envs(render_mode: str = "rgb_array") -> List[tuple]:
    """
    Get Tier 3 (Multi-Mechanism - Keys/Doors/Switches/Gates) environments.

    Returns:
        List of (task_spec, env) tuples
    """
    tier_dir = TASKS_DIR / "tier3"
    tasks = _load_tasks_from_dir(tier_dir)

    parser = TaskParser(render_mode=render_mode)
    envs = []
    for task in tasks:
        try:
            env = parser.parse(task)
            envs.append((task, env))
        except Exception as e:
            print(f"Warning: Failed to create env for {task.task_id}: {e}")

    return envs


def get_tier4_envs(render_mode: str = "rgb_array") -> List[tuple]:
    """
    Get Tier 4 (Irreversibility - Pushable blocks) environments.

    Returns:
        List of (task_spec, env) tuples
    """
    tier_dir = TASKS_DIR / "tier4"
    tasks = _load_tasks_from_dir(tier_dir)

    parser = TaskParser(render_mode=render_mode)
    envs = []
    for task in tasks:
        try:
            env = parser.parse(task)
            envs.append((task, env))
        except Exception as e:
            print(f"Warning: Failed to create env for {task.task_id}: {e}")

    return envs


def get_tier5_envs(render_mode: str = "rgb_array") -> List[tuple]:
    """
    Get Tier 5 (Hidden Information) environments.

    Returns:
        List of (task_spec, env) tuples
    """
    tier_dir = TASKS_DIR / "tier5"
    tasks = _load_tasks_from_dir(tier_dir)

    parser = TaskParser(render_mode=render_mode)
    envs = []
    for task in tasks:
        try:
            env = parser.parse(task)
            envs.append((task, env))
        except Exception as e:
            print(f"Warning: Failed to create env for {task.task_id}: {e}")

    return envs


def get_all_envs(render_mode: str = "rgb_array") -> Dict[str, List[tuple]]:
    """
    Get all environments organized by tier.

    Returns:
        Dictionary mapping tier names to lists of (task_spec, env) tuples
    """
    return {
        "tier1": get_tier1_envs(render_mode),
        "tier2": get_tier2_envs(render_mode),
        "tier3": get_tier3_envs(render_mode),
        "tier4": get_tier4_envs(render_mode),
        "tier5": get_tier5_envs(render_mode),
    }


def get_env_by_id(
    task_id: str,
    render_mode: str = "rgb_array"
) -> Optional[tuple]:
    """
    Get a specific environment by task ID.

    Args:
        task_id: The task ID to find
        render_mode: Rendering mode for the environment

    Returns:
        (task_spec, env) tuple or None if not found
    """
    # Search all tier directories
    for tier_num in range(1, 6):
        tier_dir = TASKS_DIR / f"tier{tier_num}"
        if tier_dir.exists():
            for json_file in tier_dir.glob("*.json"):
                try:
                    spec = TaskSpecification.from_json(str(json_file))
                    if spec.task_id == task_id:
                        parser = TaskParser(render_mode=render_mode)
                        env = parser.parse(spec)
                        return (spec, env)
                except Exception:
                    continue

    return None


def list_available_envs() -> Dict[str, List[str]]:
    """
    List all available task IDs organized by tier.

    Returns:
        Dictionary mapping tier names to lists of task IDs
    """
    result = {}
    for tier_num in range(1, 6):
        tier_name = f"tier{tier_num}"
        tier_dir = TASKS_DIR / tier_name
        task_ids = []

        if tier_dir.exists():
            for json_file in sorted(tier_dir.glob("*.json")):
                try:
                    spec = TaskSpecification.from_json(str(json_file))
                    task_ids.append(spec.task_id)
                except Exception:
                    task_ids.append(json_file.stem)

        result[tier_name] = task_ids

    return result


def get_standard_minigrid_env(env_name: str, render_mode: str = "rgb_array"):
    """
    Get a standard MiniGrid environment by name.

    This provides access to built-in MiniGrid environments as fallback.

    Args:
        env_name: Standard MiniGrid environment name (e.g., "MiniGrid-Empty-8x8-v0")
        render_mode: Rendering mode

    Returns:
        Gymnasium environment
    """
    import gymnasium as gym
    return gym.make(env_name, render_mode=render_mode)


# Mapping of tiers to standard MiniGrid environments (as fallback)
STANDARD_MINIGRID_ENVS = {
    "tier1": [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Empty-16x16-v0",
        "MiniGrid-FourRooms-v0",
    ],
    "tier2": [
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-DoorKey-16x16-v0",
    ],
    "tier3": [
        "MiniGrid-LockedRoom-v0",
        "MiniGrid-KeyCorridorS3R1-v0",
        "MiniGrid-KeyCorridorS3R2-v0",
        "MiniGrid-KeyCorridorS3R3-v0",
    ],
    "tier4": [
        "MiniGrid-BlockedUnlockPickup-v0",
    ],
    "tier5": [
        "MiniGrid-MemoryS7-v0",
        "MiniGrid-MemoryS9-v0",
        "MiniGrid-RedBlueDoors-8x8-v0",
    ],
}
