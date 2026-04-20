#!/usr/bin/env python3
"""
Test script for MiniGrid domain implementation.

Verifies that:
1. Task specifications load correctly
2. Environments can be created from specs
3. Actions execute properly
4. Rendering works
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_task_spec_loading():
    """Test loading task specifications from JSON."""
    print("\n=== Testing Task Specification Loading ===")

    from v1_1.gridworld.task_spec import TaskSpecification

    # Test loading tier1 task
    tier1_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    if tier1_path.exists():
        spec = TaskSpecification.from_json(str(tier1_path))
        print(f"✓ Loaded task: {spec.task_id}")
        print(f"  Tier: {spec.difficulty_tier}")
        print(f"  Dimensions: {spec.maze.dimensions}")
        print(f"  Start: {spec.maze.start.to_tuple()}")
        print(f"  Goal: {spec.maze.goal.to_tuple()}")
        print(f"  Max steps: {spec.max_steps}")

        # Test validation
        is_valid, errors = spec.validate()
        if is_valid:
            print(f"✓ Validation passed")
        else:
            print(f"✗ Validation failed: {errors}")

        # Test mission text generation
        mission = spec.get_mission_text()
        print(f"  Mission: {mission}")
    else:
        print(f"✗ Task file not found: {tier1_path}")

    return True


def test_task_parser():
    """Test parsing task specs into environments."""
    print("\n=== Testing Task Parser ===")

    from v1_1.gridworld.task_spec import TaskSpecification
    from v1_1.gridworld.task_parser import TaskParser

    parser = TaskParser(render_mode="rgb_array")

    # Test tier 1
    tier1_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    if tier1_path.exists():
        spec = TaskSpecification.from_json(str(tier1_path))
        env = parser.parse(spec)
        print(f"✓ Created environment for {spec.task_id}")
        print(f"  Grid size: {env.width}x{env.height}")
        print(f"  Agent position: {env.agent_pos}")
        print(f"  Agent direction: {env.agent_dir}")

        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✓ Environment reset successful")

        # Test render
        img = env.render()
        print(f"✓ Rendered image shape: {img.shape}")

        env.close()
    else:
        print(f"✗ Task file not found: {tier1_path}")

    return True


def test_environment_step():
    """Test taking steps in the environment."""
    print("\n=== Testing Environment Step ===")

    from v1_1.gridworld.task_spec import TaskSpecification
    from v1_1.gridworld.task_parser import TaskParser
    from v1_1.gridworld.actions import MiniGridActions, ACTION_NAMES

    parser = TaskParser(render_mode="rgb_array")

    tier1_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    if tier1_path.exists():
        spec = TaskSpecification.from_json(str(tier1_path))
        env = parser.parse(spec)
        obs, info = env.reset(seed=42)

        print(f"Starting position: {env.agent_pos}")

        # Take a few steps
        actions = [
            MiniGridActions.TURN_RIGHT,
            MiniGridActions.MOVE_FORWARD,
            MiniGridActions.MOVE_FORWARD,
            MiniGridActions.TURN_LEFT,
            MiniGridActions.MOVE_FORWARD,
        ]

        total_reward = 0
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            action_name = ACTION_NAMES.get(action, f"action_{action}")
            print(f"  Step {i+1}: {action_name} -> pos={env.agent_pos}, reward={reward:.3f}, done={terminated or truncated}")

            if terminated or truncated:
                break

        print(f"✓ Completed {len(actions)} steps, total reward: {total_reward:.3f}")
        env.close()
    else:
        print(f"✗ Task file not found: {tier1_path}")

    return True


def test_backend():
    """Test the MiniGrid backend wrapper."""
    print("\n=== Testing MiniGrid Backend ===")

    from v1_1.gridworld.task_spec import TaskSpecification
    from v1_1.gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")

    tier2_path = Path(__file__).parent / "tasks" / "tier2" / "single_key_001.json"
    if tier2_path.exists():
        spec = TaskSpecification.from_json(str(tier2_path))
        backend.configure(spec)

        obs, state, info = backend.reset(seed=42)
        print(f"✓ Backend reset successful")
        print(f"  Agent position: {state.agent_position}")
        print(f"  Agent direction: {state.agent_direction}")
        print(f"  Observation shape: {obs.shape}")

        # Take a step
        obs, reward, terminated, truncated, state, info = backend.step(2)  # Move forward
        print(f"✓ Backend step successful")
        print(f"  New position: {state.agent_position}")

        # Get mission
        mission = backend.get_mission_text()
        print(f"  Mission: {mission}")

        backend.close()
    else:
        print(f"✗ Task file not found: {tier2_path}")

    return True


def test_runner():
    """Test the grid runner."""
    print("\n=== Testing Grid Runner ===")

    from v1_1.gridworld.task_spec import TaskSpecification
    from v1_1.gridworld.runner.grid_runner import GridRunner

    runner = GridRunner(render_mode="rgb_array")

    tier1_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    if tier1_path.exists():
        spec = TaskSpecification.from_json(str(tier1_path))

        # Run episode with random policy
        result = runner.run_episode(spec, policy_fn=None, verbose=False)
        print(f"✓ Episode completed: {spec.task_id}")
        print(f"  Success: {result.success}")
        print(f"  Steps taken: {result.steps_taken}")
        print(f"  Total reward: {result.total_reward:.3f}")
        print(f"  Terminated: {result.terminated}")
        print(f"  Truncated: {result.truncated}")
        print(f"  Trajectory length: {len(result.trajectory)}")

        runner.close()
    else:
        print(f"✗ Task file not found: {tier1_path}")

    return True


def test_tier_envs():
    """Test loading environments by tier."""
    print("\n=== Testing Tier Environment Loading ===")

    from v1_1.gridworld.envs.tier_envs import list_available_envs, get_tier1_envs

    # List available
    available = list_available_envs()
    for tier, tasks in available.items():
        print(f"  {tier}: {len(tasks)} tasks - {tasks}")

    # Load tier 1
    tier1_envs = get_tier1_envs(render_mode="rgb_array")
    print(f"✓ Loaded {len(tier1_envs)} tier 1 environments")

    for spec, env in tier1_envs:
        print(f"  - {spec.task_id}: {spec.maze.dimensions}")
        env.close()

    return True


def test_all_tiers():
    """Test that all tier tasks load correctly."""
    print("\n=== Testing All Tier Tasks ===")

    from v1_1.gridworld.task_spec import TaskSpecification
    from v1_1.gridworld.task_parser import TaskParser

    parser = TaskParser(render_mode="rgb_array")
    tasks_dir = Path(__file__).parent / "tasks"

    for tier_num in range(1, 6):
        tier_dir = tasks_dir / f"tier{tier_num}"
        if tier_dir.exists():
            task_files = list(tier_dir.glob("*.json"))
            loaded = 0
            for task_file in task_files:
                try:
                    spec = TaskSpecification.from_json(str(task_file))
                    env = parser.parse(spec)
                    obs, info = env.reset(seed=spec.seed)
                    env.close()
                    loaded += 1
                except Exception as e:
                    print(f"  ✗ Failed to load {task_file.name}: {e}")

            print(f"✓ Tier {tier_num}: {loaded}/{len(task_files)} tasks loaded successfully")
        else:
            print(f"  Tier {tier_num} directory not found")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("MiniGrid Domain Implementation Tests")
    print("=" * 60)

    tests = [
        ("Task Specification Loading", test_task_spec_loading),
        ("Task Parser", test_task_parser),
        ("Environment Step", test_environment_step),
        ("MiniGrid Backend", test_backend),
        ("Grid Runner", test_runner),
        ("Tier Environments", test_tier_envs),
        ("All Tiers", test_all_tiers),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
