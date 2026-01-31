#!/usr/bin/env python3
"""
MiniGrid Backend Demo

Demonstrates the MiniGridBackend (gymnasium-based) for standard square grid tasks.
Shows loading tasks, running episodes, using policies, and saving visualizations.

Usage:
    cd src/v1_1
    python minigrid/demo.py              # Run all demos
    python minigrid/demo.py --visual     # Save PNG images of each demo
    python minigrid/demo.py --play       # Interactive play mode
    python minigrid/demo.py --play --task tier2/single_key_001  # Play specific task
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure imports work from the v1_1 directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from minigrid.task_spec import TaskSpecification
from minigrid.backends import get_backend, MiniGridBackend
from minigrid.backends.base import GridState
from minigrid.runner.grid_runner import GridRunner
from minigrid.actions import MiniGridActions, ACTION_NAMES
from minigrid.envs.tier_envs import list_available_envs


def interactive_play(task_path: str = None):
    """
    Interactive play mode - control the agent with keyboard.

    Controls:
        Arrow Keys: Move/Turn (Up=forward, Left/Right=turn)
        Space: Pickup
        D: Drop
        T or Enter: Toggle (open door, activate switch)
        R: Reset episode
        Q or Escape: Quit
    """
    import pygame

    # Default to a tier 2 task for interesting gameplay
    if task_path is None:
        task_path = Path(__file__).parent / "tasks" / "tier2" / "single_key_001.json"
    else:
        # Handle relative paths like "tier2/single_key_001"
        if not Path(task_path).exists():
            task_path = Path(__file__).parent / "tasks" / f"{task_path}.json"

    spec = TaskSpecification.from_json(str(task_path))

    print("\n" + "=" * 60)
    print("Interactive Play Mode")
    print("=" * 60)
    print(f"\nTask: {spec.task_id}")
    print(f"Description: {spec.description}")
    print(f"\nControls:")
    print("  Arrow Up    : Move forward")
    print("  Arrow Left  : Turn left")
    print("  Arrow Right : Turn right")
    print("  Space       : Pickup")
    print("  D           : Drop")
    print("  T / Enter   : Toggle (doors, switches)")
    print("  R           : Reset")
    print("  Q / Escape  : Quit")
    print("\n" + "-" * 60)

    # Create backend with rgb_array mode (we'll display via pygame)
    backend = get_backend("minigrid", render_mode="rgb_array")
    backend.configure(spec)
    obs, state, info = backend.reset(seed=42)

    # Initialize pygame
    pygame.init()

    # Scale up for visibility
    scale = 2
    display_size = (obs.shape[1] * scale, obs.shape[0] * scale)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption(f"MiniGrid: {spec.task_id}")

    # Key mapping
    key_to_action = {
        pygame.K_UP: MiniGridActions.MOVE_FORWARD,
        pygame.K_LEFT: MiniGridActions.TURN_LEFT,
        pygame.K_RIGHT: MiniGridActions.TURN_RIGHT,
        pygame.K_SPACE: MiniGridActions.PICKUP,
        pygame.K_d: MiniGridActions.DROP,
        pygame.K_t: MiniGridActions.TOGGLE,
        pygame.K_RETURN: MiniGridActions.TOGGLE,
    }

    clock = pygame.time.Clock()
    running = True
    step_count = 0

    def render_frame():
        # Convert numpy array to pygame surface
        surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, display_size)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    def print_status():
        carrying = state.agent_carrying if state.agent_carrying else "nothing"
        print(f"  Step {step_count}: pos={state.agent_position}, carrying={carrying}")

    render_frame()
    print(f"\nStarting at {state.agent_position}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    obs, state, info = backend.reset(seed=42)
                    step_count = 0
                    render_frame()
                    print("\n--- Episode Reset ---")
                    print(f"Starting at {state.agent_position}")
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, terminated, truncated, state, info = backend.step(action)
                    step_count += 1
                    render_frame()
                    print_status()

                    if terminated:
                        print("\n*** GOAL REACHED! ***")
                        print(f"Completed in {step_count} steps")
                        print("Press R to reset or Q to quit")
                    elif truncated:
                        print("\n*** TIME LIMIT REACHED ***")
                        print("Press R to reset or Q to quit")

        clock.tick(30)

    pygame.quit()
    backend.close()
    print("\n✓ Interactive session ended")


def save_image(obs: np.ndarray, path: str):
    """Save observation as PNG image."""
    try:
        from PIL import Image
        img = Image.fromarray(obs)
        img.save(path)
        print(f"  Saved: {path}")
    except ImportError:
        print("  PIL not available, skipping image save")


def demo_backend_basics(save_images: bool = False):
    """Demonstrate basic backend usage."""
    print("\n" + "=" * 60)
    print("Demo 1: Backend Basics")
    print("=" * 60)

    # Load a task
    task_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    spec = TaskSpecification.from_json(str(task_path))

    print(f"\nTask: {spec.task_id}")
    print(f"Description: {spec.description}")
    print(f"Grid size: {spec.maze.dimensions}")
    print(f"Start: {spec.maze.start.to_tuple()}")
    print(f"Goal: {spec.maze.goal.to_tuple()}")

    # Create backend
    backend = get_backend("minigrid", render_mode="rgb_array")
    backend.configure(spec)

    # Reset environment
    obs, state, info = backend.reset(seed=42)

    print(f"\nInitial state:")
    print(f"  Agent position: {state.agent_position}")
    print(f"  Agent direction: {state.agent_direction}")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Mission: {backend.get_mission_text()}")

    # Take a few steps
    actions = [
        MiniGridActions.TURN_RIGHT,
        MiniGridActions.MOVE_FORWARD,
        MiniGridActions.MOVE_FORWARD,
    ]

    print("\nExecuting actions:")
    for action in actions:
        obs, reward, terminated, truncated, state, info = backend.step(action)
        print(f"  {ACTION_NAMES[action]}: pos={state.agent_position}, reward={reward:.2f}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        save_image(obs, str(output_dir / "demo1_minigrid_basic.png"))

    backend.close()
    print("\n✓ Backend basics demo complete")


def demo_key_door_puzzle(save_images: bool = False):
    """Demonstrate a key-door puzzle (Tier 2)."""
    print("\n" + "=" * 60)
    print("Demo 2: Key-Door Puzzle (Tier 2)")
    print("=" * 60)

    task_path = Path(__file__).parent / "tasks" / "tier2" / "single_key_001.json"
    spec = TaskSpecification.from_json(str(task_path))

    print(f"\nTask: {spec.task_id}")
    print(f"Description: {spec.description}")
    print(f"Keys: {[(k.id, k.color) for k in spec.mechanisms.keys]}")
    print(f"Doors: {[(d.id, d.requires_key) for d in spec.mechanisms.doors]}")

    backend = get_backend("minigrid", render_mode="rgb_array")
    backend.configure(spec)
    obs, state, info = backend.reset(seed=42)

    print(f"\nInitial: Agent at {state.agent_position}, carrying: {state.agent_carrying}")

    # Expert solution for this puzzle
    solution = [
        MiniGridActions.TURN_RIGHT,   # Face down
        MiniGridActions.MOVE_FORWARD,  # Move down
        MiniGridActions.MOVE_FORWARD,  # Move down to key row
        MiniGridActions.TURN_LEFT,     # Face right
        MiniGridActions.MOVE_FORWARD,  # Move to key
        MiniGridActions.PICKUP,        # Get key
        MiniGridActions.MOVE_FORWARD,  # Move right
        MiniGridActions.MOVE_FORWARD,  # Move right
        MiniGridActions.TOGGLE,        # Unlock door
        MiniGridActions.MOVE_FORWARD,  # Through door
        MiniGridActions.MOVE_FORWARD,  # Continue
        MiniGridActions.TURN_RIGHT,    # Face down
        MiniGridActions.MOVE_FORWARD,  # Move to goal
        MiniGridActions.MOVE_FORWARD,
        MiniGridActions.MOVE_FORWARD,
    ]

    print("\nExecuting expert solution:")
    for i, action in enumerate(solution):
        obs, reward, terminated, truncated, state, info = backend.step(action)
        status = ""
        if state.agent_carrying:
            status = f", carrying={state.agent_carrying}"
        if terminated:
            status += " [GOAL REACHED]"
        print(f"  {i+1}. {ACTION_NAMES[action]}: pos={state.agent_position}{status}")

        if terminated:
            break

    print(f"\nResult: {'SUCCESS' if terminated else 'IN PROGRESS'}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        save_image(obs, str(output_dir / "demo2_key_door.png"))

    backend.close()
    print("\n✓ Key-door puzzle demo complete")


def demo_runner_evaluation(save_images: bool = False):
    """Demonstrate using GridRunner for evaluation."""
    print("\n" + "=" * 60)
    print("Demo 3: GridRunner Evaluation")
    print("=" * 60)

    # Load multiple tasks
    task_dir = Path(__file__).parent / "tasks"
    tasks = []
    for tier in range(1, 4):  # Tiers 1-3
        tier_dir = task_dir / f"tier{tier}"
        if tier_dir.exists():
            for json_file in sorted(tier_dir.glob("*.json"))[:1]:  # First task per tier
                tasks.append(TaskSpecification.from_json(str(json_file)))

    print(f"\nLoaded {len(tasks)} tasks:")
    for t in tasks:
        print(f"  - {t.task_id} (Tier {t.difficulty_tier})")

    # Create runner with random policy
    runner = GridRunner(render_mode="rgb_array")

    def random_policy(obs, state, mission):
        """Simple random policy with bias toward forward movement."""
        import random
        weights = [0.1, 0.1, 0.5, 0.1, 0.05, 0.1, 0.05]  # Heavy forward bias
        return random.choices(range(7), weights=weights)[0]

    print("\nRunning episodes with random policy:")
    results = []
    for spec in tasks:
        result = runner.run_episode(spec, policy_fn=random_policy, seed=42)
        results.append(result)
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  {spec.task_id}: {status} in {result.steps_taken} steps")

    # Summary
    success_rate = sum(r.success for r in results) / len(results) * 100
    avg_steps = sum(r.steps_taken for r in results) / len(results)

    print(f"\nSummary:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Average steps: {avg_steps:.1f}")

    if save_images and results:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        # Save final observation from first result
        if results[0].trajectory:
            final_obs = results[0].trajectory[-1].observation
            save_image(final_obs, str(output_dir / "demo3_evaluation.png"))

    runner.close()
    print("\n✓ Runner evaluation demo complete")


def demo_all_tiers():
    """Show all available task tiers."""
    print("\n" + "=" * 60)
    print("Demo 4: Available Tasks by Tier")
    print("=" * 60)

    available = list_available_envs()

    total = 0
    for tier_name, task_ids in sorted(available.items()):
        print(f"\n{tier_name.upper()}:")
        for task_id in task_ids:
            print(f"  - {task_id}")
        total += len(task_ids)

    print(f"\nTotal: {total} tasks available")
    print("\n✓ Task listing complete")


def demo_observation_shapes(save_images: bool = False):
    """Show observation and render shapes."""
    print("\n" + "=" * 60)
    print("Demo 5: Observation & Render Shapes")
    print("=" * 60)

    task_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    spec = TaskSpecification.from_json(str(task_path))

    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    obs, state, info = backend.reset(seed=42)

    print(f"\nObservation from reset():")
    print(f"  Shape: {obs.shape}")
    print(f"  Dtype: {obs.dtype}")
    print(f"  Range: [{obs.min()}, {obs.max()}]")

    render = backend.render()
    print(f"\nRender output:")
    print(f"  Shape: {render.shape}")
    print(f"  Dtype: {render.dtype}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        save_image(obs, str(output_dir / "demo5_observation.png"))
        save_image(render, str(output_dir / "demo5_render.png"))

    backend.close()
    print("\n✓ Observation shapes demo complete")


def demo_deterministic_replay():
    """Demonstrate deterministic behavior with same seed."""
    print("\n" + "=" * 60)
    print("Demo 6: Deterministic Replay")
    print("=" * 60)

    task_path = Path(__file__).parent / "tasks" / "tier1" / "maze_simple_001.json"
    spec = TaskSpecification.from_json(str(task_path))

    actions = [
        MiniGridActions.TURN_RIGHT,
        MiniGridActions.MOVE_FORWARD,
        MiniGridActions.MOVE_FORWARD,
        MiniGridActions.TURN_LEFT,
        MiniGridActions.MOVE_FORWARD,
    ]

    def run_with_seed(seed):
        backend = get_backend("minigrid", render_mode="rgb_array")
        backend.configure(spec)
        obs, state, _ = backend.reset(seed=seed)
        positions = [state.agent_position]

        for action in actions:
            obs, _, _, _, state, _ = backend.step(action)
            positions.append(state.agent_position)

        backend.close()
        return positions

    # Run twice with same seed
    positions1 = run_with_seed(42)
    positions2 = run_with_seed(42)
    positions3 = run_with_seed(99)  # Different seed

    print(f"\nSeed 42 (run 1): {positions1}")
    print(f"Seed 42 (run 2): {positions2}")
    print(f"Seed 99:         {positions3}")

    print(f"\nRun 1 == Run 2: {positions1 == positions2}")
    print(f"Run 1 == Run 3: {positions1 == positions3}")

    print("\n✓ Deterministic replay demo complete")


def main():
    parser = argparse.ArgumentParser(description="MiniGrid Backend Demo")
    parser.add_argument("--visual", action="store_true", help="Save PNG images")
    parser.add_argument("--demo", type=int, help="Run specific demo (1-6)")
    parser.add_argument("--play", action="store_true", help="Interactive play mode")
    parser.add_argument("--task", type=str, help="Task to play (e.g., tier2/single_key_001)")
    args = parser.parse_args()

    # Interactive play mode
    if args.play:
        interactive_play(args.task)
        return

    print("=" * 60)
    print("MiniGrid Backend Demo")
    print("=" * 60)
    print("\nThis demo uses the MiniGridBackend (gymnasium minigrid package)")
    print("for standard square grid tasks.")

    demos = [
        demo_backend_basics,
        demo_key_door_puzzle,
        demo_runner_evaluation,
        demo_all_tiers,
        demo_observation_shapes,
        demo_deterministic_replay,
    ]

    if args.demo:
        if 1 <= args.demo <= len(demos):
            demos[args.demo - 1](save_images=args.visual)
        else:
            print(f"Invalid demo number. Choose 1-{len(demos)}")
    else:
        for demo_fn in demos:
            if demo_fn == demo_all_tiers:
                demo_fn()  # No save_images param
            elif demo_fn == demo_deterministic_replay:
                demo_fn()  # No save_images param
            else:
                demo_fn(save_images=args.visual)

    print("\n" + "=" * 60)
    print("MiniGrid Demo Complete!")
    print("=" * 60)

    if args.visual:
        output_dir = Path(__file__).parent / "demo_output"
        print(f"\nImages saved to: {output_dir}")


if __name__ == "__main__":
    main()
