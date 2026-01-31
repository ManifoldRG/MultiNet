#!/usr/bin/env python3
"""
MultiGrid Backend Demo

Demonstrates the custom MultiGrid implementation with:
- Multiple tiling types (square, hex, triangle)
- All object types (keys, doors, switches, gates, hazards, teleporters, zones)
- Mechanism interactions

Usage:
    python demo.py              # Run all demos
    python demo.py --visual     # Save PNG images of each demo
    python demo.py --demo 3     # Run specific demo
    python demo.py --play       # Interactive play mode
    python demo.py --play --tiling hex  # Play with hex grid
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from multigrid.env import MultiGridEnv, TilingRegistry
from multigrid.agent import Action
from multigrid.rendering import render_multigrid


def save_image(frame: np.ndarray, path: str):
    """Save frame as PNG image."""
    try:
        from PIL import Image
        img = Image.fromarray(frame)
        img.save(path)
        print(f"  Saved: {path}")
    except ImportError:
        print("  PIL not available, skipping image save")


def interactive_play(tiling: str = "square"):
    """
    Interactive play mode - control the agent with keyboard.

    Controls:
        Arrow Keys: Move/Turn
            Up: Move forward
            Down: Move backward
            Left: Turn left
            Right: Turn right
        Space: Pickup
        D: Drop
        T or Enter: Toggle (open door, activate switch)
        P: Push
        R: Reset episode
        Q or Escape: Quit
    """
    import pygame

    print("\n" + "=" * 60)
    print("Interactive Play Mode")
    print("=" * 60)
    print(f"\nTiling: {tiling}")
    print(f"\nControls:")
    print("  Arrow Up    : Move forward")
    print("  Arrow Down  : Move backward")
    print("  Arrow Left  : Turn left")
    print("  Arrow Right : Turn right")
    print("  Space       : Pickup")
    print("  D           : Drop")
    print("  T / Enter   : Toggle (doors, switches)")
    print("  P           : Push")
    print("  R           : Reset")
    print("  Q / Escape  : Quit")
    print("\n" + "-" * 60)

    # Create a playground task with various objects
    task_spec = {
        "task_id": "interactive_play",
        "seed": 42,
        "tiling": {"type": tiling, "grid_size": {"width": 8, "height": 8}},
        "rules": {"key_consumption": True},
        "scene": {
            "agent": {"position": {"x": 0.15, "y": 0.15}, "facing": 1},
            "objects": [
                # Key and door
                {"id": "key_blue", "type": "key", "color": "blue",
                 "position": {"x": 0.35, "y": 0.15}},
                {"id": "door_blue", "type": "door", "color": "blue",
                 "position": {"x": 0.55, "y": 0.15}, "is_locked": True},

                # Switch and gate
                {"id": "switch_1", "type": "switch", "color": "yellow",
                 "position": {"x": 0.15, "y": 0.45}, "switch_type": "toggle",
                 "controls": ["gate_1"], "initial_state": False},
                {"id": "gate_1", "type": "gate", "color": "yellow",
                 "position": {"x": 0.55, "y": 0.45}, "is_open": False,
                 "controlled_by": ["switch_1"]},

                # Pushable box
                {"id": "box_1", "type": "movable", "color": "green",
                 "position": {"x": 0.35, "y": 0.65}},

                # Hazard
                {"id": "lava_1", "type": "hazard", "color": "red",
                 "position": {"x": 0.75, "y": 0.75}, "hazard_type": "lava"},

                # Goal zone
                {"id": "goal_zone", "type": "zone", "color": "cyan",
                 "position": {"x": 0.85, "y": 0.15}},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.85, "y": 0.15}},
        "limits": {"max_steps": 200}
    }

    env = MultiGridEnv(task_spec, tiling=tiling, render_mode="rgb_array")
    obs, info = env.reset()

    # Initialize pygame
    pygame.init()

    # Scale up for visibility
    scale = 2
    display_size = (obs.shape[1] * scale, obs.shape[0] * scale)
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption(f"MultiGrid ({tiling}): Interactive Play")

    # Key mapping
    key_to_action = {
        pygame.K_UP: Action.FORWARD,
        pygame.K_DOWN: Action.BACKWARD,
        pygame.K_LEFT: Action.TURN_LEFT,
        pygame.K_RIGHT: Action.TURN_RIGHT,
        pygame.K_SPACE: Action.PICKUP,
        pygame.K_d: Action.DROP,
        pygame.K_t: Action.TOGGLE,
        pygame.K_RETURN: Action.TOGGLE,
        pygame.K_p: Action.PUSH,
    }

    clock = pygame.time.Clock()
    running = True
    step_count = 0

    def render_frame():
        frame = env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, display_size)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    def print_status():
        agent = env.state.agent
        holding = agent.holding.id if agent.holding else "nothing"
        facing = agent.get_facing_direction(env.tiling)
        print(f"  Step {step_count}: cell={agent.cell_id}, facing={facing}, holding={holding}")

    render_frame()
    print(f"\nStarting at {env.state.agent.cell_id}")
    print(f"Goal: reach the cyan zone at top-right")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    obs, info = env.reset()
                    step_count = 0
                    render_frame()
                    print("\n--- Episode Reset ---")
                    print(f"Starting at {env.state.agent.cell_id}")
                elif event.key in key_to_action:
                    action = key_to_action[event.key]
                    obs, reward, terminated, truncated, info = env.step(action.value)
                    step_count += 1
                    render_frame()
                    print_status()

                    # Show action effects
                    if info.get("action_effect"):
                        print(f"    -> {info['action_effect']}")
                    if info.get("invalid_action"):
                        print(f"    -> blocked")

                    if info.get("hazard_hit"):
                        print("\n*** STEPPED IN LAVA! ***")
                        print("Press R to reset or Q to quit")
                    elif terminated:
                        print("\n*** GOAL REACHED! ***")
                        print(f"Completed in {step_count} steps")
                        print("Press R to reset or Q to quit")
                    elif truncated:
                        print("\n*** TIME LIMIT REACHED ***")
                        print("Press R to reset or Q to quit")

        clock.tick(30)

    pygame.quit()
    print("\n✓ Interactive session ended")


def demo_tiling_types(save_images: bool = False):
    """Demonstrate all three tiling types."""
    print("\n" + "=" * 60)
    print("Demo 1: Tiling Types (Square, Hex, Triangle)")
    print("=" * 60)

    output_dir = Path(__file__).parent / "demo_output"
    if save_images:
        output_dir.mkdir(exist_ok=True)

    for tiling_name in ["square", "hex", "triangle"]:
        print(f"\n--- {tiling_name.upper()} Tiling ---")

        task_spec = {
            "task_id": f"demo_{tiling_name}",
            "seed": 42,
            "tiling": {
                "type": tiling_name,
                "grid_size": {"width": 5, "height": 5}
            },
            "scene": {
                "agent": {"position": {"x": 0.3, "y": 0.3}, "facing": 0},
                "objects": [
                    {"id": "box_1", "type": "movable", "color": "blue",
                     "position": {"x": 0.5, "y": 0.5}},
                    {"id": "box_2", "type": "movable", "color": "red",
                     "position": {"x": 0.7, "y": 0.3}},
                ]
            },
            "goal": {"type": "reach_position", "target": {"x": 0.8, "y": 0.8}},
            "limits": {"max_steps": 50}
        }

        env = MultiGridEnv(task_spec, tiling=tiling_name, render_mode="rgb_array")
        obs, info = env.reset()

        tiling = env.tiling
        print(f"  Cells: {len(tiling.cells)}")
        print(f"  Directions: {len(tiling.directions)} ({', '.join(tiling.directions)})")
        print(f"  Agent at: {env.state.agent.cell_id}")
        print(f"  Observation shape: {obs.shape}")

        if save_images:
            frame = env.render()
            save_image(frame, str(output_dir / f"demo1_{tiling_name}.png"))

    print("\n✓ Tiling types demo complete")


def demo_all_objects(save_images: bool = False):
    """Demonstrate all object types."""
    print("\n" + "=" * 60)
    print("Demo 2: All Object Types")
    print("=" * 60)

    task_spec = {
        "task_id": "demo_objects",
        "seed": 42,
        "tiling": {"type": "square", "grid_size": {"width": 8, "height": 8}},
        "rules": {"key_consumption": True},
        "scene": {
            "agent": {"position": {"x": 0.1, "y": 0.1}, "facing": 1},
            "objects": [
                # Row 1: Key and Door
                {"id": "key_blue", "type": "key", "color": "blue",
                 "position": {"x": 0.25, "y": 0.15}},
                {"id": "door_blue", "type": "door", "color": "blue",
                 "position": {"x": 0.4, "y": 0.15}, "is_locked": True},

                # Row 2: Switch and Gate
                {"id": "switch_1", "type": "switch", "color": "yellow",
                 "position": {"x": 0.25, "y": 0.35}, "switch_type": "toggle",
                 "controls": ["gate_1"], "initial_state": False},
                {"id": "gate_1", "type": "gate", "color": "yellow",
                 "position": {"x": 0.5, "y": 0.35}, "is_open": False},

                # Row 3: Movable and Wall
                {"id": "box_1", "type": "movable", "color": "green",
                 "position": {"x": 0.25, "y": 0.55}},
                {"id": "wall_1", "type": "wall", "color": "grey",
                 "position": {"x": 0.5, "y": 0.55}},

                # Row 4: Hazard and Zone
                {"id": "lava_1", "type": "hazard", "color": "red",
                 "position": {"x": 0.25, "y": 0.75}, "hazard_type": "lava"},
                {"id": "zone_1", "type": "zone", "color": "cyan",
                 "position": {"x": 0.5, "y": 0.75}},

                # Teleporter pair
                {"id": "tele_1", "type": "teleporter", "color": "purple",
                 "position": {"x": 0.75, "y": 0.25}, "linked_to": "tele_2"},
                {"id": "tele_2", "type": "teleporter", "color": "purple",
                 "position": {"x": 0.75, "y": 0.75}, "linked_to": "tele_1"},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.9, "y": 0.9}},
        "limits": {"max_steps": 100}
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="rgb_array")
    env.reset()

    print("\nObjects in scene:")
    for obj_id, obj in env.state.objects.items():
        details = f"at {obj.cell_id}"
        if hasattr(obj, "is_locked"):
            details += f", locked={obj.is_locked}"
        if hasattr(obj, "is_open"):
            details += f", open={obj.is_open}"
        if hasattr(obj, "is_active"):
            details += f", active={obj.is_active}"
        if hasattr(obj, "linked_to"):
            details += f", linked_to={obj.linked_to}"
        print(f"  {obj_id} ({obj.obj_type}, {obj.color}): {details}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo2_all_objects.png"))

    print("\n✓ All objects demo complete")


def demo_key_door_mechanism(save_images: bool = False):
    """Demonstrate key + door interaction."""
    print("\n" + "=" * 60)
    print("Demo 3: Key + Door Mechanism")
    print("=" * 60)

    # Grid layout (6 wide):
    # sq_1_0 (agent) -> sq_1_1 (key) -> sq_1_2 -> sq_1_3 (door) -> sq_1_4 -> sq_1_5 (goal)
    task_spec = {
        "task_id": "demo_key_door",
        "seed": 42,
        "tiling": {"type": "square", "grid_size": {"width": 6, "height": 3}},
        "rules": {"key_consumption": True},
        "scene": {
            "agent": {"position": {"x": 0.08, "y": 0.5}, "facing": 1},  # sq_1_0, face east
            "objects": [
                {"id": "key_blue", "type": "key", "color": "blue",
                 "position": {"x": 0.25, "y": 0.5}},  # sq_1_1
                {"id": "door_blue", "type": "door", "color": "blue",
                 "position": {"x": 0.58, "y": 0.5}, "is_locked": True},  # sq_1_3
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.92, "y": 0.5}},  # sq_1_5
        "limits": {"max_steps": 20}
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="rgb_array")
    env.reset()

    door = env.state.objects["door_blue"]

    print(f"\nInitial state:")
    print(f"  Agent: {env.state.agent.cell_id}, facing: {env.state.agent.get_facing_direction(env.tiling)}")
    print(f"  Key: {env.state.objects['key_blue'].cell_id}")
    print(f"  Door: {door.cell_id}, locked={door.is_locked}, open={door.is_open}")

    # Execute solution: agent at sq_1_0, key at sq_1_1, door at sq_1_3
    actions = [
        (Action.FORWARD, "Move to key (sq_1_1)"),
        (Action.PICKUP, "Pick up key"),
        (Action.FORWARD, "Move to sq_1_2"),
        (Action.FORWARD, "Move to door (sq_1_3) - blocked"),
        (Action.TOGGLE, "Unlock door with key"),
        (Action.FORWARD, "Move through door (sq_1_3)"),
        (Action.FORWARD, "Move to sq_1_4"),
        (Action.FORWARD, "Move to goal (sq_1_5)"),
    ]

    print("\nExecuting actions:")
    for action, desc in actions:
        obs, reward, terminated, truncated, info = env.step(action.value)
        holding = env.state.agent.holding.id if env.state.agent.holding else None
        status = f"pos={env.state.agent.cell_id}, holding={holding}"
        if info.get("action_effect"):
            status += f", effect={info['action_effect']}"
        if info.get("invalid_action"):
            status += " [BLOCKED]"
        print(f"  {desc}: {status}")

        if terminated:
            print("  >>> GOAL REACHED!")
            break

    print(f"\nFinal state:")
    print(f"  Door: locked={door.is_locked}, open={door.is_open}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo3_key_door.png"))

    print("\n✓ Key + door demo complete")


def demo_switch_gate_mechanism(save_images: bool = False):
    """Demonstrate switch + gate interaction."""
    print("\n" + "=" * 60)
    print("Demo 4: Switch + Gate Mechanism")
    print("=" * 60)

    # Grid layout (6 wide):
    # sq_1_0 (agent) -> sq_1_1 (switch) -> sq_1_2 -> sq_1_3 (gate) -> sq_1_4 -> sq_1_5 (goal)
    task_spec = {
        "task_id": "demo_switch_gate",
        "seed": 42,
        "tiling": {"type": "square", "grid_size": {"width": 6, "height": 3}},
        "scene": {
            "agent": {"position": {"x": 0.08, "y": 0.5}, "facing": 1},  # sq_1_0
            "objects": [
                {"id": "switch_1", "type": "switch", "color": "yellow",
                 "position": {"x": 0.25, "y": 0.5}, "switch_type": "toggle",  # sq_1_1
                 "controls": ["gate_1"], "initial_state": False},
                {"id": "gate_1", "type": "gate", "color": "yellow",
                 "position": {"x": 0.58, "y": 0.5}, "is_open": False,  # sq_1_3
                 "controlled_by": ["switch_1"]},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.92, "y": 0.5}},  # sq_1_5
        "limits": {"max_steps": 20}
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="rgb_array")
    env.reset()

    switch = env.state.objects["switch_1"]
    gate = env.state.objects["gate_1"]

    print(f"\nInitial state:")
    print(f"  Agent: {env.state.agent.cell_id}")
    print(f"  Switch: {switch.cell_id}, active={switch.is_active}")
    print(f"  Gate: {gate.cell_id}, open={gate.is_open}")

    actions = [
        (Action.FORWARD, "Move to switch (sq_1_1)"),
        (Action.TOGGLE, "Activate switch"),
        (Action.FORWARD, "Move to sq_1_2"),
        (Action.FORWARD, "Move through gate (sq_1_3)"),
        (Action.FORWARD, "Move to sq_1_4"),
        (Action.FORWARD, "Move to goal (sq_1_5)"),
    ]

    print("\nExecuting actions:")
    for action, desc in actions:
        obs, reward, terminated, truncated, info = env.step(action.value)
        status = f"pos={env.state.agent.cell_id}, switch={switch.is_active}, gate={gate.is_open}"
        if info.get("action_effect"):
            status += f", effect={info['action_effect']}"
        print(f"  {desc}: {status}")

        if terminated:
            print("  >>> GOAL REACHED!")
            break

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo4_switch_gate.png"))

    print("\n✓ Switch + gate demo complete")


def demo_hazard(save_images: bool = False):
    """Demonstrate hazard termination."""
    print("\n" + "=" * 60)
    print("Demo 5: Hazard (Lava)")
    print("=" * 60)

    task_spec = {
        "task_id": "demo_hazard",
        "seed": 42,
        "tiling": {"type": "square", "grid_size": {"width": 4, "height": 3}},
        "scene": {
            "agent": {"position": {"x": 0.15, "y": 0.5}, "facing": 1},
            "objects": [
                {"id": "lava_1", "type": "hazard", "color": "red",
                 "position": {"x": 0.5, "y": 0.5}, "hazard_type": "lava"},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.85, "y": 0.5}},
        "limits": {"max_steps": 10}
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="rgb_array")
    env.reset()

    print(f"\nAgent starting at {env.state.agent.cell_id}")
    print(f"Lava at {env.state.objects['lava_1'].cell_id}")

    print("\nMoving toward lava...")
    obs, reward, terminated, truncated, info = env.step(Action.FORWARD.value)
    print(f"  Step 1: pos={env.state.agent.cell_id}")

    obs, reward, terminated, truncated, info = env.step(Action.FORWARD.value)
    print(f"  Step 2: pos={env.state.agent.cell_id}")
    print(f"  Hazard hit: {info.get('hazard_hit', False)}")
    print(f"  Terminated: {terminated}")

    if terminated:
        print("\n  >>> AGENT DIED IN LAVA!")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo5_hazard.png"))

    print("\n✓ Hazard demo complete")


def demo_push_action(save_images: bool = False):
    """Demonstrate push action."""
    print("\n" + "=" * 60)
    print("Demo 6: Push Action")
    print("=" * 60)

    task_spec = {
        "task_id": "demo_push",
        "seed": 42,
        "tiling": {"type": "square", "grid_size": {"width": 5, "height": 3}},
        "scene": {
            "agent": {"position": {"x": 0.1, "y": 0.5}, "facing": 1},
            "objects": [
                {"id": "box_1", "type": "movable", "color": "green",
                 "position": {"x": 0.3, "y": 0.5}},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.9, "y": 0.5}},
        "limits": {"max_steps": 20}
    }

    env = MultiGridEnv(task_spec, tiling="square", render_mode="rgb_array")
    env.reset()

    box = env.state.objects["box_1"]

    print(f"\nInitial: Agent at {env.state.agent.cell_id}, Box at {box.cell_id}")

    # Push the box
    obs, reward, terminated, truncated, info = env.step(Action.PUSH.value)
    print(f"\nAfter PUSH:")
    print(f"  Agent at {env.state.agent.cell_id}")
    print(f"  Box at {box.cell_id}")
    print(f"  Effect: {info.get('action_effect')}")

    # Push again
    obs, reward, terminated, truncated, info = env.step(Action.FORWARD.value)
    obs, reward, terminated, truncated, info = env.step(Action.PUSH.value)
    print(f"\nAfter move + PUSH:")
    print(f"  Agent at {env.state.agent.cell_id}")
    print(f"  Box at {box.cell_id}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo6_push.png"))

    print("\n✓ Push demo complete")


def demo_triangle_navigation(save_images: bool = False):
    """Demonstrate navigation in triangle tiling."""
    print("\n" + "=" * 60)
    print("Demo 7: Triangle Tiling Navigation")
    print("=" * 60)

    task_spec = {
        "task_id": "demo_triangle_nav",
        "seed": 42,
        "tiling": {"type": "triangle", "grid_size": {"width": 4, "height": 4}},
        "scene": {
            "agent": {"position": {"x": 0.3, "y": 0.3}, "facing": 0},
            "objects": [
                {"id": "goal_marker", "type": "zone", "color": "green",
                 "position": {"x": 0.7, "y": 0.7}},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.7, "y": 0.7}},
        "limits": {"max_steps": 30}
    }

    env = MultiGridEnv(task_spec, tiling="triangle", render_mode="rgb_array")
    env.reset()

    print(f"\nTriangle tiling:")
    print(f"  Total cells: {len(env.tiling.cells)}")
    print(f"  Directions: {env.tiling.directions}")
    print(f"  Agent at: {env.state.agent.cell_id}")
    print(f"  Agent facing: {env.state.agent.get_facing_direction(env.tiling)}")

    print("\nNavigating (10 random moves):")
    import random
    for i in range(10):
        action = random.choice([Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT])
        obs, reward, terminated, truncated, info = env.step(action.value)
        facing = env.state.agent.get_facing_direction(env.tiling)
        print(f"  {i+1}. {action.name}: cell={env.state.agent.cell_id}, facing={facing}")

        if terminated:
            print("  >>> GOAL REACHED!")
            break

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo7_triangle.png"))

    print("\n✓ Triangle navigation demo complete")


def demo_hex_with_mechanisms(save_images: bool = False):
    """Demonstrate hex tiling with mechanisms."""
    print("\n" + "=" * 60)
    print("Demo 8: Hex Tiling with Mechanisms")
    print("=" * 60)

    task_spec = {
        "task_id": "demo_hex_mechanisms",
        "seed": 42,
        "tiling": {"type": "hex", "grid_size": {"width": 4, "height": 4}},
        "rules": {"key_consumption": True},
        "scene": {
            "agent": {"position": {"x": 0.2, "y": 0.2}, "facing": 1},
            "objects": [
                {"id": "key_red", "type": "key", "color": "red",
                 "position": {"x": 0.4, "y": 0.3}},
                {"id": "door_red", "type": "door", "color": "red",
                 "position": {"x": 0.6, "y": 0.5}, "is_locked": True},
                {"id": "box_1", "type": "movable", "color": "blue",
                 "position": {"x": 0.3, "y": 0.6}},
            ]
        },
        "goal": {"type": "reach_position", "target": {"x": 0.8, "y": 0.8}},
        "limits": {"max_steps": 50}
    }

    env = MultiGridEnv(task_spec, tiling="hex", render_mode="rgb_array")
    env.reset()

    print(f"\nHex tiling:")
    print(f"  Total cells: {len(env.tiling.cells)}")
    print(f"  Directions: {env.tiling.directions}")

    print("\nObjects:")
    for obj_id, obj in env.state.objects.items():
        print(f"  {obj_id} ({obj.obj_type}): {obj.cell_id}")

    if save_images:
        output_dir = Path(__file__).parent / "demo_output"
        output_dir.mkdir(exist_ok=True)
        frame = env.render()
        save_image(frame, str(output_dir / "demo8_hex_mechanisms.png"))

    print("\n✓ Hex mechanisms demo complete")


def main():
    parser = argparse.ArgumentParser(description="MultiGrid Backend Demo")
    parser.add_argument("--visual", action="store_true", help="Save PNG images")
    parser.add_argument("--demo", type=int, help="Run specific demo (1-8)")
    parser.add_argument("--play", action="store_true", help="Interactive play mode")
    parser.add_argument("--tiling", type=str, default="square",
                        choices=["square", "hex", "triangle"],
                        help="Tiling type for play mode (default: square)")
    args = parser.parse_args()

    # Interactive play mode
    if args.play:
        interactive_play(args.tiling)
        return

    print("=" * 60)
    print("MultiGrid Backend Demo")
    print("=" * 60)
    print("\nThis demo uses the custom MultiGrid implementation with")
    print("support for square, hex, and triangle tilings.")

    demos = [
        ("Tiling Types", demo_tiling_types),
        ("All Objects", demo_all_objects),
        ("Key + Door", demo_key_door_mechanism),
        ("Switch + Gate", demo_switch_gate_mechanism),
        ("Hazard", demo_hazard),
        ("Push Action", demo_push_action),
        ("Triangle Navigation", demo_triangle_navigation),
        ("Hex with Mechanisms", demo_hex_with_mechanisms),
    ]

    if args.demo:
        if 1 <= args.demo <= len(demos):
            name, fn = demos[args.demo - 1]
            fn(save_images=args.visual)
        else:
            print(f"Invalid demo number. Choose 1-{len(demos)}")
            print("\nAvailable demos:")
            for i, (name, _) in enumerate(demos, 1):
                print(f"  {i}. {name}")
    else:
        for name, fn in demos:
            fn(save_images=args.visual)

    print("\n" + "=" * 60)
    print("MultiGrid Demo Complete!")
    print("=" * 60)

    if args.visual:
        output_dir = Path(__file__).parent / "demo_output"
        print(f"\nImages saved to: {output_dir}")


if __name__ == "__main__":
    main()
