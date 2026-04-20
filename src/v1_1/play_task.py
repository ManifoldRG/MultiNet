#!/usr/bin/env python3
"""
Interactive MiniGrid Task Player

A pygame-based interactive player for MiniGrid task JSON files.
Load any task specification and play through it using keyboard controls.

Usage:
    python play_task.py gridworld/tasks/tier3/gates_switches_002.json
    python play_task.py gridworld/tasks/tier1/maze_simple_001.json --record

Controls:
    Arrow Up / W    : Move forward
    Arrow Left / A  : Turn left
    Arrow Right / D : Turn right
    Space           : Pick up item
    X               : Drop item
    T / E           : Toggle (open door, press switch)
    Backspace       : Wait / done (no-op)
    R               : Reset current task
    Q / Escape      : Quit
    1-5             : Switch to tier N (loads first task from that tier)
    [ / ]           : Previous / next task within current tier
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent

# Ensure our v1_1 directory is on sys.path for gridworld imports
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str not in sys.path:
    sys.path.insert(0, _script_dir_str)

import numpy as np

try:
    import pygame
except ImportError:
    print(
        "Error: pygame is not installed.\n"
        "Install it with: pip install pygame\n"
        "  or: conda install -c conda-forge pygame"
    )
    sys.exit(1)

from gridworld.task_spec import TaskSpecification
from gridworld.task_parser import TaskParser
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.base import GridState
from gridworld.actions import MiniGridActions, ACTION_SHORT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window layout
GRID_DISPLAY_SIZE = 512      # Grid rendering area (square, left side)
INFO_PANEL_WIDTH = 320       # Info panel width (right side)
WINDOW_HEIGHT = GRID_DISPLAY_SIZE
WINDOW_WIDTH = GRID_DISPLAY_SIZE + INFO_PANEL_WIDTH

# Colors
COLOR_BG = (30, 30, 30)
COLOR_PANEL_BG = (40, 40, 48)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 150)
COLOR_TEXT_HIGHLIGHT = (100, 220, 130)
COLOR_TEXT_WARNING = (255, 180, 60)
COLOR_TEXT_ERROR = (255, 80, 80)
COLOR_TEXT_TITLE = (180, 200, 255)
COLOR_SEPARATOR = (70, 70, 80)
COLOR_SUCCESS_BG = (20, 100, 40, 180)
COLOR_FAIL_BG = (120, 20, 20, 180)
COLOR_OVERLAY_TEXT = (255, 255, 255)

# Direction labels
DIRECTION_NAMES = {0: "East (right)", 1: "South (down)", 2: "West (left)", 3: "North (up)"}
DIRECTION_ARROWS = {0: "->", 1: "v", 2: "<-", 3: "^"}

# Key repeat settings (milliseconds)
KEY_REPEAT_DELAY = 200
KEY_REPEAT_INTERVAL = 100

# Frame rate
FPS = 30


# ---------------------------------------------------------------------------
# Task discovery: find all task JSON files organized by tier
# ---------------------------------------------------------------------------

def discover_tasks(base_dir: Path) -> dict[int, list[Path]]:
    """
    Scan the tasks directory and return a mapping of tier number to sorted
    list of JSON task file paths.
    """
    tasks_dir = base_dir / "gridworld" / "tasks"
    tier_tasks: dict[int, list[Path]] = {}

    if not tasks_dir.exists():
        return tier_tasks

    for tier_num in range(1, 6):
        tier_dir = tasks_dir / f"tier{tier_num}"
        if tier_dir.exists():
            json_files = sorted(tier_dir.glob("*.json"))
            if json_files:
                tier_tasks[tier_num] = json_files

    return tier_tasks


# ---------------------------------------------------------------------------
# Interactive player
# ---------------------------------------------------------------------------

class MiniGridPlayer:
    """
    Pygame-based interactive player for MiniGrid task JSON files.
    """

    def __init__(self, task_path: str, record: bool = False):
        self.base_dir = _SCRIPT_DIR
        self.record = record
        self.trajectory: list[dict] = []
        self.task_path: Optional[Path] = None
        self.task_spec: Optional[TaskSpecification] = None

        # Backend for environment logic
        self.backend = MiniGridBackend(render_mode="rgb_array")

        # Discover all tier tasks for tier-switching and prev/next navigation
        self.tier_tasks = discover_tasks(self.base_dir)
        self.current_tier: int = 1
        self.current_task_index: int = 0

        # Episode state
        self.state: Optional[GridState] = None
        self.episode_done = False
        self.episode_success = False
        self.total_reward: float = 0.0
        self.last_action_name: str = ""

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("MiniGrid Task Player")
        pygame.key.set_repeat(KEY_REPEAT_DELAY, KEY_REPEAT_INTERVAL)
        self.clock = pygame.time.Clock()

        # Font setup -- use a clean monospace font
        self.font_title = self._load_font(22, bold=True)
        self.font_main = self._load_font(16)
        self.font_small = self._load_font(13)
        self.font_overlay = self._load_font(48, bold=True)
        self.font_overlay_sub = self._load_font(20)

        # Load the initial task
        self._load_task(task_path)

    def _load_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Load a monospace font, falling back to the default if needed."""
        # Try common monospace fonts
        mono_names = ["DejaVu Sans Mono", "Consolas", "Courier New", "monospace"]
        for name in mono_names:
            path = pygame.font.match_font(name, bold=bold)
            if path:
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    pass
        # Fallback to pygame default
        return pygame.font.SysFont(None, size, bold=bold)

    # ------------------------------------------------------------------
    # Task loading
    # ------------------------------------------------------------------

    def _load_task(self, path: str) -> None:
        """Load a task JSON file and reset the environment."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.base_dir / resolved

        if not resolved.exists():
            print(f"Error: task file not found: {resolved}")
            return

        self.task_path = resolved
        self.task_spec = TaskSpecification.from_json(str(resolved))

        # Update current tier and index tracking
        self.current_tier = self.task_spec.difficulty_tier
        if self.current_tier in self.tier_tasks:
            try:
                self.current_task_index = self.tier_tasks[self.current_tier].index(resolved)
            except ValueError:
                self.current_task_index = 0

        self._reset_env()

    def _reset_env(self) -> None:
        """Reset the environment from the current task spec."""
        if self.task_spec is None:
            return

        # Save previous trajectory if recording and it has content
        if self.record and self.trajectory:
            self._save_trajectory()

        self.backend.configure(self.task_spec)
        _obs, self.state, _info = self.backend.reset(seed=self.task_spec.seed)

        self.episode_done = False
        self.episode_success = False
        self.total_reward = 0.0
        self.last_action_name = ""
        self.trajectory = []

        if self.record:
            self.trajectory.append({
                "step": 0,
                "action": None,
                "action_name": None,
                "state": self.state.to_dict() if self.state else {},
            })

        pygame.display.set_caption(
            f"MiniGrid Player  |  {self.task_spec.task_id}  "
            f"(Tier {self.task_spec.difficulty_tier})"
        )

    def _load_tier(self, tier: int) -> None:
        """Switch to the first task in the given tier."""
        if tier in self.tier_tasks and self.tier_tasks[tier]:
            self.current_tier = tier
            self.current_task_index = 0
            self._load_task(str(self.tier_tasks[tier][0]))

    def _load_adjacent_task(self, delta: int) -> None:
        """Load the next (+1) or previous (-1) task within the current tier."""
        if self.current_tier not in self.tier_tasks:
            return
        tasks = self.tier_tasks[self.current_tier]
        if not tasks:
            return
        self.current_task_index = (self.current_task_index + delta) % len(tasks)
        self._load_task(str(tasks[self.current_task_index]))

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def _step(self, action: int) -> None:
        """Execute a single action in the environment."""
        if self.episode_done or self.state is None:
            return

        self.last_action_name = ACTION_SHORT.get(action, f"#{action}")

        _obs, reward, terminated, truncated, self.state, _info = self.backend.step(action)
        self.total_reward += reward

        if self.record:
            self.trajectory.append({
                "step": self.state.step_count,
                "action": action,
                "action_name": self.last_action_name,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "state": self.state.to_dict(),
            })

        if terminated or truncated:
            self.episode_done = True
            self.episode_success = self.state.goal_reached

    # ------------------------------------------------------------------
    # Recording / trajectory saving
    # ------------------------------------------------------------------

    def _save_trajectory(self) -> None:
        """Save the recorded trajectory to a JSON file."""
        if not self.trajectory:
            return

        task_id = self.task_spec.task_id if self.task_spec else "unknown"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_{task_id}_{timestamp}.json"
        output_path = self.base_dir / filename

        data = {
            "task_id": task_id,
            "task_file": str(self.task_path) if self.task_path else None,
            "difficulty_tier": self.task_spec.difficulty_tier if self.task_spec else None,
            "total_steps": len(self.trajectory) - 1,  # exclude initial state
            "total_reward": self.total_reward,
            "success": self.episode_success,
            "episode_done": self.episode_done,
            "trajectory": self.trajectory,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Trajectory saved to: {output_path}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_grid(self) -> None:
        """Render the MiniGrid environment onto the left side of the screen."""
        rgb_array = self.backend.render()  # numpy ndarray (H, W, 3)

        # pygame.surfarray expects (W, H, 3) so we transpose
        # But pygame.image.frombuffer can work with (H, W, 3) directly
        h, w, _c = rgb_array.shape

        # Create a surface from the raw RGB data
        surf = pygame.image.frombuffer(rgb_array.tobytes(), (w, h), "RGB")

        # Scale to fit the display area
        scaled = pygame.transform.smoothscale(surf, (GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE))
        self.screen.blit(scaled, (0, 0))

    def _render_info_panel(self) -> None:
        """Render the info panel on the right side of the screen."""
        panel_x = GRID_DISPLAY_SIZE
        panel_rect = pygame.Rect(panel_x, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)

        # Draw a vertical separator line
        pygame.draw.line(
            self.screen, COLOR_SEPARATOR,
            (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2
        )

        x = panel_x + 12
        y = 10

        # -- Title --
        task_id = self.task_spec.task_id if self.task_spec else "No task loaded"
        y = self._draw_text(f"Task: {task_id}", x, y, self.font_title, COLOR_TEXT_TITLE)
        y += 2

        if self.task_spec:
            y = self._draw_text(
                f"Tier {self.task_spec.difficulty_tier}",
                x, y, self.font_main, COLOR_TEXT_DIM
            )

        # Separator
        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 8

        # -- Agent State --
        if self.state:
            y = self._draw_text("AGENT STATE", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
            y += 2

            pos = self.state.agent_position
            y = self._draw_text(
                f"Position:  ({pos[0]}, {pos[1]})",
                x, y, self.font_main, COLOR_TEXT
            )

            dir_name = DIRECTION_NAMES.get(self.state.agent_direction, "?")
            arrow = DIRECTION_ARROWS.get(self.state.agent_direction, "?")
            y = self._draw_text(
                f"Direction: {arrow} {dir_name}",
                x, y, self.font_main, COLOR_TEXT
            )

            carrying = self.state.agent_carrying or "nothing"
            color = COLOR_TEXT_WARNING if self.state.agent_carrying else COLOR_TEXT_DIM
            y = self._draw_text(f"Carrying:  {carrying}", x, y, self.font_main, color)

            y += 2
            step_text = f"Steps: {self.state.step_count} / {self.state.max_steps}"
            y = self._draw_text(step_text, x, y, self.font_main, COLOR_TEXT)

            reward_text = f"Reward: {self.total_reward:.3f}"
            y = self._draw_text(reward_text, x, y, self.font_main, COLOR_TEXT)

            if self.last_action_name:
                y = self._draw_text(
                    f"Last action: {self.last_action_name}",
                    x, y, self.font_main, COLOR_TEXT_DIM
                )
        else:
            y = self._draw_text("No environment loaded", x, y, self.font_main, COLOR_TEXT_ERROR)

        # Separator
        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 8

        # -- Mechanism State --
        if self.state:
            has_mechanisms = (
                self.state.active_switches
                or self.state.open_gates
                or self.state.block_positions
                or self.state.teleporter_cooldowns
            )

            if has_mechanisms:
                y = self._draw_text("MECHANISMS", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
                y += 2

                if self.state.active_switches:
                    switches_str = ", ".join(sorted(self.state.active_switches))
                    y = self._draw_text(f"Active switches: {switches_str}", x, y, self.font_small, COLOR_TEXT_WARNING)

                if self.state.open_gates:
                    gates_str = ", ".join(sorted(self.state.open_gates))
                    y = self._draw_text(f"Open gates: {gates_str}", x, y, self.font_small, COLOR_TEXT_HIGHLIGHT)

                if self.state.block_positions:
                    for bid, bpos in self.state.block_positions.items():
                        y = self._draw_text(
                            f"Block {bid}: ({bpos[0]}, {bpos[1]})",
                            x, y, self.font_small, COLOR_TEXT
                        )

                if self.state.teleporter_cooldowns:
                    for tid, cd in self.state.teleporter_cooldowns.items():
                        cd_text = f"ready" if cd == 0 else f"cooldown {cd}"
                        y = self._draw_text(
                            f"Teleporter {tid}: {cd_text}",
                            x, y, self.font_small, COLOR_TEXT
                        )

                y += 4
                pygame.draw.line(
                    self.screen, COLOR_SEPARATOR,
                    (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y)
                )
                y += 8

        # -- Mission --
        if self.task_spec:
            y = self._draw_text("MISSION", x, y, self.font_main, COLOR_TEXT_HIGHLIGHT)
            y += 2
            mission = self.backend.get_mission_text()
            # Word-wrap the mission text
            y = self._draw_wrapped_text(mission, x, y, self.font_small, COLOR_TEXT, INFO_PANEL_WIDTH - 24)

        # Separator
        y += 4
        pygame.draw.line(self.screen, COLOR_SEPARATOR, (x, y), (panel_x + INFO_PANEL_WIDTH - 12, y))
        y += 8

        # -- Task navigation --
        if self.current_tier in self.tier_tasks:
            tasks = self.tier_tasks[self.current_tier]
            nav_text = f"Task {self.current_task_index + 1}/{len(tasks)} in tier {self.current_tier}"
            y = self._draw_text(nav_text, x, y, self.font_small, COLOR_TEXT_DIM)
            y += 4

        # -- Recording indicator --
        if self.record:
            y = self._draw_text("REC", x, y, self.font_main, COLOR_TEXT_ERROR)
            y += 4

        # -- Controls Reference (at the bottom) --
        controls_y = WINDOW_HEIGHT - 195
        pygame.draw.line(
            self.screen, COLOR_SEPARATOR,
            (x, controls_y), (panel_x + INFO_PANEL_WIDTH - 12, controls_y)
        )
        controls_y += 6
        controls_y = self._draw_text("CONTROLS", x, controls_y, self.font_main, COLOR_TEXT_HIGHLIGHT)
        controls_y += 2

        controls = [
            ("Up / W", "Move forward"),
            ("Left / A", "Turn left"),
            ("Right / D", "Turn right"),
            ("Space", "Pick up"),
            ("X", "Drop"),
            ("T / E", "Toggle"),
            ("Backspace", "Wait"),
            ("R", "Reset"),
            ("1-5", "Switch tier"),
            ("[ / ]", "Prev / next task"),
            ("Q / Esc", "Quit"),
        ]
        for key, desc in controls:
            controls_y = self._draw_text(
                f"{key:>11s}  {desc}", x, controls_y, self.font_small, COLOR_TEXT_DIM
            )

    def _render_overlay(self) -> None:
        """Render success/failure overlay when episode ends."""
        if not self.episode_done:
            return

        # Semi-transparent overlay
        overlay = pygame.Surface((GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE), pygame.SRCALPHA)
        if self.episode_success:
            overlay.fill((20, 100, 40, 160))
            main_text = "SUCCESS!"
            main_color = (100, 255, 130)
        else:
            overlay.fill((120, 20, 20, 160))
            main_text = "FAILED"
            main_color = (255, 100, 100)

        self.screen.blit(overlay, (0, 0))

        # Main text centered on the grid area
        text_surf = self.font_overlay.render(main_text, True, main_color)
        text_rect = text_surf.get_rect(
            center=(GRID_DISPLAY_SIZE // 2, GRID_DISPLAY_SIZE // 2 - 20)
        )
        self.screen.blit(text_surf, text_rect)

        # Sub text
        if self.state:
            sub_text = f"Steps: {self.state.step_count} / {self.state.max_steps}   Reward: {self.total_reward:.3f}"
        else:
            sub_text = ""
        sub_surf = self.font_overlay_sub.render(sub_text, True, COLOR_OVERLAY_TEXT)
        sub_rect = sub_surf.get_rect(
            center=(GRID_DISPLAY_SIZE // 2, GRID_DISPLAY_SIZE // 2 + 30)
        )
        self.screen.blit(sub_surf, sub_rect)

        # Hint
        hint_text = "Press R to reset, Q to quit, [ ] to switch task"
        hint_surf = self.font_small.render(hint_text, True, COLOR_TEXT_DIM)
        hint_rect = hint_surf.get_rect(
            center=(GRID_DISPLAY_SIZE // 2, GRID_DISPLAY_SIZE // 2 + 65)
        )
        self.screen.blit(hint_surf, hint_rect)

    def _draw_text(self, text: str, x: int, y: int, font: pygame.font.Font, color: tuple) -> int:
        """Draw a single line of text and return the y position below it."""
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 2

    def _draw_wrapped_text(
        self, text: str, x: int, y: int,
        font: pygame.font.Font, color: tuple, max_width: int
    ) -> int:
        """Draw word-wrapped text and return the y position below it."""
        words = text.split()
        lines: list[str] = []
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            if font.size(test)[0] <= max_width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        for line in lines:
            y = self._draw_text(line, x, y, font, color)
        return y

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the main event loop."""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.KEYDOWN:
                    action = self._handle_keydown(event)

                    if action == "quit":
                        running = False
                        break
                    elif action == "reset":
                        self._reset_env()
                    elif isinstance(action, int):
                        self._step(action)

            # Render
            self.screen.fill(COLOR_BG)

            if self.backend.env is not None:
                self._render_grid()
            else:
                # No env loaded -- show placeholder
                placeholder_surf = self.font_main.render(
                    "No environment loaded. Press 1-5 to load a tier.",
                    True, COLOR_TEXT_DIM
                )
                self.screen.blit(placeholder_surf, (20, GRID_DISPLAY_SIZE // 2))

            self._render_info_panel()
            self._render_overlay()

            pygame.display.flip()
            self.clock.tick(FPS)

        # Cleanup
        if self.record and self.trajectory:
            self._save_trajectory()

        self.backend.close()
        pygame.quit()

    def _handle_keydown(self, event: pygame.event.Event) -> Optional[int | str]:
        """
        Map a pygame KEYDOWN event to an action integer, or a control string
        ('quit', 'reset'), or None if not mapped.
        """
        key = event.key

        # Quit
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return "quit"

        # Reset
        if key == pygame.K_r:
            return "reset"

        # Tier switching (number keys 1-5)
        if key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            tier = key - pygame.K_0
            self._load_tier(tier)
            return None

        # Task navigation
        if key == pygame.K_LEFTBRACKET:
            self._load_adjacent_task(-1)
            return None
        if key == pygame.K_RIGHTBRACKET:
            self._load_adjacent_task(1)
            return None

        # If episode is done, ignore action keys (must reset first)
        if self.episode_done:
            return None

        # Movement and interaction
        if key in (pygame.K_UP, pygame.K_w):
            return MiniGridActions.MOVE_FORWARD   # 2
        if key in (pygame.K_LEFT, pygame.K_a):
            return MiniGridActions.TURN_LEFT      # 0
        if key in (pygame.K_RIGHT, pygame.K_d):
            return MiniGridActions.TURN_RIGHT     # 1
        if key == pygame.K_SPACE:
            return MiniGridActions.PICKUP          # 3
        if key == pygame.K_x:
            return MiniGridActions.DROP            # 4
        if key in (pygame.K_t, pygame.K_e):
            return MiniGridActions.TOGGLE          # 5
        if key == pygame.K_BACKSPACE:
            return MiniGridActions.DONE            # 6

        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive MiniGrid task player",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default="gridworld/tasks/tier1/maze_simple_001.json",
        help="Path to a task JSON file (default: tier1 simple maze)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record trajectory to a JSON file on exit or task switch",
    )
    args = parser.parse_args()

    player = MiniGridPlayer(task_path=args.task_file, record=args.record)
    player.run()


if __name__ == "__main__":
    main()
