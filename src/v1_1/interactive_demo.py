#!/usr/bin/env python3
"""
Interactive pygame demo for MultiGrid.

Controls:
- Arrow Keys / WASD: Move agent (FORWARD in facing direction)
- Q/E: Turn left/right
- SPACE: Pick up / Drop object
- P: Push object
- R: Reset environment
- 1/2/3: Switch between Square/Hex/Triangle grids
- ESC: Quit
"""

import sys
import os
import pygame
import math
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multigrid.env import MultiGridEnv
from multigrid.agent import Action


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 50)
PURPLE = (200, 50, 200)
ORANGE = (255, 165, 0)


def draw_hex(surface, center, size, color, filled=True):
    """Draw a hexagon."""
    vertices = []
    for i in range(6):
        angle = math.pi / 2 - i * math.pi / 3
        x = center[0] + size * math.cos(angle)
        y = center[1] - size * math.sin(angle)
        vertices.append((x, y))

    if filled:
        pygame.draw.polygon(surface, color, vertices)
    pygame.draw.polygon(surface, BLACK, vertices, 2)


def draw_triangle(surface, center, size, color, pointing_up, filled=True):
    """
    Draw an equilateral triangle.

    Args:
        center: (x, y) position of triangle centroid
        size: height of the triangle
        pointing_up: True for upward pointing, False for downward
    """
    # For equilateral triangle with height h:
    # - Side length s = 2h / sqrt(3)
    # - Half of base = s / 2 = h / sqrt(3)
    # - Centroid is h/3 from base, 2h/3 from apex

    half_base = size / math.sqrt(3)

    if pointing_up:
        # Apex is 2/3 of height above centroid
        # Base is 1/3 of height below centroid
        vertices = [
            (center[0], center[1] - 2 * size / 3),  # Top apex
            (center[0] - half_base, center[1] + size / 3),  # Bottom left
            (center[0] + half_base, center[1] + size / 3)   # Bottom right
        ]
    else:
        # Apex is 2/3 of height below centroid
        # Base is 1/3 of height above centroid
        vertices = [
            (center[0], center[1] + 2 * size / 3),  # Bottom apex
            (center[0] - half_base, center[1] - size / 3),  # Top left
            (center[0] + half_base, center[1] - size / 3)   # Top right
        ]

    if filled:
        pygame.draw.polygon(surface, color, vertices)
    pygame.draw.polygon(surface, BLACK, vertices, 2)


def draw_square(surface, center, size, color, filled=True):
    """Draw a square."""
    rect = pygame.Rect(center[0] - size / 2, center[1] - size / 2, size, size)
    if filled:
        pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, BLACK, rect, 2)


def draw_agent(surface, center, size, facing_angle):
    """Draw the agent as a triangle pointing in facing direction."""
    # Draw body (circle)
    pygame.draw.circle(surface, BLUE, (int(center[0]), int(center[1])), int(size * 0.6))

    # Draw facing indicator (triangle)
    indicator_size = size * 0.8
    angle = facing_angle
    vertices = [
        (center[0] + indicator_size * math.cos(angle),
         center[1] - indicator_size * math.sin(angle)),
        (center[0] + indicator_size * 0.3 * math.cos(angle + 2.5),
         center[1] - indicator_size * 0.3 * math.sin(angle + 2.5)),
        (center[0] + indicator_size * 0.3 * math.cos(angle - 2.5),
         center[1] - indicator_size * 0.3 * math.sin(angle - 2.5))
    ]
    pygame.draw.polygon(surface, WHITE, vertices)
    pygame.draw.polygon(surface, BLACK, vertices, 1)


def draw_object(surface, center, size, color):
    """Draw an object (cube)."""
    pygame.draw.circle(surface, color, (int(center[0]), int(center[1])), int(size * 0.5))
    pygame.draw.circle(surface, BLACK, (int(center[0]), int(center[1])), int(size * 0.5), 2)


class InteractiveDemo:
    def __init__(self, width=800, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height + 100))  # Extra space for info
        pygame.display.set_caption("MultiGrid Interactive Demo")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)

        self.tiling_type = "square"
        self.grid_size = 10

        self.env = None
        self.reset_env()

    def reset_env(self):
        """Create/reset the environment."""
        task_spec = {
            "task_id": "interactive_demo",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [
                    {
                        "id": "cube_red",
                        "type": "movable",
                        "color": "red",
                        "position": {"x": 0.7, "y": 0.3},
                        "size": 0.1
                    },
                    {
                        "id": "cube_green",
                        "type": "movable",
                        "color": "green",
                        "position": {"x": 0.3, "y": 0.7},
                        "size": 0.1
                    }
                ],
                "agent": {
                    "position": {"x": 0.2, "y": 0.2},
                    "facing": 1  # Facing east
                }
            },
            "goal": {},
            "limits": {"max_steps": 1000},
            "tiling": {"type": self.tiling_type, "grid_size": {"width": self.grid_size, "height": self.grid_size}}
        }

        self.env = MultiGridEnv(task_spec, tiling=self.tiling_type)
        self.env.reset()

    def handle_input(self):
        """Handle keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.reset_env()
                elif event.key == pygame.K_1:
                    self.tiling_type = "square"
                    self.reset_env()
                elif event.key == pygame.K_2:
                    self.tiling_type = "hex"
                    self.reset_env()
                elif event.key == pygame.K_3:
                    self.tiling_type = "triangle"
                    self.reset_env()
                elif event.key in [pygame.K_UP, pygame.K_w]:
                    self.env.step(Action.FORWARD)
                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    self.env.step(Action.BACKWARD)
                elif event.key in [pygame.K_LEFT, pygame.K_a, pygame.K_q]:
                    self.env.step(Action.TURN_LEFT)
                elif event.key in [pygame.K_RIGHT, pygame.K_d, pygame.K_e]:
                    self.env.step(Action.TURN_RIGHT)
                elif event.key == pygame.K_SPACE:
                    if self.env.state.agent.holding:
                        self.env.step(Action.DROP)
                    else:
                        self.env.step(Action.PICKUP)
                elif event.key == pygame.K_p:
                    self.env.step(Action.PUSH)

        return True

    def draw_grid(self):
        """Draw the grid."""
        self.screen.fill(WHITE)

        tiling = self.env.tiling

        # Calculate proper cell sizes for each tiling type
        margin = 50
        usable_width = self.width - 2 * margin
        usable_height = self.height - 2 * margin

        # Draw grid cells
        for cell_id, cell in tiling.cells.items():
            x_norm, y_norm = cell.position_hint
            x = x_norm * usable_width + margin
            y = y_norm * usable_height + margin

            if self.tiling_type == "square":
                cell_size = usable_width / self.grid_size
                draw_square(self.screen, (x, y), cell_size, LIGHT_GRAY, filled=True)
            elif self.tiling_type == "hex":
                # Calculate hex size matching HexTiling coordinate system
                width_spacing = (self.grid_size - 1) if self.grid_size > 1 else 1
                height_spacing = (self.grid_size - 1) if self.grid_size > 1 else 1
                size_from_width = 0.95 / ((self.grid_size + 0.5) * math.sqrt(3)) if self.grid_size > 0 else 0.1
                size_from_height = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
                size = min(size_from_width, size_from_height)
                # Convert to screen space
                hex_size = size * usable_width
                draw_hex(self.screen, (x, y), hex_size, LIGHT_GRAY, filled=True)
            elif self.tiling_type == "triangle":
                # Triangles are subdivisions of hexagons
                # Parse triangle ID: tri_hexcol_hexrow_triidx
                parts = cell_id.split("_")
                if len(parts) == 4:
                    from multigrid.tilings.hex import OffsetCoord, offset_to_axial
                    _, hex_col_str, hex_row_str, tri_idx_str = parts
                    tri_idx = int(tri_idx_str)
                    hex_col = int(hex_col_str)
                    hex_row = int(hex_row_str)

                    # Calculate hex size (same as HexTiling)
                    width_spacing = (self.grid_size - 1) if self.grid_size > 1 else 1
                    height_spacing = (self.grid_size - 1) if self.grid_size > 1 else 1
                    size_from_width = 0.95 / ((self.grid_size + 0.5) * math.sqrt(3)) if self.grid_size > 0 else 0.1
                    size_from_height = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
                    hex_size = min(size_from_width, size_from_height)

                    # Calculate hex center in normalized coordinates
                    col_pos = hex_col * math.sqrt(3) * hex_size
                    row_pos = hex_row * 1.5 * hex_size
                    if hex_row % 2 == 1:
                        col_pos += math.sqrt(3) / 2 * hex_size

                    grid_width = (self.grid_size + 0.5) * math.sqrt(3) * hex_size
                    grid_height = (self.grid_size - 0.5) * 1.5 * hex_size
                    x_offset = (1.0 - grid_width) / 2
                    y_offset = (1.0 - grid_height) / 2

                    hex_center_x_norm = col_pos + x_offset
                    hex_center_y_norm = row_pos + y_offset

                    # Convert to screen coordinates
                    hex_center_x = hex_center_x_norm * usable_width + margin
                    hex_center_y = hex_center_y_norm * usable_height + margin
                    hex_size_screen = hex_size * usable_width

                    # Calculate the 3 vertices of this triangle
                    angle_apex = math.pi / 2 - tri_idx * math.pi / 3
                    angle_base1 = math.pi / 2 - ((tri_idx - 1) % 6) * math.pi / 3
                    angle_base2 = math.pi / 2 - ((tri_idx + 1) % 6) * math.pi / 3

                    # Apex vertex
                    apex_x = hex_center_x + hex_size_screen * math.cos(angle_apex)
                    apex_y = hex_center_y - hex_size_screen * math.sin(angle_apex)

                    # Base vertices (adjacent hex vertices)
                    base1_x = hex_center_x + hex_size_screen * math.cos(angle_base1)
                    base1_y = hex_center_y - hex_size_screen * math.sin(angle_base1)

                    base2_x = hex_center_x + hex_size_screen * math.cos(angle_base2)
                    base2_y = hex_center_y - hex_size_screen * math.sin(angle_base2)

                    vertices = [
                        (apex_x, apex_y),
                        (base1_x, base1_y),
                        (base2_x, base2_y)
                    ]

                    pygame.draw.polygon(self.screen, LIGHT_GRAY, vertices)
                    pygame.draw.polygon(self.screen, BLACK, vertices, 2)

        # Calculate cell size for objects/agent
        if self.tiling_type == "square":
            cell_size = usable_width / self.grid_size
        elif self.tiling_type == "hex":
            # Use same calculation as hex rendering
            width_spacing = (self.grid_size - 1) if self.grid_size > 1 else 1
            height_spacing = (self.grid_size - 1) if self.grid_size > 1 else 1
            size_from_width = 0.95 / ((self.grid_size + 0.5) * math.sqrt(3)) if self.grid_size > 0 else 0.1
            size_from_height = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
            size = min(size_from_width, size_from_height)
            cell_size = size * usable_width
        else:  # triangle
            # Use triangle side length
            side_length = 0.95 * 2 / (self.grid_size + 0.5)
            cell_size = side_length * usable_width

        # Draw objects
        for obj in self.env.state.objects.values():
            if obj.cell_id:
                x_norm, y_norm = tiling.cell_to_canonical(obj.cell_id)
                x = x_norm * usable_width + margin
                y = y_norm * usable_height + margin

                color_map = {'red': RED, 'green': GREEN, 'blue': BLUE, 'yellow': YELLOW}
                draw_object(self.screen, (x, y), cell_size, color_map.get(obj.color, GRAY))

        # Draw agent
        agent_x_norm, agent_y_norm = tiling.cell_to_canonical(self.env.state.agent.cell_id)
        agent_x = agent_x_norm * usable_width + margin
        agent_y = agent_y_norm * usable_height + margin

        # Calculate facing angle - match direction vectors
        facing_dir = self.env.state.agent.get_facing_direction(tiling)
        angle_map_square = {
            "north": math.pi / 2,    # Up
            "east": 0,               # Right
            "south": -math.pi / 2,   # Down
            "west": math.pi          # Left
        }
        angle_map_hex = {
            "north": math.pi / 2,           # Up (0, -1)
            "northeast": math.pi / 6,       # Up-right (1, -1)
            "southeast": -math.pi / 6,      # Down-right (1, 0)
            "south": -math.pi / 2,          # Down (0, 1)
            "southwest": -5 * math.pi / 6,  # Down-left (-1, 1)
            "northwest": 5 * math.pi / 6    # Up-left (-1, 0)
        }
        angle_map_triangle = {
            "edge0": math.pi,        # Left
            "edge1": 0,              # Right
            "edge2": -math.pi / 2    # Down or Up depending on orientation
        }

        if self.tiling_type == "square":
            facing_angle = angle_map_square.get(facing_dir, 0)
        elif self.tiling_type == "hex":
            facing_angle = angle_map_hex.get(facing_dir, 0)
        else:
            facing_angle = angle_map_triangle.get(facing_dir, 0)

        draw_agent(self.screen, (agent_x, agent_y), cell_size, facing_angle)

        # Draw held object indicator above agent (adjusts with facing)
        if self.env.state.agent.holding:
            held_obj = self.env.state.agent.holding
            color_map = {'red': RED, 'green': GREEN, 'blue': BLUE, 'yellow': YELLOW}
            color = color_map.get(held_obj.color, GRAY)
            # Position held object in direction agent is facing
            held_x = agent_x + cell_size * 0.6 * math.cos(facing_angle)
            held_y = agent_y - cell_size * 0.6 * math.sin(facing_angle)
            pygame.draw.circle(self.screen, color, (int(held_x), int(held_y)), int(cell_size * 0.3))
            pygame.draw.circle(self.screen, BLACK, (int(held_x), int(held_y)), int(cell_size * 0.3), 2)

    def draw_info(self):
        """Draw information panel."""
        info_y = self.height + 10

        state = self.env.get_state_dict()

        # Title
        title = self.big_font.render(f"{self.tiling_type.upper()} GRID", True, BLACK)
        self.screen.blit(title, (10, info_y))

        # Info text
        info_texts = [
            f"Position: {state['agent']['cell_id']}",
            f"Facing: {state['agent']['facing_direction']}",
            f"Holding: {state['agent']['holding'] or 'Nothing'}",
            f"Steps: {self.env.steps}"
        ]

        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, BLACK)
            self.screen.blit(surface, (10, info_y + 40 + i * 25))

        # Controls
        controls = [
            "Arrow/WASD: Move | Q/E: Turn | SPACE: Pickup/Drop | P: Push",
            "1: Square | 2: Hex | 3: Triangle | R: Reset | ESC: Quit"
        ]

        for i, text in enumerate(controls):
            surface = self.font.render(text, True, DARK_GRAY)
            self.screen.blit(surface, (self.width // 2 + 10, info_y + 40 + i * 25))

    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_input()
            self.draw_grid()
            self.draw_info()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    demo = InteractiveDemo(width=800, height=800)
    demo.run()
