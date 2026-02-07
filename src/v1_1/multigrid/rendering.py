# multigrid/rendering.py

"""
Rendering System for MultiGrid Environments

Provides vector-based rendering for all tiling types (square, hex, triangle).
Uses PIL for high-quality polygon drawing suitable for VLM evaluation.
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from PIL import Image, ImageDraw

from .objects.base import WorldObj
from .core import Cell


# Color palette for rendering
COLORS = {
    "background": (245, 245, 245),  # Light gray
    "grid_line": (200, 200, 200),   # Gray
    "wall": (64, 64, 64),           # Dark gray
    "agent": (0, 100, 200),         # Blue
    "goal": (0, 200, 0),            # Green
    "red": (255, 60, 60),
    "green": (60, 200, 60),
    "blue": (60, 60, 255),
    "yellow": (255, 255, 60),
    "purple": (160, 60, 200),
    "orange": (255, 165, 60),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "cyan": (60, 200, 200),
}


class Renderer(ABC):
    """Abstract renderer supporting multiple visual styles."""

    @abstractmethod
    def begin_frame(self, width: int, height: int) -> None:
        """Start a new frame."""
        pass

    @abstractmethod
    def draw_cell_background(
        self,
        vertices: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        outline: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """Draw cell polygon background."""
        pass

    @abstractmethod
    def draw_object(
        self,
        center: Tuple[float, float],
        obj: WorldObj,
        size: float
    ) -> None:
        """Draw an object at given position."""
        pass

    @abstractmethod
    def draw_agent(
        self,
        center: Tuple[float, float],
        facing: float,  # Angle in radians
        size: float,
        holding: Optional[WorldObj] = None
    ) -> None:
        """Draw the agent."""
        pass

    @abstractmethod
    def draw_goal(
        self,
        center: Tuple[float, float],
        size: float
    ) -> None:
        """Draw the goal marker."""
        pass

    @abstractmethod
    def end_frame(self) -> np.ndarray:
        """Finish frame and return RGB array."""
        pass


class MinimalRenderer(Renderer):
    """Clean vector-based rendering for VLM evaluation using PIL."""

    def __init__(self):
        self.img: Optional[Image.Image] = None
        self.draw: Optional[ImageDraw.ImageDraw] = None
        self.width = 0
        self.height = 0

    def begin_frame(self, width: int, height: int) -> None:
        """Start a new frame."""
        self.width = width
        self.height = height
        self.img = Image.new('RGB', (width, height), COLORS["background"])
        self.draw = ImageDraw.Draw(self.img)

    def draw_cell_background(
        self,
        vertices: List[Tuple[float, float]],
        color: Tuple[int, int, int],
        outline: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """Draw cell polygon background."""
        if self.draw is None:
            return

        # Convert to pixel coordinates
        pixel_vertices = [(int(x), int(y)) for x, y in vertices]

        if outline is None:
            outline = COLORS["grid_line"]

        self.draw.polygon(pixel_vertices, fill=color, outline=outline)

    def draw_object(
        self,
        center: Tuple[float, float],
        obj: WorldObj,
        size: float
    ) -> None:
        """Draw an object at given position."""
        if self.draw is None:
            return

        x, y = int(center[0]), int(center[1])
        color = self._color_name_to_rgb(obj.color)
        r = int(size * 0.4)

        obj_type = obj.obj_type

        if obj_type == "wall":
            # Draw wall as filled square
            self.draw.rectangle(
                [x - r, y - r, x + r, y + r],
                fill=COLORS["wall"],
                outline=COLORS["black"]
            )

        elif obj_type == "movable":
            # Draw movable as circle
            self.draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=color,
                outline=COLORS["black"]
            )

        elif obj_type == "zone":
            # Draw zone as semi-transparent circle (just outline)
            self.draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=None,
                outline=color,
                width=2
            )

        elif obj_type == "key":
            # Draw key as a small circle with a stem (simplified key shape)
            key_head_r = int(r * 0.5)
            stem_width = int(r * 0.2)
            # Key head (circle)
            self.draw.ellipse(
                [x - key_head_r, y - r, x + key_head_r, y - r + key_head_r * 2],
                fill=color,
                outline=COLORS["black"]
            )
            # Key stem (rectangle)
            self.draw.rectangle(
                [x - stem_width, y, x + stem_width, y + r],
                fill=color,
                outline=COLORS["black"]
            )
            # Key teeth
            tooth_y = y + int(r * 0.5)
            self.draw.rectangle(
                [x, tooth_y, x + int(r * 0.3), tooth_y + int(r * 0.2)],
                fill=color
            )

        elif obj_type == "door":
            # Draw door as vertical rectangle with handle
            door_width = int(r * 0.6)
            # Check if door is open/locked
            is_open = getattr(obj, 'is_open', False)
            is_locked = getattr(obj, 'is_locked', True)

            if is_open:
                # Open door - just an outline
                self.draw.rectangle(
                    [x - door_width, y - r, x + door_width, y + r],
                    fill=None,
                    outline=color,
                    width=2
                )
            else:
                # Closed door - filled
                self.draw.rectangle(
                    [x - door_width, y - r, x + door_width, y + r],
                    fill=color,
                    outline=COLORS["black"]
                )
                # Draw lock indicator if locked
                if is_locked:
                    lock_r = int(r * 0.2)
                    self.draw.ellipse(
                        [x - lock_r, y - lock_r, x + lock_r, y + lock_r],
                        fill=COLORS["black"]
                    )

        elif obj_type == "switch":
            # Draw switch as a small square with indicator
            switch_r = int(r * 0.5)
            is_active = getattr(obj, 'is_active', False)

            # Base
            self.draw.rectangle(
                [x - switch_r, y - switch_r, x + switch_r, y + switch_r],
                fill=COLORS["grey"],
                outline=COLORS["black"]
            )
            # Indicator (lit if active)
            indicator_r = int(r * 0.25)
            indicator_color = color if is_active else COLORS["black"]
            self.draw.ellipse(
                [x - indicator_r, y - indicator_r, x + indicator_r, y + indicator_r],
                fill=indicator_color
            )

        elif obj_type == "gate":
            # Draw gate as vertical bars
            is_open = getattr(obj, 'is_open', False)
            bar_width = int(r * 0.15)
            num_bars = 3

            if is_open:
                # Open gate - bars to the side
                for i in range(num_bars):
                    bar_x = x + r + i * bar_width * 2
                    self.draw.rectangle(
                        [bar_x, y - r, bar_x + bar_width, y + r],
                        fill=color,
                        outline=COLORS["black"]
                    )
            else:
                # Closed gate - bars blocking
                spacing = (r * 2) // (num_bars + 1)
                for i in range(num_bars):
                    bar_x = x - r + spacing * (i + 1)
                    self.draw.rectangle(
                        [bar_x - bar_width, y - r, bar_x + bar_width, y + r],
                        fill=color,
                        outline=COLORS["black"]
                    )

        elif obj_type == "hazard":
            # Draw hazard as warning triangle or lava pool
            hazard_type = getattr(obj, 'hazard_type', 'lava')
            if hazard_type == "lava":
                # Lava - wavy orange/red
                self.draw.ellipse(
                    [x - r, y - int(r * 0.5), x + r, y + int(r * 0.5)],
                    fill=COLORS["orange"],
                    outline=COLORS["red"]
                )
            else:
                # Generic hazard - warning triangle
                triangle = [
                    (x, y - r),
                    (x + r, y + r),
                    (x - r, y + r)
                ]
                self.draw.polygon(triangle, fill=COLORS["red"], outline=COLORS["black"])
                # Exclamation mark
                self.draw.rectangle(
                    [x - 2, y - int(r * 0.3), x + 2, y + int(r * 0.2)],
                    fill=COLORS["black"]
                )
                self.draw.ellipse(
                    [x - 2, y + int(r * 0.4), x + 2, y + int(r * 0.6)],
                    fill=COLORS["black"]
                )

        elif obj_type == "teleporter":
            # Draw teleporter as concentric circles (portal)
            for i in range(3, 0, -1):
                ring_r = int(r * i / 3)
                ring_color = color if i % 2 == 1 else COLORS["white"]
                self.draw.ellipse(
                    [x - ring_r, y - ring_r, x + ring_r, y + ring_r],
                    fill=ring_color,
                    outline=COLORS["black"] if i == 3 else None
                )

        else:
            # Default: draw as diamond
            diamond = [
                (x, y - r),
                (x + r, y),
                (x, y + r),
                (x - r, y)
            ]
            self.draw.polygon(diamond, fill=color, outline=COLORS["black"])

    def draw_agent(
        self,
        center: Tuple[float, float],
        facing: float,  # Angle in radians
        size: float,
        holding: Optional[WorldObj] = None
    ) -> None:
        """Draw the agent as a triangle pointing in facing direction."""
        if self.draw is None:
            return

        x, y = center[0], center[1]
        r = size * 0.5

        # Triangle vertices relative to center, pointing in facing direction
        # Tip at front, base at back
        tip_angle = facing
        base_angle_1 = facing + math.pi * 2 / 3
        base_angle_2 = facing - math.pi * 2 / 3

        tip = (x + r * math.cos(tip_angle), y + r * math.sin(tip_angle))
        base1 = (x + r * 0.6 * math.cos(base_angle_1), y + r * 0.6 * math.sin(base_angle_1))
        base2 = (x + r * 0.6 * math.cos(base_angle_2), y + r * 0.6 * math.sin(base_angle_2))

        triangle = [
            (int(tip[0]), int(tip[1])),
            (int(base1[0]), int(base1[1])),
            (int(base2[0]), int(base2[1]))
        ]

        self.draw.polygon(triangle, fill=COLORS["agent"], outline=COLORS["black"])

        # If holding something, draw a small indicator
        if holding is not None:
            carry_r = int(r * 0.25)
            carry_x = int(x)
            carry_y = int(y)
            carry_color = self._color_name_to_rgb(holding.color)
            self.draw.ellipse(
                [carry_x - carry_r, carry_y - carry_r, carry_x + carry_r, carry_y + carry_r],
                fill=carry_color,
                outline=COLORS["white"]
            )

    def draw_goal(
        self,
        center: Tuple[float, float],
        size: float
    ) -> None:
        """Draw the goal marker as a star."""
        if self.draw is None:
            return

        x, y = int(center[0]), int(center[1])
        r = int(size * 0.4)

        # Draw as filled green square with border
        self.draw.rectangle(
            [x - r, y - r, x + r, y + r],
            fill=COLORS["goal"],
            outline=COLORS["black"]
        )

    def end_frame(self) -> np.ndarray:
        """Finish frame and return RGB array."""
        if self.img is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return np.array(self.img)

    def _color_name_to_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to RGB tuple."""
        return COLORS.get(color_name.lower(), COLORS["grey"])


def get_square_vertices(
    center: Tuple[float, float],
    size: float
) -> List[Tuple[float, float]]:
    """Get vertices for a square cell."""
    x, y = center
    half = size / 2
    return [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half)
    ]


def get_hex_vertices(
    center: Tuple[float, float],
    size: float
) -> List[Tuple[float, float]]:
    """Get vertices for a pointy-top hexagon."""
    x, y = center
    vertices = []
    for i in range(6):
        angle = math.pi / 2 - i * math.pi / 3  # Start from top, go clockwise
        vx = x + size * math.cos(angle)
        vy = y - size * math.sin(angle)  # Flip y
        vertices.append((vx, vy))
    return vertices


def get_triangle_vertices(
    hex_center: Tuple[float, float],
    hex_size: float,
    triangle_index: int
) -> List[Tuple[float, float]]:
    """Get vertices for a triangle within a hexagon."""
    cx, cy = hex_center

    # Vertices of the hexagon
    hex_vertices = []
    for i in range(6):
        angle = math.pi / 2 - i * math.pi / 3
        vx = cx + hex_size * math.cos(angle)
        vy = cy - hex_size * math.sin(angle)
        hex_vertices.append((vx, vy))

    # Triangle i uses: center, vertex i, vertex (i+1)%6
    return [
        (cx, cy),
        hex_vertices[triangle_index],
        hex_vertices[(triangle_index + 1) % 6]
    ]


def _dim_color(color: Tuple[int, int, int], factor: float = 0.4) -> Tuple[int, int, int]:
    """Dim a color by blending it toward dark gray."""
    return tuple(int(c * factor) for c in color)


def render_multigrid(
    state,  # WorldState
    tiling,  # Tiling
    width: int = 640,
    height: int = 640,
    goal_cell_id: Optional[str] = None,
    visible_cells: Optional[set] = None,
    explored_cells: Optional[set] = None,
) -> np.ndarray:
    """
    Render a MultiGrid world state to an RGB image.

    Args:
        state: WorldState object
        tiling: Tiling object
        width: Output image width
        height: Output image height
        goal_cell_id: Optional cell ID to mark as goal
        visible_cells: Set of currently visible cell IDs (None = all visible)
        explored_cells: Set of previously explored cell IDs (None = all explored)

    Returns:
        RGB numpy array of shape (height, width, 3)
    """
    renderer = MinimalRenderer()
    renderer.begin_frame(width, height)

    # Calculate cell size based on tiling type and canvas size
    tiling_name = tiling.name
    margin = 0.05
    usable_width = width * (1 - 2 * margin)
    usable_height = height * (1 - 2 * margin)
    offset_x = width * margin
    offset_y = height * margin

    # Draw all cells
    for cell_id, cell in tiling.cells.items():
        # Get canonical position and convert to pixel coordinates
        pos = cell.position_hint
        px = offset_x + pos[0] * usable_width
        py = offset_y + pos[1] * usable_height

        # Calculate cell size
        if tiling_name == "square":
            num_cells = max(tiling.width, tiling.height)
            cell_size = min(usable_width, usable_height) / num_cells * 0.9
            vertices = get_square_vertices((px, py), cell_size)
        elif tiling_name == "hex":
            hex_size = min(usable_width, usable_height) / (tiling.height * 2) * 0.9
            vertices = get_hex_vertices((px, py), hex_size)
        elif tiling_name == "triangle":
            # Use stored tiling_coords for accurate rendering
            tc = cell.tiling_coords
            if tc is not None:
                hc = tc["hex_center"]
                tri_idx = tc["tri_idx"]
                hex_size_norm = tc["hex_size"]
                # Convert hex center from normalized to pixel coords
                hc_px = offset_x + hc[0] * usable_width
                hc_py = offset_y + hc[1] * usable_height
                # Scale hex size from normalized to pixel space
                hex_size_px = hex_size_norm * min(usable_width, usable_height)
            else:
                # Fallback for cells without tiling_coords
                hc_px, hc_py = px, py
                hex_size_px = min(usable_width, usable_height) / (tiling.height * 2) * 0.9
                _, _, _, tri_idx_str = cell_id.split("_")
                tri_idx = int(tri_idx_str)
            vertices = get_triangle_vertices((hc_px, hc_py), hex_size_px, tri_idx)
        elif tiling_name in ("3464", "488"):
            # Archimedean tilings: read pre-computed vertices from tiling_coords
            tc = cell.tiling_coords
            if tc is not None and "vertices" in tc:
                # Vertices are in normalized [0,1] space; scale to pixel space
                vertices = [
                    (offset_x + vx * usable_width, offset_y + vy * usable_height)
                    for vx, vy in tc["vertices"]
                ]
            else:
                # Fallback: draw a small square at the position hint
                cell_size = min(usable_width, usable_height) / 10
                vertices = get_square_vertices((px, py), cell_size)
        else:
            # Fallback to square
            cell_size = min(usable_width, usable_height) / 10
            vertices = get_square_vertices((px, py), cell_size)

        # Determine cell color
        if goal_cell_id and cell_id == goal_cell_id:
            color = COLORS["goal"]
        else:
            color = COLORS["background"]

        # Apply partial observability dimming
        if visible_cells is not None and cell_id not in visible_cells:
            if explored_cells is not None and cell_id in explored_cells:
                # Previously explored but not currently visible: dim
                color = _dim_color(color)
            else:
                # Never explored: dark background
                color = (30, 30, 30)

        renderer.draw_cell_background(vertices, color)

    # Calculate object/agent size
    if tiling_name == "square":
        obj_size = min(usable_width, usable_height) / max(tiling.width, tiling.height) * 0.7
    elif tiling_name == "hex":
        obj_size = min(usable_width, usable_height) / (tiling.height * 2) * 0.8
    elif tiling_name in ("3464", "488"):
        # Archimedean tilings: estimate size from total cell count
        num_cells = max(len(tiling.cells), 1)
        # Approximate: tiles_per_row ~ sqrt(num_cells * aspect_ratio)
        tiles_per_side = max(math.sqrt(num_cells), 1)
        obj_size = min(usable_width, usable_height) / tiles_per_side * 0.5
    else:
        obj_size = min(usable_width, usable_height) / (tiling.height * 3) * 0.8

    # Draw objects (skip non-visible cells)
    for obj_id, obj in state.objects.items():
        if obj.cell_id is None:
            continue
        if visible_cells is not None and obj.cell_id not in visible_cells:
            continue
        cell = tiling.cells.get(obj.cell_id)
        if cell is None:
            continue

        pos = cell.position_hint
        px = offset_x + pos[0] * usable_width
        py = offset_y + pos[1] * usable_height
        renderer.draw_object((px, py), obj, obj_size)

    # Draw goal marker (skip if not visible)
    if goal_cell_id and goal_cell_id in tiling.cells:
        if visible_cells is None or goal_cell_id in visible_cells:
            goal_cell = tiling.cells[goal_cell_id]
            pos = goal_cell.position_hint
            px = offset_x + pos[0] * usable_width
            py = offset_y + pos[1] * usable_height
            renderer.draw_goal((px, py), obj_size)

    # Draw agent
    agent_cell = tiling.cells.get(state.agent.cell_id)
    if agent_cell is not None:
        pos = agent_cell.position_hint
        px = offset_x + pos[0] * usable_width
        py = offset_y + pos[1] * usable_height

        # Calculate facing angle
        num_dirs = len(tiling.directions)
        # Facing 0 = first direction (e.g., north for hex, edge0 for triangle)
        facing_angle = -state.agent.facing * (2 * math.pi / num_dirs)

        # Adjust based on tiling orientation
        if tiling_name == "square":
            # Square: 0=north, 1=east, 2=south, 3=west
            facing_angle = -math.pi / 2 - state.agent.facing * (math.pi / 2)
        elif tiling_name == "hex":
            # Hex: 0=north, 1=northeast, etc.
            facing_angle = -math.pi / 2 - state.agent.facing * (math.pi / 3)

        renderer.draw_agent((px, py), facing_angle, obj_size, state.agent.holding)

    return renderer.end_frame()
