# tilings/triangle.py

import math
from ..base import Tiling
from ..core import Cell
from typing import Optional
from .hex import HexTiling, offset_to_axial, axial_to_offset, OffsetCoord, AxialCoord, DIR_VECTORS_AXIAL
from .hex import DIRECTIONS as HEX_DIRECTIONS


# Direction labels for triangular tiling
# Each triangle has 3 edges
DIRECTIONS = ["edge0", "edge1", "edge2"]

DIR_INDEX = {
    "edge0": 0,
    "edge1": 1,
    "edge2": 2
}


def parse_triangle_id(cell_id: str) -> tuple[int, int, int]:
    """Parse triangle cell ID to (hex_col, hex_row, tri_index)."""
    _, hex_col, hex_row, tri_idx = cell_id.split("_")
    return int(hex_col), int(hex_row), int(tri_idx)


def make_triangle_id(hex_col: int, hex_row: int, tri_index: int) -> str:
    """Create triangle cell ID from hex position and triangle index."""
    return f"tri_{hex_col}_{hex_row}_{tri_index}"


class TriangleTiling(Tiling):
    """Triangular tiling by subdividing hexagons into 6 triangles each."""

    @property
    def name(self) -> str:
        return "triangle"

    @property
    def directions(self) -> list[str]:
        return DIRECTIONS

    def generate_graph(self, width: int, height: int, seed: int = 0) -> dict[str, Cell]:
        """
        Generate triangular grid by subdividing hexagons.

        Each hexagon is divided into 6 triangles radiating from its center.
        Triangles are numbered 0-5 going counterclockwise from north.

        Args:
            width: Number of hex columns
            height: Number of hex rows
            seed: Random seed (unused)

        Returns:
            Dictionary of cell_id -> Cell
        """
        self.width = width
        self.height = height
        self.cells = {}

        # First create the underlying hex grid to get positions
        hex_tiling = HexTiling()
        hex_tiling.generate_graph(width, height, seed)

        # For each hexagon, create 6 triangles
        for hex_col in range(width):
            for hex_row in range(height):
                # Get hex center position
                offset = OffsetCoord(hex_col, hex_row)
                axial = offset_to_axial(offset)
                hex_center = hex_tiling._axial_to_normalized(axial)

                # Calculate hex size
                width_spacing = (width - 1) if width > 1 else 1
                height_spacing = (height - 1) if height > 1 else 1
                size_from_width = 0.95 / ((width + 0.5) * math.sqrt(3))
                size_from_height = 0.95 / (height_spacing * 1.5)
                hex_size = min(size_from_width, size_from_height)

                # Create 6 triangles for this hex
                for tri_idx in range(6):
                    cell_id = make_triangle_id(hex_col, hex_row, tri_idx)

                    # Triangle center is 2/3 of the way from hex center to vertex
                    angle = math.pi / 2 - tri_idx * math.pi / 3  # Start from north, go counterclockwise
                    vertex_x = hex_center[0] + hex_size * math.cos(angle)
                    vertex_y = hex_center[1] - hex_size * math.sin(angle)

                    # Centroid is 1/3 from base (at hex center) to apex (at vertex)
                    tri_center_x = hex_center[0] + (vertex_x - hex_center[0]) * (2/3)
                    tri_center_y = hex_center[1] + (vertex_y - hex_center[1]) * (2/3)

                    self.cells[cell_id] = Cell(
                        id=cell_id,
                        neighbors={},
                        row=hex_row,
                        col=hex_col,
                        position_hint=(tri_center_x, tri_center_y),
                        tiling_coords={"hex_center": hex_center, "tri_idx": tri_idx, "hex_size": hex_size}
                    )

        # Connect neighbors
        # Within a hex: triangles share edges with adjacent triangles
        # Between hexes: triangles share edges with triangles in adjacent hexes
        for hex_col in range(width):
            for hex_row in range(height):
                for tri_idx in range(6):
                    cell_id = make_triangle_id(hex_col, hex_row, tri_idx)
                    cell = self.cells[cell_id]

                    # edge0: counterclockwise triangle in same hex
                    prev_tri = (tri_idx - 1) % 6
                    neighbor_id = make_triangle_id(hex_col, hex_row, prev_tri)
                    cell.neighbors["edge0"] = neighbor_id

                    # edge1: clockwise triangle in same hex
                    next_tri = (tri_idx + 1) % 6
                    neighbor_id = make_triangle_id(hex_col, hex_row, next_tri)
                    cell.neighbors["edge1"] = neighbor_id

                    # edge2: triangle in adjacent hex (if it exists)
                    # Each triangle points toward one of the 6 hex directions
                    # Get the hex neighbor in that direction
                    offset = OffsetCoord(hex_col, hex_row)
                    axial = offset_to_axial(offset)

                    # Direction mapping: triangle 0 points north, etc.
                    hex_direction = HEX_DIRECTIONS[tri_idx]
                    delta = DIR_VECTORS_AXIAL[hex_direction]
                    neighbor_axial = axial + delta

                    # Check if neighbor hex exists
                    neighbor_offset = axial_to_offset(neighbor_axial)
                    if 0 <= neighbor_offset.col < width and 0 <= neighbor_offset.row < height:
                        # The outer edge of triangle tri_idx in this hex
                        # connects to the triangle pointing back in the opposite direction
                        opposite_tri = (tri_idx + 3) % 6
                        neighbor_id = make_triangle_id(neighbor_offset.col, neighbor_offset.row, opposite_tri)
                        if neighbor_id in self.cells:
                            cell.neighbors["edge2"] = neighbor_id

        return self.cells

    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized coordinates to nearest triangle cell ID."""
        # Find nearest hex first
        hex_tiling = HexTiling()
        hex_tiling.generate_graph(self.width, self.height)
        hex_cell_id = hex_tiling.canonical_to_cell(x, y)

        # Parse hex position from ID
        _, hex_q, hex_r = hex_cell_id.split("_")
        offset = axial_to_offset(AxialCoord(int(hex_q), int(hex_r)))
        hex_col, hex_row = offset.col, offset.row

        # Get hex center
        axial = offset_to_axial(OffsetCoord(hex_col, hex_row))
        hex_center = hex_tiling._axial_to_normalized(axial)

        # Determine which triangle based on angle from hex center
        dx = x - hex_center[0]
        dy = y - hex_center[1]
        angle = math.atan2(-dy, dx)  # Note: -dy because y increases downward

        # Convert angle to triangle index (0-5, starting from north counterclockwise)
        # North is at angle π/2
        adjusted_angle = (math.pi / 2 - angle) % (2 * math.pi)
        tri_idx = int(adjusted_angle / (math.pi / 3)) % 6

        return make_triangle_id(hex_col, hex_row, tri_idx)

    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized coordinates (triangle center)."""
        if cell_id in self.cells:
            return self.cells[cell_id].position_hint
        # Fallback
        return (0.5, 0.5)

    def get_neighbor(self, cell_id: str, direction: str) -> Optional[str]:
        """Get neighbor in given direction."""
        return self.cells[cell_id].neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Graph distance (hops) between cells using BFS."""
        if cell_a == cell_b:
            return 0

        from collections import deque
        visited = {cell_a}
        queue = deque([(cell_a, 0)])

        while queue:
            current, dist = queue.popleft()
            if current == cell_b:
                return dist

            cell = self.cells[current]
            for neighbor_id in cell.neighbors.values():
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, dist + 1))

        return 999
