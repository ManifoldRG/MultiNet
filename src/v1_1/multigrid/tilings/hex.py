# tilings/hex.py

import math
from dataclasses import dataclass
from ..base import Tiling
from ..core import Cell
from typing import Optional


@dataclass
class AxialCoord:
    """Axial coordinates for hexagonal grids."""
    q: int
    r: int

    def __add__(self, other: "AxialCoord") -> "AxialCoord":
        return AxialCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: "AxialCoord") -> "AxialCoord":
        return AxialCoord(self.q - other.q, self.r - other.r)

    def __hash__(self):
        return hash((self.q, self.r))

    def __eq__(self, other):
        if not isinstance(other, AxialCoord):
            return False
        return self.q == other.q and self.r == other.r

    @property
    def s(self) -> int:
        """Implicit third coordinate."""
        return -self.q - self.r


@dataclass
class OffsetCoord:
    """Offset coordinates for hexagonal grids (odd-r layout)."""
    col: int
    row: int


# Direction labels (clockwise from north)
DIRECTIONS = ["north", "northeast", "southeast", "south", "southwest", "northwest"]

DIR_INDEX = {
    "north": 0,
    "northeast": 1,
    "southeast": 2,
    "south": 3,
    "southwest": 4,
    "northwest": 5
}

# Direction vectors in axial coordinates
# Pointy-top hex, starting from north (up), going clockwise
DIR_VECTORS_AXIAL = {
    "north":     AxialCoord(0, -1),
    "northeast": AxialCoord(1, -1),
    "southeast": AxialCoord(1, 0),
    "south":     AxialCoord(0, 1),
    "southwest": AxialCoord(-1, 1),
    "northwest": AxialCoord(-1, 0)
}

# Opposite directions
OPPOSITE = {
    "north": "south",
    "northeast": "southwest",
    "southeast": "northwest",
    "south": "north",
    "southwest": "northeast",
    "northwest": "southeast"
}


def offset_to_axial(offset: OffsetCoord) -> AxialCoord:
    """Convert odd-r offset to axial coordinates."""
    q = offset.col - (offset.row - (offset.row & 1)) // 2
    r = offset.row
    return AxialCoord(q, r)


def axial_to_offset(axial: AxialCoord) -> OffsetCoord:
    """Convert axial to odd-r offset coordinates."""
    col = axial.q + (axial.r - (axial.r & 1)) // 2
    row = axial.r
    return OffsetCoord(col, row)


def axial_to_cell_id(coord: AxialCoord) -> str:
    """Convert axial coordinates to cell ID."""
    return f"hex_{coord.q}_{coord.r}"


def cell_id_to_axial(cell_id: str) -> AxialCoord:
    """Parse cell ID to axial coordinates."""
    _, q, r = cell_id.split("_")
    return AxialCoord(int(q), int(r))


def axial_round(q_frac: float, r_frac: float) -> AxialCoord:
    """Round fractional axial coordinates to nearest hex."""
    s_frac = -q_frac - r_frac

    q = round(q_frac)
    r = round(r_frac)
    s = round(s_frac)

    q_diff = abs(q - q_frac)
    r_diff = abs(r - r_frac)
    s_diff = abs(s - s_frac)

    # Reset the component with largest rounding error
    if q_diff > r_diff and q_diff > s_diff:
        q = -r - s
    elif r_diff > s_diff:
        r = -q - s
    # else: s = -q - r (implicit, we don't store s)

    return AxialCoord(q, r)


def axial_distance(a: AxialCoord, b: AxialCoord) -> int:
    """Distance in axial coordinates (derived from cube)."""
    return (
        abs(a.q - b.q) +
        abs(a.q + a.r - b.q - b.r) +
        abs(a.r - b.r)
    ) // 2


class HexTiling(Tiling):
    """Hexagonal tiling implementation with pointy-top orientation."""

    def __init__(self):
        super().__init__()
        self._bounds: set[AxialCoord] = set()

    @property
    def name(self) -> str:
        return "hex"

    @property
    def directions(self) -> list[str]:
        return DIRECTIONS

    def generate_graph(self, width: int, height: int, seed: int = 0) -> dict[str, Cell]:
        """
        Generate hexagonal grid as adjacency graph.

        Creates a rectangular region of hexes using offset coordinates
        for layout, then converts to axial for math.

        Args:
            width: Number of columns
            height: Number of rows
            seed: Random seed (unused for regular grids)

        Returns:
            Dictionary of cell_id -> Cell
        """
        self.width = width
        self.height = height
        self.cells = {}
        self._bounds = set()

        # Create cells using offset coordinates for rectangular layout
        for row in range(height):
            for col in range(width):
                offset = OffsetCoord(col, row)
                axial = offset_to_axial(offset)

                cell_id = axial_to_cell_id(axial)
                pos = self._axial_to_normalized(axial)

                self.cells[cell_id] = Cell(
                    id=cell_id,
                    neighbors={},
                    row=row,
                    col=col,
                    position_hint=pos,
                    tiling_coords=axial
                )
                self._bounds.add(axial)

        # Connect neighbors
        for cell_id, cell in self.cells.items():
            axial = cell.tiling_coords
            for direction, delta in DIR_VECTORS_AXIAL.items():
                neighbor_axial = axial + delta
                if neighbor_axial in self._bounds:
                    neighbor_id = axial_to_cell_id(neighbor_axial)
                    cell.neighbors[direction] = neighbor_id

        return self.cells

    def _axial_to_normalized(self, axial: AxialCoord) -> tuple[float, float]:
        """Convert axial to normalized [0,1] coordinates for rendering."""
        # Convert axial back to offset coordinates for positioning
        offset = axial_to_offset(axial)
        col, row = offset.col, offset.row

        # For pointy-top hexagons in odd-r offset layout:
        # - Horizontal spacing between columns: sqrt(3) * size
        # - Vertical spacing between rows: 3/2 * size
        # - Odd rows are offset by sqrt(3)/2 * size to the right

        # Calculate size to fit grid in [0,1] space with margin
        width_spacing = (self.width - 1) if self.width > 1 else 1
        height_spacing = (self.height - 1) if self.height > 1 else 1

        # Account for odd-row offset in horizontal extent
        # Max horizontal extent is width * sqrt(3) * size + (for odd row) sqrt(3)/2 * size
        # = (width + 0.5) * sqrt(3) * size
        size_from_width = 0.95 / ((self.width + 0.5) * math.sqrt(3)) if self.width > 0 else 0.1
        size_from_height = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
        size = min(size_from_width, size_from_height)

        # Position hex based on offset coordinates
        x = col * math.sqrt(3) * size
        y = row * 1.5 * size

        # Odd rows are shifted right by sqrt(3)/2 * size
        if row % 2 == 1:
            x += math.sqrt(3) / 2 * size

        # Center the grid
        grid_width = (self.width + 0.5) * math.sqrt(3) * size
        grid_height = (self.height - 0.5) * 1.5 * size

        x_offset = (1.0 - grid_width) / 2
        y_offset = (1.0 - grid_height) / 2

        return x + x_offset, y + y_offset

    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized coordinates to nearest cell ID."""
        # Calculate size (same as in _axial_to_normalized)
        width_spacing = (self.width - 1) if self.width > 1 else 1
        height_spacing = (self.height - 1) if self.height > 1 else 1

        size_from_width = 0.95 / ((self.width + 0.5) * math.sqrt(3)) if self.width > 0 else 0.1
        size_from_height = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
        size = min(size_from_width, size_from_height)

        # Calculate grid offset
        grid_width = (self.width + 0.5) * math.sqrt(3) * size
        grid_height = (self.height - 0.5) * 1.5 * size
        x_offset = (1.0 - grid_width) / 2
        y_offset = (1.0 - grid_height) / 2

        # Reverse the transformation
        px = (x - x_offset) / size
        py = (y - y_offset) / size

        # Pixel to fractional offset coordinates
        # Account for odd-row shifting
        row_frac = py / 1.5
        row = round(row_frac)

        # If odd row, subtract the offset before calculating column
        x_adjusted = px
        if row % 2 == 1:
            x_adjusted -= math.sqrt(3) / 2

        col_frac = x_adjusted / math.sqrt(3)
        col = round(col_frac)

        # Clamp to valid bounds
        col = max(0, min(self.width - 1, col))
        row = max(0, min(self.height - 1, row))

        # Convert to axial
        offset = OffsetCoord(col, row)
        axial = offset_to_axial(offset)

        return axial_to_cell_id(axial)

    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized coordinates (hex center)."""
        axial = cell_id_to_axial(cell_id)
        return self._axial_to_normalized(axial)

    def get_neighbor(self, cell_id: str, direction: str) -> Optional[str]:
        """Get neighbor in given direction."""
        return self.cells[cell_id].neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Graph distance (hops) between cells."""
        axial_a = cell_id_to_axial(cell_a)
        axial_b = cell_id_to_axial(cell_b)
        return axial_distance(axial_a, axial_b)
