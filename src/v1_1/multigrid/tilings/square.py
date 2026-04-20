# tilings/square.py

from ..base import Tiling
from ..core import Cell
from typing import Optional


# Direction labels
DIRECTIONS = ["north", "east", "south", "west"]

# Direction index mapping
DIR_INDEX = {
    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3
}

# Direction vectors (row_delta, col_delta)
DIR_VECTORS = {
    "north": (-1, 0),   # Up (row decreases)
    "east":  (0, 1),    # Right (col increases)
    "south": (1, 0),    # Down (row increases)
    "west":  (0, -1)    # Left (col decreases)
}

# Opposite directions (for backward movement)
OPPOSITE = {
    "north": "south",
    "east": "west",
    "south": "north",
    "west": "east"
}


def row_col_to_cell_id(row: int, col: int) -> str:
    """Convert row,col to cell ID."""
    return f"sq_{row}_{col}"


def cell_id_to_row_col(cell_id: str) -> tuple[int, int]:
    """Parse cell ID to row,col."""
    _, row, col = cell_id.split("_")
    return int(row), int(col)


def canonical_to_row_col(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """
    Convert normalized [0,1] coordinates to grid row,col.

    Args:
        x: Horizontal position [0,1]
        y: Vertical position [0,1]
        width: Grid width in cells
        height: Grid height in cells

    Returns:
        (row, col) tuple
    """
    col = min(int(x * width), width - 1)
    row = min(int(y * height), height - 1)
    return row, col


def row_col_to_canonical(row: int, col: int, width: int, height: int) -> tuple[float, float]:
    """
    Convert grid row,col to normalized [0,1] coordinates (cell center).

    Returns:
        (x, y) tuple with x,y in [0,1]
    """
    x = (col + 0.5) / width
    y = (row + 0.5) / height
    return x, y


def get_neighbor(row: int, col: int, direction: str, width: int, height: int) -> Optional[tuple[int, int]]:
    """
    Get neighbor cell in given direction.

    Args:
        row, col: Current cell coordinates
        direction: One of "north", "east", "south", "west"
        width, height: Grid dimensions

    Returns:
        (new_row, new_col) or None if out of bounds
    """
    dr, dc = DIR_VECTORS[direction]
    new_row = row + dr
    new_col = col + dc

    # Bounds check
    if 0 <= new_row < height and 0 <= new_col < width:
        return new_row, new_col
    return None


def manhattan_distance(row1: int, col1: int, row2: int, col2: int) -> int:
    """
    Manhattan (L1) distance between two cells.
    This is the minimum number of moves without obstacles.
    """
    return abs(row1 - row2) + abs(col1 - col2)


class SquareTiling(Tiling):
    """Square tiling implementation."""

    @property
    def name(self) -> str:
        return "square"

    @property
    def directions(self) -> list[str]:
        return DIRECTIONS

    def generate_graph(self, width: int, height: int, seed: int = 0) -> dict[str, Cell]:
        """
        Generate square grid as adjacency graph.

        Args:
            width: Number of columns
            height: Number of rows
            seed: Random seed (unused for square grids, but kept for interface)

        Returns:
            Dictionary of cell_id -> Cell
        """
        self.width = width
        self.height = height
        self.cells = {}

        # Create all cells
        for row in range(height):
            for col in range(width):
                cell_id = row_col_to_cell_id(row, col)
                pos = row_col_to_canonical(row, col, width, height)

                self.cells[cell_id] = Cell(
                    id=cell_id,
                    neighbors={},
                    row=row,
                    col=col,
                    position_hint=pos
                )

        # Connect neighbors
        for row in range(height):
            for col in range(width):
                cell_id = row_col_to_cell_id(row, col)
                cell = self.cells[cell_id]

                for direction in self.directions:
                    neighbor_coords = get_neighbor(row, col, direction, width, height)
                    if neighbor_coords:
                        neighbor_id = row_col_to_cell_id(*neighbor_coords)
                        cell.neighbors[direction] = neighbor_id

        return self.cells

    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized coordinates to cell ID."""
        row, col = canonical_to_row_col(x, y, self.width, self.height)
        return row_col_to_cell_id(row, col)

    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized coordinates (cell center)."""
        row, col = cell_id_to_row_col(cell_id)
        return row_col_to_canonical(row, col, self.width, self.height)

    def get_neighbor(self, cell_id: str, direction: str) -> Optional[str]:
        """Get neighbor in given direction."""
        return self.cells[cell_id].neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Graph distance (hops) between cells."""
        row_a, col_a = cell_id_to_row_col(cell_a)
        row_b, col_b = cell_id_to_row_col(cell_b)
        return manhattan_distance(row_a, col_a, row_b, col_b)
