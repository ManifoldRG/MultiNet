# Appendix A: Square Tiling

**Status:** Implementation-Ready (PoC baseline)

## A.1 Overview

The square tiling is the simplest regular tiling, using squares that meet four at each vertex. While this is the most common grid in AI training data (MiniGrid, NetHack, Pokémon, etc.), it serves as:

1. **Proof of Concept**: Validate the adjacency graph architecture
2. **Baseline**: Compare model performance on familiar vs novel tilings
3. **Foundation**: Other tilings build on similar patterns

## A.2 Coordinate System

### A.2.1 Primary Coordinates (Row, Column)

```
     0   1   2   3   4   (column)
   +---+---+---+---+---+
 0 |   |   |   |   |   |
   +---+---+---+---+---+
 1 |   |   | X |   |   |   X is at (row=1, col=2)
   +---+---+---+---+---+
 2 |   |   |   |   |   |
   +---+---+---+---+---+
(row)
```

- **Cell ID format**: `sq_{row}_{col}` (e.g., `sq_1_2`)
- **Origin**: Top-left corner (row=0, col=0)
- **Row**: Increases downward (y-axis inverted from Cartesian)
- **Column**: Increases rightward

### A.2.2 Coordinate Conversions

```python
def row_col_to_cell_id(row: int, col: int) -> str:
    """Convert row,col to cell ID."""
    return f"sq_{row}_{col}"

def cell_id_to_row_col(cell_id: str) -> tuple[int, int]:
    """Parse cell ID to row,col."""
    _, row, col = cell_id.split("_")
    return int(row), int(col)

def canonical_to_row_col(
    x: float, y: float,
    width: int, height: int
) -> tuple[int, int]:
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

def row_col_to_canonical(
    row: int, col: int,
    width: int, height: int
) -> tuple[float, float]:
    """
    Convert grid row,col to normalized [0,1] coordinates (cell center).

    Returns:
        (x, y) tuple with x,y in [0,1]
    """
    x = (col + 0.5) / width
    y = (row + 0.5) / height
    return x, y
```

## A.3 Directions and Neighbors

### A.3.1 Direction Labels

Square grids support 4 cardinal directions:

```python
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
```

### A.3.2 Neighbor Computation

```python
def get_neighbor(
    row: int, col: int,
    direction: str,
    width: int, height: int
) -> tuple[int, int] | None:
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

def get_all_neighbors(
    row: int, col: int,
    width: int, height: int
) -> dict[str, tuple[int, int]]:
    """Get all valid neighbors with their direction labels."""
    neighbors = {}
    for direction in DIRECTIONS:
        neighbor = get_neighbor(row, col, direction, width, height)
        if neighbor is not None:
            neighbors[direction] = neighbor
    return neighbors
```

### A.3.3 Turn Operations

```python
def turn_left(facing: int) -> int:
    """Rotate facing counter-clockwise."""
    return (facing - 1) % 4

def turn_right(facing: int) -> int:
    """Rotate facing clockwise."""
    return (facing + 1) % 4

def get_facing_direction(facing: int) -> str:
    """Get direction label for facing index."""
    return DIRECTIONS[facing]
```

## A.4 Distance and Pathfinding

### A.4.1 Manhattan Distance

```python
def manhattan_distance(
    row1: int, col1: int,
    row2: int, col2: int
) -> int:
    """
    Manhattan (L1) distance between two cells.
    This is the minimum number of moves without obstacles.
    """
    return abs(row1 - row2) + abs(col1 - col2)
```

### A.4.2 Euclidean Distance (for canonical coordinates)

```python
import math

def euclidean_distance(
    x1: float, y1: float,
    x2: float, y2: float
) -> float:
    """Euclidean distance in canonical coordinates."""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
```

### A.4.3 Line Drawing (Bresenham)

```python
def bresenham_line(
    row1: int, col1: int,
    row2: int, col2: int
) -> list[tuple[int, int]]:
    """
    Generate cells along a line using Bresenham's algorithm.
    Used for line-of-sight and projectile paths.
    """
    cells = []
    dr = abs(row2 - row1)
    dc = abs(col2 - col1)
    row, col = row1, col1
    row_step = 1 if row1 < row2 else -1
    col_step = 1 if col1 < col2 else -1

    if dc > dr:
        err = dc // 2
        while col != col2:
            cells.append((row, col))
            err -= dr
            if err < 0:
                row += row_step
                err += dc
            col += col_step
    else:
        err = dr // 2
        while row != row2:
            cells.append((row, col))
            err -= dc
            if err < 0:
                col += col_step
                err += dr
            row += row_step

    cells.append((row2, col2))
    return cells
```

## A.5 Graph Generation

### A.5.1 Full Implementation

```python
from dataclasses import dataclass

@dataclass
class Cell:
    id: str
    neighbors: dict[str, str]  # direction -> neighbor_id
    row: int
    col: int
    position_hint: tuple[float, float]
    contents: object = None

class SquareTiling:
    """Square tiling implementation."""

    name = "square"
    directions = ["north", "east", "south", "west"]

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells: dict[str, Cell] = {}

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

    def get_neighbor(self, cell_id: str, direction: str) -> str | None:
        """Get neighbor in given direction."""
        return self.cells[cell_id].neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Graph distance (hops) between cells."""
        row_a, col_a = cell_id_to_row_col(cell_a)
        row_b, col_b = cell_id_to_row_col(cell_b)
        return manhattan_distance(row_a, col_a, row_b, col_b)
```

## A.6 Zone Computation

Zones are defined by center + radius in hops:

```python
def compute_zone_cells(
    center_row: int, center_col: int,
    radius: int,
    width: int, height: int
) -> set[str]:
    """
    Compute all cells within radius hops of center.
    For square grids, this creates a diamond/rhombus shape.
    """
    cells = set()

    for row in range(height):
        for col in range(width):
            dist = manhattan_distance(center_row, center_col, row, col)
            if dist <= radius:
                cells.add(row_col_to_cell_id(row, col))

    return cells
```

Zone shape for different radii:

```
Radius 1:         Radius 2:           Radius 3:
    X                 X                   X
   XXX               XXX                 XXX
    X               XXXXX               XXXXX
                     XXX               XXXXXXX
                      X                 XXXXX
                                         XXX
                                          X
```

## A.7 Rendering

### A.7.1 Cell Vertices

```python
def get_cell_vertices(
    row: int, col: int,
    cell_size: float,
    offset_x: float = 0,
    offset_y: float = 0
) -> list[tuple[float, float]]:
    """
    Get pixel coordinates of cell corners (clockwise from top-left).

    Args:
        row, col: Cell coordinates
        cell_size: Size of cell in pixels
        offset_x, offset_y: Render offset

    Returns:
        List of 4 (x, y) tuples for corners
    """
    x = offset_x + col * cell_size
    y = offset_y + row * cell_size

    return [
        (x, y),                      # Top-left
        (x + cell_size, y),          # Top-right
        (x + cell_size, y + cell_size),  # Bottom-right
        (x, y + cell_size)           # Bottom-left
    ]

def get_cell_center(
    row: int, col: int,
    cell_size: float,
    offset_x: float = 0,
    offset_y: float = 0
) -> tuple[float, float]:
    """Get pixel coordinates of cell center."""
    x = offset_x + (col + 0.5) * cell_size
    y = offset_y + (row + 0.5) * cell_size
    return x, y
```

### A.7.2 Direction Angles

For rendering agent facing direction:

```python
import math

DIRECTION_ANGLES = {
    "north": -math.pi / 2,   # -90° (pointing up)
    "east":  0,              # 0° (pointing right)
    "south": math.pi / 2,    # 90° (pointing down)
    "west":  math.pi         # 180° (pointing left)
}

def facing_to_angle(facing: int) -> float:
    """Convert facing index to angle in radians."""
    return DIRECTION_ANGLES[DIRECTIONS[facing]]
```

## A.8 Test Cases

### A.8.1 Graph Generation

```python
def test_square_graph_generation():
    """Test basic graph generation."""
    tiling = SquareTiling()
    cells = tiling.generate_graph(3, 3)

    # Should have 9 cells
    assert len(cells) == 9

    # Center cell should have 4 neighbors
    center = cells["sq_1_1"]
    assert len(center.neighbors) == 4
    assert center.neighbors["north"] == "sq_0_1"
    assert center.neighbors["east"] == "sq_1_2"
    assert center.neighbors["south"] == "sq_2_1"
    assert center.neighbors["west"] == "sq_1_0"

    # Corner cell should have 2 neighbors
    corner = cells["sq_0_0"]
    assert len(corner.neighbors) == 2
    assert "north" not in corner.neighbors
    assert "west" not in corner.neighbors
```

### A.8.2 Coordinate Conversion

```python
def test_coordinate_round_trip():
    """Test canonical <-> cell coordinate conversion."""
    tiling = SquareTiling()
    tiling.generate_graph(10, 10)

    # Test round-trip for center of grid
    cell_id = tiling.canonical_to_cell(0.55, 0.45)
    x, y = tiling.cell_to_canonical(cell_id)

    # Should be near original (within half cell)
    assert abs(x - 0.55) < 0.1
    assert abs(y - 0.45) < 0.1
```

### A.8.3 Movement Sequence

```python
def test_movement_sequence():
    """Test a sequence of movements."""
    tiling = SquareTiling()
    tiling.generate_graph(5, 5)

    # Start at center
    current = "sq_2_2"

    # Move east, then south, then west
    moves = ["east", "south", "west"]
    expected = ["sq_2_3", "sq_3_3", "sq_3_2"]

    for move, expected_cell in zip(moves, expected):
        current = tiling.get_neighbor(current, move)
        assert current == expected_cell
```

## A.9 Contamination Notes

Square grids are the most contaminated tiling in AI training data:

- **MiniGrid**: OpenAI Gym's standard gridworld
- **NetHack**: ASCII dungeon crawler with grid navigation
- **Pokémon games**: Tile-based movement
- **Sokoban**: Classic push puzzle
- **Many RL benchmarks**: Default to square grids

**Mitigation**: Use square grid only as baseline; primary evaluation should use hex or exotic tilings.
