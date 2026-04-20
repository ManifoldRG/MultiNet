# Appendix C: Triangular Tiling

**Status:** Implementation-Ready

## C.1 Overview

Triangular tilings use equilateral triangles that meet 6 at each vertex. Key properties:

- **3 neighbors per cell**: Fewer movement options than square or hex
- **Alternating orientation**: Triangles alternate between pointing up (▲) and down (▽)
- **Rare in training data**: Much less common than square or hex grids
- **Unique movement patterns**: Requires different planning strategies

## C.2 Grid Structure

### C.2.1 Visual Layout

```
Row 0:   ▲   ▽   ▲   ▽   ▲   ▽
Row 1:   ▽   ▲   ▽   ▲   ▽   ▲
Row 2:   ▲   ▽   ▲   ▽   ▲   ▽
```

Each row contains alternating up-pointing (▲) and down-pointing (▽) triangles. Adjacent rows are offset so triangles interlock.

### C.2.2 Triangle Orientation

A triangle's orientation is determined by:
- **Up-pointing (▲)**: `(row + col) % 2 == 0`
- **Down-pointing (▽)**: `(row + col) % 2 == 1`

```python
from enum import Enum

class TriOrientation(Enum):
    UP = 0    # ▲
    DOWN = 1  # ▽

def get_orientation(row: int, col: int) -> TriOrientation:
    """Determine triangle orientation from position."""
    if (row + col) % 2 == 0:
        return TriOrientation.UP
    else:
        return TriOrientation.DOWN
```

## C.3 Coordinate System

### C.3.1 Primary Coordinates (Row, Column)

We use a row/column system where each cell is identified by (row, col):

```
Col:    0   1   2   3   4   5
      +---+---+---+---+---+---+
Row 0 | ▲ | ▽ | ▲ | ▽ | ▲ | ▽ |
      +---+---+---+---+---+---+
Row 1 | ▽ | ▲ | ▽ | ▲ | ▽ | ▲ |
      +---+---+---+---+---+---+
Row 2 | ▲ | ▽ | ▲ | ▽ | ▲ | ▽ |
      +---+---+---+---+---+---+
```

```python
@dataclass
class TriCoord:
    row: int
    col: int

    @property
    def orientation(self) -> TriOrientation:
        return get_orientation(self.row, self.col)

def tri_to_cell_id(coord: TriCoord) -> str:
    """Convert coordinates to cell ID."""
    return f"tri_{coord.row}_{coord.col}"

def cell_id_to_tri(cell_id: str) -> TriCoord:
    """Parse cell ID to coordinates."""
    _, row, col = cell_id.split("_")
    return TriCoord(int(row), int(col))
```

### C.3.2 Canonical Coordinate Conversion

```python
import math

def canonical_to_tri(
    x: float, y: float,
    width: int, height: int
) -> TriCoord:
    """
    Convert normalized [0,1] coordinates to triangle coordinates.

    Triangle layout:
    - Each triangle has width = 1 unit, height = sqrt(3)/2 units
    - Rows are packed vertically with height sqrt(3)/2
    - Columns are packed horizontally with width 0.5 (half triangle width)
    """
    # Scale based on grid dimensions
    tri_width = 1.0 / width
    tri_height = (math.sqrt(3) / 2) / height

    # Rough column estimate (2 triangles per unit width)
    col = int(x / (tri_width / 2))
    col = min(col, width - 1)

    # Rough row estimate
    row = int(y / tri_height)
    row = min(row, height - 1)

    # Refine based on exact position within cell
    # This requires checking which triangle the point falls into
    return TriCoord(row, col)

def tri_to_canonical(
    coord: TriCoord,
    width: int, height: int
) -> tuple[float, float]:
    """
    Convert triangle coordinates to normalized [0,1] (centroid).
    """
    tri_width = 1.0 / width
    tri_height = (math.sqrt(3) / 2) / height

    # Base position
    x = (coord.col + 0.5) * (tri_width / 2)
    y = (coord.row + 0.5) * tri_height

    # Adjust centroid based on orientation
    if coord.orientation == TriOrientation.UP:
        # Centroid is at 1/3 height from base
        y += tri_height / 6
    else:
        # Centroid is at 2/3 height from top
        y -= tri_height / 6

    return x, y
```

## C.4 Directions and Neighbors

### C.4.1 Direction Labels

Triangles have **3 edge-adjacent neighbors**. The direction labels depend on orientation:

**Up-pointing triangle (▲):**
```
        /\
       /  \
      / ▲  \
     /______\
    left  base  right

Neighbors: left (▽), right (▽), base (▽ below)
```

**Down-pointing triangle (▽):**
```
     ______
     \    /
      \ ▽/
       \/

Neighbors: left (▲), right (▲), apex (▲ above)
```

```python
# Directions vary by orientation
DIRECTIONS_UP = ["left", "right", "base"]      # ▲
DIRECTIONS_DOWN = ["left", "right", "apex"]    # ▽

def get_directions(orientation: TriOrientation) -> list[str]:
    """Get valid directions for given orientation."""
    if orientation == TriOrientation.UP:
        return DIRECTIONS_UP
    else:
        return DIRECTIONS_DOWN

# Unified direction set for interface consistency
ALL_DIRECTIONS = ["left", "right", "vertical"]  # "vertical" = base or apex
```

### C.4.2 Neighbor Computation

```python
# Neighbor offsets depend on orientation
# Format: (row_delta, col_delta)

NEIGHBOR_OFFSETS_UP = {
    "left":     (0, -1),   # Same row, previous column (▽)
    "right":    (0, 1),    # Same row, next column (▽)
    "base":     (1, 0),    # Next row, same column (▽)
}

NEIGHBOR_OFFSETS_DOWN = {
    "left":     (0, -1),   # Same row, previous column (▲)
    "right":    (0, 1),    # Same row, next column (▲)
    "apex":     (-1, 0),   # Previous row, same column (▲)
}

def get_neighbor_tri(
    coord: TriCoord,
    direction: str,
    width: int,
    height: int
) -> TriCoord | None:
    """
    Get neighbor in given direction.

    Args:
        coord: Current triangle coordinates
        direction: "left", "right", "base" (for ▲), or "apex" (for ▽)
        width, height: Grid dimensions

    Returns:
        Neighbor coordinates or None if out of bounds
    """
    if coord.orientation == TriOrientation.UP:
        if direction == "vertical":
            direction = "base"
        offsets = NEIGHBOR_OFFSETS_UP
    else:
        if direction == "vertical":
            direction = "apex"
        offsets = NEIGHBOR_OFFSETS_DOWN

    if direction not in offsets:
        return None

    dr, dc = offsets[direction]
    new_row = coord.row + dr
    new_col = coord.col + dc

    # Bounds check
    if 0 <= new_row < height and 0 <= new_col < width:
        return TriCoord(new_row, new_col)
    return None

def get_all_neighbors_tri(
    coord: TriCoord,
    width: int,
    height: int
) -> dict[str, TriCoord]:
    """Get all valid neighbors with direction labels."""
    if coord.orientation == TriOrientation.UP:
        directions = DIRECTIONS_UP
    else:
        directions = DIRECTIONS_DOWN

    neighbors = {}
    for direction in directions:
        neighbor = get_neighbor_tri(coord, direction, width, height)
        if neighbor is not None:
            neighbors[direction] = neighbor
    return neighbors
```

### C.4.3 Facing and Turning

With only 3 directions per orientation, facing works differently:

```python
def turn_left_tri(facing: int, orientation: TriOrientation) -> int:
    """
    Turn left in triangular grid.
    Cycles through the 3 directions counter-clockwise.
    """
    return (facing - 1) % 3

def turn_right_tri(facing: int, orientation: TriOrientation) -> int:
    """
    Turn right in triangular grid.
    Cycles through the 3 directions clockwise.
    """
    return (facing + 1) % 3

def get_facing_direction_tri(facing: int, orientation: TriOrientation) -> str:
    """Get direction label for facing index."""
    if orientation == TriOrientation.UP:
        return DIRECTIONS_UP[facing]
    else:
        return DIRECTIONS_DOWN[facing]
```

**Note:** When moving between triangles, the agent's facing may need to be remapped since the direction set changes with orientation.

## C.5 Distance Computation

### C.5.1 Graph Distance

Since triangles have irregular connectivity, distance is computed via graph traversal:

```python
from collections import deque

def triangle_distance(
    start: TriCoord,
    end: TriCoord,
    width: int,
    height: int
) -> int:
    """
    Compute minimum hops between two triangles using BFS.

    This is necessary because the irregular connectivity makes
    direct distance formulas unreliable.
    """
    if start.row == end.row and start.col == end.col:
        return 0

    visited = {(start.row, start.col)}
    queue = deque([(start, 0)])

    while queue:
        current, dist = queue.popleft()

        for neighbor in get_all_neighbors_tri(current, width, height).values():
            if neighbor.row == end.row and neighbor.col == end.col:
                return dist + 1

            key = (neighbor.row, neighbor.col)
            if key not in visited:
                visited.add(key)
                queue.append((neighbor, dist + 1))

    return -1  # Unreachable

def triangle_distance_approx(start: TriCoord, end: TriCoord) -> int:
    """
    Approximate distance using Manhattan-like formula.
    May overestimate due to orientation constraints.
    """
    row_diff = abs(start.row - end.row)
    col_diff = abs(start.col - end.col)

    # Rough approximation: need to traverse both row and column differences
    # but can sometimes move diagonally
    return row_diff + max(0, col_diff - row_diff)
```

## C.6 Graph Generation

```python
@dataclass
class Cell:
    id: str
    neighbors: dict[str, str]
    row: int
    col: int
    position_hint: tuple[float, float]
    orientation: TriOrientation
    contents: object = None

class TriangleTiling:
    """Triangular tiling implementation."""

    name = "triangle"
    directions = ["left", "right", "vertical"]

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells: dict[str, Cell] = {}

    def generate_graph(self, width: int, height: int, seed: int = 0) -> dict[str, Cell]:
        """
        Generate triangular grid as adjacency graph.

        Args:
            width: Number of columns (triangles per row)
            height: Number of rows
            seed: Random seed (unused for regular grids)

        Returns:
            Dictionary of cell_id -> Cell
        """
        self.width = width
        self.height = height
        self.cells = {}

        # Create all cells
        for row in range(height):
            for col in range(width):
                coord = TriCoord(row, col)
                cell_id = tri_to_cell_id(coord)
                pos = tri_to_canonical(coord, width, height)

                self.cells[cell_id] = Cell(
                    id=cell_id,
                    neighbors={},
                    row=row,
                    col=col,
                    position_hint=pos,
                    orientation=coord.orientation
                )

        # Connect neighbors
        for row in range(height):
            for col in range(width):
                coord = TriCoord(row, col)
                cell_id = tri_to_cell_id(coord)
                cell = self.cells[cell_id]

                neighbors = get_all_neighbors_tri(coord, width, height)
                for direction, neighbor_coord in neighbors.items():
                    neighbor_id = tri_to_cell_id(neighbor_coord)
                    # Normalize direction to unified set
                    unified_dir = direction if direction in ["left", "right"] else "vertical"
                    cell.neighbors[unified_dir] = neighbor_id

        return self.cells

    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized coordinates to cell ID."""
        coord = canonical_to_tri(x, y, self.width, self.height)
        return tri_to_cell_id(coord)

    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized coordinates (centroid)."""
        coord = cell_id_to_tri(cell_id)
        return tri_to_canonical(coord, self.width, self.height)

    def get_neighbor(self, cell_id: str, direction: str) -> str | None:
        """Get neighbor in given direction."""
        return self.cells[cell_id].neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Graph distance (hops) between cells."""
        coord_a = cell_id_to_tri(cell_a)
        coord_b = cell_id_to_tri(cell_b)
        return triangle_distance(coord_a, coord_b, self.width, self.height)
```

## C.7 Zone Computation

Zones on triangular grids have irregular shapes:

```python
def compute_zone_cells_tri(
    center: TriCoord,
    radius: int,
    width: int,
    height: int
) -> set[str]:
    """
    Compute all cells within radius hops of center.
    Uses BFS since distance formula is complex.
    """
    cells = set()
    visited = {(center.row, center.col)}
    queue = deque([(center, 0)])

    while queue:
        current, dist = queue.popleft()
        cells.add(tri_to_cell_id(current))

        if dist < radius:
            for neighbor in get_all_neighbors_tri(current, width, height).values():
                key = (neighbor.row, neighbor.col)
                if key not in visited:
                    visited.add(key)
                    queue.append((neighbor, dist + 1))

    return cells
```

Zone shapes are irregular due to alternating orientations.

## C.8 Rendering

### C.8.1 Triangle Vertices

```python
def get_triangle_vertices(
    row: int, col: int,
    cell_width: float,
    cell_height: float,
    offset_x: float = 0,
    offset_y: float = 0
) -> list[tuple[float, float]]:
    """
    Get pixel coordinates of triangle vertices.

    Args:
        row, col: Cell coordinates
        cell_width: Width of one triangle
        cell_height: Height of one triangle (sqrt(3)/2 * width for equilateral)
        offset_x, offset_y: Render offset

    Returns:
        List of 3 (x, y) tuples for vertices
    """
    orientation = get_orientation(row, col)

    # Base position
    base_x = offset_x + col * (cell_width / 2)
    base_y = offset_y + row * cell_height

    if orientation == TriOrientation.UP:
        # ▲ - apex at top
        return [
            (base_x + cell_width / 2, base_y),                    # Top (apex)
            (base_x, base_y + cell_height),                       # Bottom-left
            (base_x + cell_width, base_y + cell_height)           # Bottom-right
        ]
    else:
        # ▽ - apex at bottom
        return [
            (base_x, base_y),                                     # Top-left
            (base_x + cell_width, base_y),                        # Top-right
            (base_x + cell_width / 2, base_y + cell_height)       # Bottom (apex)
        ]

def get_triangle_centroid(
    row: int, col: int,
    cell_width: float,
    cell_height: float,
    offset_x: float = 0,
    offset_y: float = 0
) -> tuple[float, float]:
    """Get centroid of triangle."""
    vertices = get_triangle_vertices(row, col, cell_width, cell_height, offset_x, offset_y)
    cx = sum(v[0] for v in vertices) / 3
    cy = sum(v[1] for v in vertices) / 3
    return cx, cy
```

### C.8.2 Direction Angles

```python
# Angles for facing directions (pointing outward from centroid)
# For up-pointing triangles (▲)
DIRECTION_ANGLES_UP = {
    "left":  5 * math.pi / 6,   # 150° (upper-left edge)
    "right": math.pi / 6,       # 30° (upper-right edge)
    "base":  -math.pi / 2       # -90° / 270° (bottom edge)
}

# For down-pointing triangles (▽)
DIRECTION_ANGLES_DOWN = {
    "left":  -5 * math.pi / 6,  # -150° / 210° (lower-left edge)
    "right": -math.pi / 6,      # -30° / 330° (lower-right edge)
    "apex":  math.pi / 2        # 90° (top edge)
}

def facing_to_angle(facing: int, orientation: TriOrientation) -> float:
    """Convert facing index to angle in radians."""
    if orientation == TriOrientation.UP:
        directions = DIRECTIONS_UP
        angles = DIRECTION_ANGLES_UP
    else:
        directions = DIRECTIONS_DOWN
        angles = DIRECTION_ANGLES_DOWN

    return angles[directions[facing]]
```

## C.9 Test Cases

### C.9.1 Orientation Check

```python
def test_orientation_alternates():
    """Test that orientation alternates correctly."""
    for row in range(10):
        for col in range(10):
            orientation = get_orientation(row, col)
            expected = TriOrientation.UP if (row + col) % 2 == 0 else TriOrientation.DOWN
            assert orientation == expected
```

### C.9.2 Neighbor Count

```python
def test_triangle_neighbors():
    """Test that each triangle has exactly 3 neighbors (interior cells)."""
    tiling = TriangleTiling()
    tiling.generate_graph(20, 20)

    # Interior cell
    interior_id = tri_to_cell_id(TriCoord(10, 10))
    cell = tiling.cells[interior_id]

    # Should have 3 neighbors
    assert len(cell.neighbors) == 3
```

### C.9.3 Neighbor Orientation

```python
def test_neighbor_orientation_alternates():
    """Test that neighbors always have opposite orientation."""
    for row in range(1, 9):
        for col in range(1, 9):
            coord = TriCoord(row, col)
            my_orientation = coord.orientation

            for neighbor in get_all_neighbors_tri(coord, 10, 10).values():
                assert neighbor.orientation != my_orientation
```

### C.9.4 Movement Sequence

```python
def test_triangle_movement():
    """Test basic movement in triangular grid."""
    tiling = TriangleTiling()
    tiling.generate_graph(10, 10)

    # Start at (5, 5) - check orientation
    start = TriCoord(5, 5)
    current_id = tri_to_cell_id(start)

    # Move right, then vertical, then left should form a triangle
    moves = ["right", "vertical", "left"]

    for move in moves:
        next_id = tiling.get_neighbor(current_id, move)
        assert next_id is not None
        current_id = next_id
```

## C.10 Contamination Notes

Triangular grids are **rare** in AI training data:

- **Very few games**: Some abstract puzzles use triangles
- **Mathematical contexts**: Tessellation demonstrations
- **Minimal RL benchmarks**: Almost no standard environments use triangles

**Risk level**: Low - excellent for contamination resistance.

**Design consideration**: The 3-neighbor constraint creates unique planning challenges. Models must learn that:
- Not all cells are created equal (orientation matters)
- Movement patterns are asymmetric
- Direct paths may not exist between adjacent-looking cells
