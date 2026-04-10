# Appendix B: Hexagonal Tiling

**Status:** Implementation-Ready
**Primary Reference:** [Red Blob Games: Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/)

## B.1 Overview

Hexagonal tilings have 6 neighbors per cell (vs 4 for squares), requiring models to reason about more complex connectivity. Key properties:

- **6 movement directions**: More options per step
- **Consistent distance metric**: All neighbors are equidistant
- **Less common in training data**: Strategy games (Civ, Settlers) but less saturated than square grids

## B.2 Coordinate Systems

### B.2.1 Axial Coordinates (Primary)

Axial coordinates use two axes (q, r) at 60° angles:

```
        _____
       /     \
 _____/  0,0  \_____
/     \       /     \
\ -1,0 \_____/  1,0  \
/     /       \      /
\_____\ 0,-1  /_____/
      /       \
      \_______/
        0,1
```

- **q axis**: Points east-northeast
- **r axis**: Points south
- **Implicit s axis**: s = -q - r (for cube coordinate conversion)

```python
@dataclass
class AxialCoord:
    q: int
    r: int

    def __add__(self, other: "AxialCoord") -> "AxialCoord":
        return AxialCoord(self.q + other.q, self.r + other.r)

    def __sub__(self, other: "AxialCoord") -> "AxialCoord":
        return AxialCoord(self.q - other.q, self.r - other.r)

    def __mul__(self, scalar: int) -> "AxialCoord":
        return AxialCoord(self.q * scalar, self.r * scalar)

    @property
    def s(self) -> int:
        """Implicit third coordinate."""
        return -self.q - self.r
```

### B.2.2 Cube Coordinates (For Complex Math)

Cube coordinates use three axes (q, r, s) with constraint q + r + s = 0:

```python
@dataclass
class CubeCoord:
    q: int
    r: int
    s: int

    def __post_init__(self):
        assert self.q + self.r + self.s == 0, "Invalid cube coord: q+r+s must equal 0"

    def __add__(self, other: "CubeCoord") -> "CubeCoord":
        return CubeCoord(self.q + other.q, self.r + other.r, self.s + other.s)

    def __sub__(self, other: "CubeCoord") -> "CubeCoord":
        return CubeCoord(self.q - other.q, self.r - other.r, self.s - other.s)

    def to_axial(self) -> AxialCoord:
        return AxialCoord(self.q, self.r)

    @staticmethod
    def from_axial(axial: AxialCoord) -> "CubeCoord":
        return CubeCoord(axial.q, axial.r, -axial.q - axial.r)
```

### B.2.3 Offset Coordinates (For Storage/Rendering)

Offset coordinates work like row/col but with offset rows. We use **odd-r** (odd rows shifted right):

```
Row 0:  [0,0] [1,0] [2,0] [3,0]
Row 1:    [0,1] [1,1] [2,1] [3,1]  <- shifted right
Row 2:  [0,2] [1,2] [2,2] [3,2]
Row 3:    [0,3] [1,3] [2,3] [3,3]  <- shifted right
```

```python
@dataclass
class OffsetCoord:
    col: int
    row: int

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
```

### B.2.4 Canonical Coordinate Conversion

```python
import math

def canonical_to_axial(
    x: float, y: float,
    width: int, height: int
) -> AxialCoord:
    """
    Convert normalized [0,1] coordinates to axial hex coordinates.

    Uses pointy-top hexagons with horizontal rows.
    """
    # Scale to grid dimensions
    # Hex width = sqrt(3) * size, height = 2 * size
    # For a grid of width W hexes, total width ≈ W * sqrt(3) * size
    size = 1.0 / (height * 1.5 + 0.5)  # Approximate hex size

    # Convert to pixel-like coordinates
    px = x / size
    py = y / size

    # Convert pixel to axial (fractional)
    q_frac = (math.sqrt(3)/3 * px - 1/3 * py)
    r_frac = (2/3 * py)

    # Round to nearest hex
    return axial_round(q_frac, r_frac)

def axial_to_canonical(
    axial: AxialCoord,
    width: int, height: int
) -> tuple[float, float]:
    """
    Convert axial coordinates to normalized [0,1] (hex center).
    """
    size = 1.0 / (height * 1.5 + 0.5)

    # Axial to pixel
    px = size * (math.sqrt(3) * axial.q + math.sqrt(3)/2 * axial.r)
    py = size * (3/2 * axial.r)

    return px, py

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
```

## B.3 Directions and Neighbors

### B.3.1 Direction Labels

Hexagons have 6 directions. For **pointy-top** orientation:

```
           N
          ___
     NW /     \ NE
       /       \
       \       /
     SW \_____/ SE
           S
```

```python
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

# Cube coordinate vectors (same directions)
DIR_VECTORS_CUBE = {
    "north":     CubeCoord(0, -1, 1),
    "northeast": CubeCoord(1, -1, 0),
    "southeast": CubeCoord(1, 0, -1),
    "south":     CubeCoord(0, 1, -1),
    "southwest": CubeCoord(-1, 1, 0),
    "northwest": CubeCoord(-1, 0, 1)
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
```

### B.3.2 Neighbor Computation

```python
def get_neighbor_axial(coord: AxialCoord, direction: str) -> AxialCoord:
    """Get neighbor in given direction using axial coordinates."""
    return coord + DIR_VECTORS_AXIAL[direction]

def get_all_neighbors_axial(coord: AxialCoord) -> dict[str, AxialCoord]:
    """Get all 6 neighbors."""
    return {
        direction: coord + delta
        for direction, delta in DIR_VECTORS_AXIAL.items()
    }
```

### B.3.3 Turn Operations

```python
def turn_left(facing: int) -> int:
    """Rotate facing counter-clockwise (60° left)."""
    return (facing - 1) % 6

def turn_right(facing: int) -> int:
    """Rotate facing clockwise (60° right)."""
    return (facing + 1) % 6

def get_facing_direction(facing: int) -> str:
    """Get direction label for facing index."""
    return DIRECTIONS[facing]
```

## B.4 Distance and Pathfinding

### B.4.1 Hex Distance

In cube coordinates, hex distance is elegant:

```python
def cube_distance(a: CubeCoord, b: CubeCoord) -> int:
    """
    Distance between two hexes in cube coordinates.
    Equivalent to Manhattan distance in cube space / 2.
    """
    return max(abs(a.q - b.q), abs(a.r - b.r), abs(a.s - b.s))

def axial_distance(a: AxialCoord, b: AxialCoord) -> int:
    """Distance in axial coordinates (derived from cube)."""
    return (
        abs(a.q - b.q) +
        abs(a.q + a.r - b.q - b.r) +
        abs(a.r - b.r)
    ) // 2
```

### B.4.2 Line Drawing

```python
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t

def cube_lerp(a: CubeCoord, b: CubeCoord, t: float) -> tuple[float, float, float]:
    """Linearly interpolate between two cube coordinates."""
    return (
        lerp(a.q, b.q, t),
        lerp(a.r, b.r, t),
        lerp(a.s, b.s, t)
    )

def cube_round(q: float, r: float, s: float) -> CubeCoord:
    """Round fractional cube coordinates to nearest hex."""
    rq = round(q)
    rr = round(r)
    rs = round(s)

    q_diff = abs(rq - q)
    r_diff = abs(rr - r)
    s_diff = abs(rs - s)

    if q_diff > r_diff and q_diff > s_diff:
        rq = -rr - rs
    elif r_diff > s_diff:
        rr = -rq - rs
    else:
        rs = -rq - rr

    return CubeCoord(rq, rr, rs)

def hex_line(a: CubeCoord, b: CubeCoord) -> list[CubeCoord]:
    """
    Draw a line between two hexes.
    Returns list of hexes the line passes through.
    """
    n = cube_distance(a, b)
    if n == 0:
        return [a]

    results = []
    for i in range(n + 1):
        t = i / n
        q, r, s = cube_lerp(a, b, t)
        results.append(cube_round(q, r, s))

    return results
```

## B.5 Graph Generation

### B.5.1 Full Implementation

```python
def axial_to_cell_id(coord: AxialCoord) -> str:
    """Convert axial coordinates to cell ID."""
    return f"hex_{coord.q}_{coord.r}"

def cell_id_to_axial(cell_id: str) -> AxialCoord:
    """Parse cell ID to axial coordinates."""
    _, q, r = cell_id.split("_")
    return AxialCoord(int(q), int(r))

class HexTiling:
    """Hexagonal tiling implementation with pointy-top orientation."""

    name = "hex"
    directions = ["north", "northeast", "southeast", "south", "southwest", "northwest"]

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells: dict[str, Cell] = {}
        self._bounds: set[AxialCoord] = set()

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
        # Hex dimensions: width = sqrt(3) * size, height = 2 * size
        # For pointy-top, horizontal spacing is sqrt(3) * size
        # Vertical spacing is 1.5 * size (3/4 overlap)

        size = 1.0 / max(self.width, self.height * 0.866)

        x = size * math.sqrt(3) * (axial.q + axial.r / 2.0)
        y = size * 1.5 * axial.r

        # Normalize to [0,1] based on grid bounds
        # Add offset to center the grid
        x = (x + 0.5) / 1.2
        y = (y + 0.5) / 1.2

        return x, y

    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized coordinates to nearest cell ID."""
        # Reverse the normalization
        size = 1.0 / max(self.width, self.height * 0.866)

        px = (x * 1.2 - 0.5) / size
        py = (y * 1.2 - 0.5) / size

        # Pixel to fractional axial
        q_frac = (math.sqrt(3)/3 * px - 1/3 * py) / math.sqrt(3)
        r_frac = py / 1.5

        axial = axial_round(q_frac, r_frac)

        # Clamp to valid bounds
        if axial not in self._bounds:
            # Find nearest valid cell
            axial = min(
                self._bounds,
                key=lambda a: axial_distance(a, axial)
            )

        return axial_to_cell_id(axial)

    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized coordinates (hex center)."""
        axial = cell_id_to_axial(cell_id)
        return self._axial_to_normalized(axial)

    def get_neighbor(self, cell_id: str, direction: str) -> str | None:
        """Get neighbor in given direction."""
        return self.cells[cell_id].neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Graph distance (hops) between cells."""
        axial_a = cell_id_to_axial(cell_a)
        axial_b = cell_id_to_axial(cell_b)
        return axial_distance(axial_a, axial_b)
```

## B.6 Zone Computation

Zones form **hexagonal** shapes on hex grids:

```python
def compute_zone_cells_hex(
    center: AxialCoord,
    radius: int,
    valid_cells: set[AxialCoord]
) -> set[str]:
    """
    Compute all cells within radius hops of center.
    For hex grids, this creates a hexagonal shape.
    """
    cells = set()

    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coord = AxialCoord(center.q + q, center.r + r)
            if coord in valid_cells:
                cells.add(axial_to_cell_id(coord))

    return cells
```

Zone shapes:

```
Radius 1:       Radius 2:
   _             _____
 _/ \_         _/     \_
/     \       /         \
\_   _/       \    _    /
  \_/          \__/ \__/
               /         \
               \_________/
```

## B.7 Rendering

### B.7.1 Hex Vertices (Pointy-Top)

```python
def get_hex_vertices(
    center_x: float,
    center_y: float,
    size: float
) -> list[tuple[float, float]]:
    """
    Get the 6 vertices of a pointy-top hexagon.

    Args:
        center_x, center_y: Center of hexagon
        size: Distance from center to vertex

    Returns:
        List of 6 (x, y) tuples, starting from top vertex, clockwise
    """
    vertices = []
    for i in range(6):
        # Start from top (90°), go clockwise
        angle = math.pi / 2 - i * math.pi / 3
        vx = center_x + size * math.cos(angle)
        vy = center_y - size * math.sin(angle)  # Y inverted for screen coords
        vertices.append((vx, vy))
    return vertices
```

### B.7.2 Direction Angles

```python
# Angles for each direction (pointing outward from hex center)
# Measured from positive x-axis, counter-clockwise
DIRECTION_ANGLES = {
    "north":     math.pi / 2,       # 90° (up)
    "northeast": math.pi / 6,       # 30°
    "southeast": -math.pi / 6,      # -30° (330°)
    "south":     -math.pi / 2,      # -90° (270°)
    "southwest": -5 * math.pi / 6,  # -150° (210°)
    "northwest": 5 * math.pi / 6    # 150°
}

def facing_to_angle(facing: int) -> float:
    """Convert facing index to angle in radians."""
    return DIRECTION_ANGLES[DIRECTIONS[facing]]
```

## B.8 Test Cases

### B.8.1 Coordinate Conversions

```python
def test_axial_cube_roundtrip():
    """Test axial <-> cube conversion."""
    for q in range(-5, 6):
        for r in range(-5, 6):
            axial = AxialCoord(q, r)
            cube = CubeCoord.from_axial(axial)

            # Verify constraint
            assert cube.q + cube.r + cube.s == 0

            # Verify roundtrip
            back = cube.to_axial()
            assert back.q == axial.q
            assert back.r == axial.r

def test_offset_axial_roundtrip():
    """Test offset <-> axial conversion."""
    for row in range(10):
        for col in range(10):
            offset = OffsetCoord(col, row)
            axial = offset_to_axial(offset)
            back = axial_to_offset(axial)

            assert back.col == offset.col
            assert back.row == offset.row
```

### B.8.2 Distance Computation

```python
def test_hex_distance():
    """Test hex distance calculation."""
    origin = AxialCoord(0, 0)

    # Adjacent cells are distance 1
    for direction in DIRECTIONS:
        neighbor = get_neighbor_axial(origin, direction)
        assert axial_distance(origin, neighbor) == 1

    # Two steps away
    two_north = AxialCoord(0, -2)
    assert axial_distance(origin, two_north) == 2

    # Diagonal movement
    northeast_2 = AxialCoord(2, -2)
    assert axial_distance(origin, northeast_2) == 2
```

### B.8.3 Neighbor Count

```python
def test_hex_neighbors():
    """Test that interior hex has 6 neighbors."""
    tiling = HexTiling()
    tiling.generate_graph(10, 10)

    # Find an interior cell
    interior_offset = OffsetCoord(5, 5)
    interior_axial = offset_to_axial(interior_offset)
    interior_id = axial_to_cell_id(interior_axial)

    cell = tiling.cells[interior_id]
    assert len(cell.neighbors) == 6

    # All 6 directions should have neighbors
    for direction in DIRECTIONS:
        assert direction in cell.neighbors
```

### B.8.4 Movement Sequence

```python
def test_hex_movement():
    """Test movement in hex grid."""
    tiling = HexTiling()
    tiling.generate_graph(10, 10)

    # Start at center-ish
    start_offset = OffsetCoord(5, 5)
    start_axial = offset_to_axial(start_offset)
    current = axial_to_cell_id(start_axial)

    # Move in a hexagon pattern (should return to start)
    moves = ["north", "northeast", "southeast", "south", "southwest", "northwest"]

    for move in moves:
        current = tiling.get_neighbor(current, move)
        assert current is not None

    # After 6 moves in a circle, we're back at start
    # (Only true for unit circle, not for this sequence)
```

## B.9 Contamination Notes

Hex grids are present in:

- **Strategy games**: Civilization series, Battle for Wesnoth
- **Board game adaptations**: Settlers of Catan, various wargames
- **Some puzzle games**: Hexcells

**Risk level**: Moderate - less saturated than square grids but not rare.

**Mitigation strategies**:
1. Use visual styles different from common games
2. Combine with unusual object types/colors
3. Progress to exotic tilings (3-4-6-4) for lower contamination
