# Appendix D: Exotic Tilings

**Status:** Algorithm References (Implementation Deferred)

## D.1 Overview

Exotic tilings extend beyond the three regular tilings (square, hexagon, triangle) to include:

1. **Archimedean (Semi-regular) Tilings**: Multiple polygon types meeting at each vertex
2. **Aperiodic Tilings**: Non-repeating patterns (e.g., Penrose tilings)
3. **Custom Tilings**: Application-specific or procedurally generated

These tilings offer the **lowest contamination risk** because they are rarely if ever seen in AI training data.

## D.2 Archimedean Tilings

### D.2.1 Background

Archimedean tilings use two or more regular polygon types arranged so that the same sequence of polygons appears at every vertex. There are exactly **8 distinct Archimedean tilings** of the Euclidean plane.

The naming convention lists the polygon types (by edge count) around each vertex in order:

| Name | Vertex Configuration | Polygons Used |
|------|---------------------|---------------|
| 3.3.3.3.6 | 4 triangles + 1 hexagon | Triangle, Hexagon |
| 3.3.3.4.4 | 3 triangles + 2 squares | Triangle, Square |
| 3.3.4.3.4 | Alternating triangles/squares | Triangle, Square |
| **3.4.6.4** | Triangle, square, hexagon, square | Triangle, Square, Hexagon |
| 3.6.3.6 | Alternating triangles/hexagons | Triangle, Hexagon |
| 3.12.12 | Triangle + 2 dodecagons | Triangle, Dodecagon |
| 4.6.12 | Square, hexagon, dodecagon | Square, Hexagon, Dodecagon |
| 4.8.8 | Square + 2 octagons | Square, Octagon |

**Reference:** [Euclidean tilings by convex regular polygons (Wikipedia)](https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons)

### D.2.2 The 3.4.6.4 Tiling (Primary Target)

The **3.4.6.4** (rhombitrihexagonal) tiling is an excellent candidate for MultiGrid because:

- Uses three different polygon types (visual complexity)
- Variable neighbor counts (3, 4, or 6 depending on tile type)
- Extremely rare in training data
- Still tractable for implementation

```
Vertex configuration: at each vertex, going clockwise:
- 1 triangle (3 edges)
- 1 square (4 edges)
- 1 hexagon (6 edges)
- 1 square (4 edges)

Vertex angle sum: 60° + 90° + 120° + 90° = 360° ✓
```

**Visual representation:**

```
        ___
       /   \
   ___/     \___
  |   |     |   |
  |___|     |___|
     /       \
    /    _    \
   /    / \    \
  |    |   |    |
  |____|   |____|
       \___/
```

### D.2.3 Generation Algorithm for 3.4.6.4

**Approach 1: Template-Based Construction**

1. Define a fundamental domain (smallest repeating unit)
2. Tile the plane by translating the fundamental domain
3. Build adjacency graph from the resulting structure

```python
# Pseudocode for 3.4.6.4 generation

class Tile346:
    """A single tile in the 3.4.6.4 tiling."""
    def __init__(self, tile_type: str, position: tuple):
        self.tile_type = tile_type  # "triangle", "square", or "hexagon"
        self.position = position     # (x, y) center
        self.vertices = []           # Computed from type and position
        self.neighbors = []          # Adjacent tiles

def generate_346_fundamental_domain(origin: tuple) -> list[Tile346]:
    """
    Generate one fundamental domain of 3.4.6.4 tiling.

    The fundamental domain contains:
    - 2 triangles
    - 3 squares
    - 1 hexagon

    Returns list of tiles with local adjacency.
    """
    # Position calculations based on:
    # - Triangle side length = 1 (unit)
    # - Square side length = 1
    # - Hexagon side length = 1
    pass  # Implementation deferred

def tile_plane_346(width: int, height: int) -> dict[str, Tile346]:
    """
    Tile a rectangular region with 3.4.6.4 pattern.

    Args:
        width, height: Region size in fundamental domain units

    Returns:
        Dictionary of tile_id -> Tile346
    """
    tiles = {}

    for row in range(height):
        for col in range(width):
            # Generate fundamental domain at this position
            origin = compute_domain_origin(row, col)
            domain_tiles = generate_346_fundamental_domain(origin)

            # Add to collection with unique IDs
            for i, tile in enumerate(domain_tiles):
                tile_id = f"t346_{row}_{col}_{i}"
                tiles[tile_id] = tile

    # Connect neighbors across domain boundaries
    connect_adjacent_tiles(tiles)

    return tiles
```

**Approach 2: Dual Graph Construction**

The 3.4.6.4 tiling is the **rectification** of the trihexagonal tiling (3.6.3.6). We can:

1. Generate the dual trihexagonal tiling
2. Place vertices at edge midpoints
3. Connect to form the 3.4.6.4 structure

**Reference:** Grünbaum, B., & Shephard, G. C. (1987). *Tilings and Patterns*. W.H. Freeman. (Chapter 2: Tilings by Regular Polygons)

### D.2.4 Neighbor Relationships in 3.4.6.4

| Tile Type | Edges | Neighbors |
|-----------|-------|-----------|
| Triangle | 3 | 1 hexagon, 2 squares |
| Square | 4 | 2 triangles, 2 hexagons OR 1 triangle, 2 hexagons, 1 square |
| Hexagon | 6 | 6 tiles (alternating triangles and squares) |

Direction labeling must account for variable neighbor counts:

```python
# Dynamic direction system for 3.4.6.4
def get_directions_346(tile_type: str) -> list[str]:
    """Get direction labels for a tile type."""
    if tile_type == "triangle":
        return ["edge_0", "edge_1", "edge_2"]
    elif tile_type == "square":
        return ["edge_0", "edge_1", "edge_2", "edge_3"]
    elif tile_type == "hexagon":
        return ["edge_0", "edge_1", "edge_2", "edge_3", "edge_4", "edge_5"]
```

### D.2.5 Resources for Archimedean Tiling Implementation

1. **GomJau-Hogg's Notation**: A systematic notation for generating uniform tilings
   - [Antwerp v3.0](https://www.gomjau-hogg.com/antwerp) - Web application for tiling generation

2. **Academic Papers**:
   - Sahr, K. (2011). "Hexagonal Discrete Global Grid Systems for Geospatial Computing"
   - [PDF](https://www.discreteglobalgrids.org/wp-content/uploads/2016/01/sahrMMT11us.pdf)

3. **Code Libraries**:
   - `tessellation` Python package (limited, may need extension)
   - PostGIS for geospatial tiling (overkill but reference)

## D.3 Aperiodic Tilings

### D.3.1 Penrose Tilings

Penrose tilings are **aperiodic** - they never repeat, yet have local order. Two main variants:

**P2 (Kite and Dart):**
- Uses two quadrilateral shapes
- Golden ratio proportions
- Local 5-fold symmetry

**P3 (Rhombus):**
- Uses two rhombus shapes (thin and thick)
- Angles based on 36° and 72°
- Easier to implement than P2

### D.3.2 Penrose P3 Generation

**Approach: Substitution/Inflation**

1. Start with a single rhombus
2. Apply substitution rules to split into smaller rhombi
3. Repeat until desired resolution
4. Build adjacency graph from resulting tiles

```python
# Pseudocode for Penrose P3 generation

import math

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618

class PenroseRhombus:
    """A rhombus tile in Penrose P3 tiling."""
    def __init__(self, vertices: list[tuple], is_thick: bool):
        self.vertices = vertices  # 4 vertices in order
        self.is_thick = is_thick  # Thick (72°) or thin (36°)

def subdivide_thick_rhombus(r: PenroseRhombus) -> list[PenroseRhombus]:
    """
    Subdivide a thick rhombus into smaller tiles.

    A thick rhombus splits into:
    - 2 thick rhombi
    - 1 thin rhombus
    """
    # Compute subdivision vertices using golden ratio
    pass  # Implementation deferred

def subdivide_thin_rhombus(r: PenroseRhombus) -> list[PenroseRhombus]:
    """
    Subdivide a thin rhombus into smaller tiles.

    A thin rhombus splits into:
    - 1 thick rhombus
    - 1 thin rhombus
    """
    pass  # Implementation deferred

def generate_penrose(iterations: int, bounds: tuple) -> dict[str, PenroseRhombus]:
    """
    Generate Penrose tiling via substitution.

    Args:
        iterations: Number of subdivision iterations
        bounds: (width, height) of region to fill

    Returns:
        Dictionary of tile_id -> PenroseRhombus
    """
    # Start with initial configuration (e.g., 5-fold symmetric star)
    tiles = create_initial_star()

    for _ in range(iterations):
        new_tiles = []
        for tile in tiles:
            if tile.is_thick:
                new_tiles.extend(subdivide_thick_rhombus(tile))
            else:
                new_tiles.extend(subdivide_thin_rhombus(tile))
        tiles = new_tiles

    # Clip to bounds and assign IDs
    return {f"pen_{i}": t for i, t in enumerate(tiles) if in_bounds(t, bounds)}
```

### D.3.3 Neighbor Relationships in Penrose

Both thick and thin rhombi have **4 neighbors** (one per edge), but:
- Matching rules constrain which tiles can be adjacent
- Not all edge pairings are valid

```python
# Penrose matching rules
# Edges are labeled by type to ensure valid adjacency
EDGE_TYPES = {
    "thick": ["A", "B", "A", "B"],  # Alternating edge types
    "thin": ["C", "D", "C", "D"]
}

def can_match(edge1_type: str, edge2_type: str) -> bool:
    """Check if two edges can be adjacent."""
    # Matching rules: A-A, B-B, C-C, D-D only
    return edge1_type == edge2_type
```

### D.3.4 Resources for Penrose Implementation

1. **Canonical Reference**:
   - de Bruijn, N.G. (1981). "Algebraic theory of Penrose's non-periodic tilings"
   - [PDF available through academic sources]

2. **Implementation Guides**:
   - [Preshing on Programming: Penrose Tiling Explained](https://preshing.com/20110831/penrose-tiling-explained/)
   - [rosettacode.org: Penrose tiling](https://rosettacode.org/wiki/Penrose_tiling)

3. **Python Libraries**:
   - `penrose` PyPI package (basic implementation)
   - Custom implementation recommended for MultiGrid integration

## D.4 Implementation Strategy for Exotic Tilings

### D.4.1 Adjacency Graph Adapter

All exotic tilings should produce the same `TilingGraph` structure as regular tilings:

```python
class ExoticTiling(Tiling):
    """Base class for exotic tilings."""

    def generate_graph(self, width: int, height: int, seed: int) -> TilingGraph:
        """
        Generate exotic tiling as adjacency graph.

        The exotic-specific generation (substitution, template, etc.)
        happens internally. The output is a standard TilingGraph.
        """
        # 1. Generate tiles using exotic-specific algorithm
        tiles = self._generate_tiles(width, height, seed)

        # 2. Build adjacency from tile geometry
        graph = self._build_adjacency(tiles)

        # 3. Compute canonical positions for rendering
        self._compute_positions(graph)

        return graph

    @abstractmethod
    def _generate_tiles(self, width: int, height: int, seed: int) -> list:
        """Exotic-specific tile generation."""
        pass

    def _build_adjacency(self, tiles: list) -> TilingGraph:
        """
        Build adjacency graph from tile geometry.

        Uses computational geometry to detect shared edges.
        """
        # For each pair of tiles, check if they share an edge
        # This is O(n²) but can be optimized with spatial indexing
        pass
```

### D.4.2 Direction Handling

Exotic tilings have **variable neighbor counts**. The action space must accommodate this:

```python
# Option 1: Dynamic direction labels
# Cons: Action space varies per cell

# Option 2: Indexed directions (recommended)
# Use "neighbor_0", "neighbor_1", etc.
# Pros: Fixed action space, consistent interface

class ExoticTiling(Tiling):
    @property
    def directions(self) -> list[str]:
        # Return maximum possible directions
        return [f"neighbor_{i}" for i in range(self.max_neighbors)]

    @property
    @abstractmethod
    def max_neighbors(self) -> int:
        """Maximum neighbors any tile can have."""
        pass
```

### D.4.3 Testing Exotic Tilings

```python
def test_exotic_tiling_invariants(tiling: ExoticTiling):
    """Test that exotic tiling satisfies basic invariants."""
    graph = tiling.generate_graph(10, 10, seed=42)

    # All cells should have at least 1 neighbor
    for cell_id, cell in graph.cells.items():
        assert len(cell.neighbors) >= 1, f"Cell {cell_id} has no neighbors"

    # Adjacency should be symmetric
    for cell_id, cell in graph.cells.items():
        for direction, neighbor_id in cell.neighbors.items():
            neighbor = graph.cells[neighbor_id]
            # Neighbor should have a direction pointing back
            reverse_found = any(
                nid == cell_id
                for nid in neighbor.neighbors.values()
            )
            assert reverse_found, f"Asymmetric adjacency: {cell_id} -> {neighbor_id}"

    # Canonical positions should be unique
    positions = [cell.position_hint for cell in graph.cells.values()]
    # Allow small tolerance for floating point
    unique_count = len(set((round(x, 6), round(y, 6)) for x, y in positions))
    assert unique_count == len(positions), "Duplicate positions detected"
```

## D.5 Contamination Analysis

| Tiling Type | Training Data Presence | Risk Level |
|-------------|----------------------|------------|
| Square | Ubiquitous | Very High |
| Hexagon | Common (strategy games) | Moderate |
| Triangle | Rare | Low |
| 3.4.6.4 | Extremely rare | Very Low |
| Penrose | Mathematical contexts only | Minimal |
| Custom | None | None |

**Recommendation**: Progress through tilings in order of contamination risk:
1. Square (baseline only)
2. Hexagon (primary evaluation)
3. Triangle (alternative evaluation)
4. 3.4.6.4 (advanced evaluation)
5. Penrose (research frontier)

## D.6 Future Work

### D.6.1 Procedural Tiling Generation

Beyond fixed tiling types, MultiGrid could support:

- **Parameterized tilings**: Continuous deformation of regular tilings
- **Stochastic tilings**: Random tile placement with constraints
- **Learned tilings**: Optimize tiling for maximum model confusion

### D.6.2 3D Extension

The adjacency graph architecture naturally extends to 3D:

- **Polyhedra**: Cubes, tetrahedra, etc.
- **Space-filling**: Truncated octahedra, rhombic dodecahedra
- **Quasi-crystalline**: 3D Penrose analogs

This aligns with the Domain 2 (physics) integration path.

## D.7 References

### Archimedean Tilings
- Grünbaum, B., & Shephard, G. C. (1987). *Tilings and Patterns*. W.H. Freeman.
- [Wikipedia: Euclidean tilings by convex regular polygons](https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons)
- [Wolfram MathWorld: Semiregular Tessellation](https://mathworld.wolfram.com/SemiregularTessellation.html)

### Penrose Tilings
- Penrose, R. (1974). "The role of aesthetics in pure and applied mathematical research". *Bull. Inst. Math. Appl.* 10: 266–271.
- de Bruijn, N.G. (1981). "Algebraic theory of Penrose's non-periodic tilings of the plane". *Kon. Nederl. Akad. Wetensch. Proc. Ser. A* 84: 39–66.
- [Preshing on Programming: Penrose Tiling Explained](https://preshing.com/20110831/penrose-tiling-explained/)

### General Tessellation Algorithms
- [NRICH: Semi-regular Tessellations](https://nrich.maths.org/semiregular)
- [GomJau-Hogg's Antwerp Notation](https://www.gomjau-hogg.com/antwerp)
