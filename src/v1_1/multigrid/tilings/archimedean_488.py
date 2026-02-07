# tilings/archimedean_488.py

"""
Truncated Square (4-8-8) Archimedean Tiling

This tiling alternates regular octagons and squares. At every vertex,
one square and two octagons meet (vertex configuration 4.8.8).

Layout:
  - A checkerboard grid of spacing d = s * (1 + sqrt(2)) where s is edge length.
  - At even (row+col) positions: octagons (8 edges/neighbors).
  - At odd (row+col) positions: squares (4 edges/neighbors).

Adjacency is determined by shared-edge detection: two cells are neighbors
if they share exactly 2 vertices (within epsilon tolerance).
"""

import math
from collections import deque
from typing import Optional
from ..base import Tiling
from ..core import Cell


# Epsilon for floating-point vertex matching
_EPS = 1e-6


def _regular_polygon_vertices(center: tuple[float, float], n: int,
                               radius: float, rotation: float = 0.0
                               ) -> list[tuple[float, float]]:
    """
    Compute vertices of a regular n-gon centered at `center` with
    circumradius `radius` and an initial rotation angle (radians).
    """
    cx, cy = center
    verts = []
    for i in range(n):
        angle = rotation + 2 * math.pi * i / n
        vx = cx + radius * math.cos(angle)
        vy = cy + radius * math.sin(angle)
        verts.append((vx, vy))
    return verts


def _edge_length_to_circumradius(n: int, s: float) -> float:
    """Circumradius of a regular n-gon with edge length s."""
    return s / (2 * math.sin(math.pi / n))


def _vertices_match(v1: tuple[float, float], v2: tuple[float, float],
                    eps: float = _EPS) -> bool:
    """Check if two 2D points are within epsilon."""
    return abs(v1[0] - v2[0]) < eps and abs(v1[1] - v2[1]) < eps


def _shared_vertex_count(verts_a: list[tuple[float, float]],
                         verts_b: list[tuple[float, float]],
                         eps: float = _EPS) -> int:
    """Count the number of shared vertices between two polygons."""
    count = 0
    for va in verts_a:
        for vb in verts_b:
            if _vertices_match(va, vb, eps):
                count += 1
    return count


class Archimedean488Tiling(Tiling):
    """
    Truncated Square (4-8-8) Archimedean tiling.

    Alternating octagons (8 neighbors) and squares (4 neighbors) on a
    checkerboard grid.
    """

    _MAX_EDGES = 8

    def __init__(self):
        super().__init__()
        self._cell_list: list[str] = []
        self._grid_cols = 0
        self._grid_rows = 0

    @property
    def name(self) -> str:
        return "488"

    @property
    def directions(self) -> list[str]:
        return [f"edge_{i}" for i in range(self._MAX_EDGES)]

    def generate_graph(self, width: int, height: int, seed: int = 0
                       ) -> dict[str, Cell]:
        """
        Generate the 4-8-8 tiling as an adjacency graph.

        Args:
            width: Number of grid columns (of the checkerboard).
            height: Number of grid rows (of the checkerboard).
            seed: Random seed (unused for deterministic tilings).

        Returns:
            Dictionary of cell_id -> Cell.
        """
        self.width = width
        self.height = height
        self._grid_cols = width
        self._grid_rows = height
        self.cells = {}

        s = 1.0  # edge length

        # Circumradii
        oct_R = _edge_length_to_circumradius(8, s)
        sq_R = _edge_length_to_circumradius(4, s)

        # Apothems (center to edge midpoint)
        oct_apothem = oct_R * math.cos(math.pi / 8)
        sq_apothem = sq_R * math.cos(math.pi / 4)

        # Grid spacing: center-to-center distance between adjacent oct and sq
        # equals the sum of their apothems so edges align perfectly
        d = oct_apothem + sq_apothem

        # Octagon rotation: rotate by pi/8 so edges are horizontal/vertical
        oct_rot = math.pi / 8

        # Square rotation: 45 degrees so vertices point toward octagon edges
        sq_rot = math.pi / 4

        # Build all tiles
        all_tiles = []

        for row in range(height):
            for col in range(width):
                cx = col * d
                cy = row * d
                is_octagon = (row + col) % 2 == 0

                if is_octagon:
                    cell_id = f"a488_oct_{row}_{col}"
                    verts = _regular_polygon_vertices((cx, cy), 8, oct_R, oct_rot)
                    tile_type = "octagon"
                    n_sides = 8
                else:
                    cell_id = f"a488_sq_{row}_{col}"
                    verts = _regular_polygon_vertices((cx, cy), 4, sq_R, sq_rot)
                    tile_type = "square"
                    n_sides = 4

                all_tiles.append({
                    "cell_id": cell_id,
                    "tile_type": tile_type,
                    "center": (cx, cy),
                    "vertices": verts,
                    "rotation": oct_rot if is_octagon else sq_rot,
                    "n_sides": n_sides,
                    "grid_row": row,
                    "grid_col": col,
                })

        # Compute bounding box for normalization
        all_xs = []
        all_ys = []
        for tile in all_tiles:
            for vx, vy in tile["vertices"]:
                all_xs.append(vx)
                all_ys.append(vy)

        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)
        range_x = max_x - min_x if max_x > min_x else 1.0
        range_y = max_y - min_y if max_y > min_y else 1.0

        # Uniform scaling to preserve aspect ratio
        scale = max(range_x, range_y)
        if scale < _EPS:
            scale = 1.0

        def normalize(px, py):
            nx = (px - min_x) / scale
            ny = (py - min_y) / scale
            offset_x = (1.0 - range_x / scale) / 2
            offset_y = (1.0 - range_y / scale) / 2
            return nx + offset_x, ny + offset_y

        for tile in all_tiles:
            cell_id = tile["cell_id"]
            norm_center = normalize(tile["center"][0], tile["center"][1])
            norm_verts = [normalize(vx, vy) for vx, vy in tile["vertices"]]

            tiling_coords = {
                "tile_type": tile["tile_type"],
                "vertices": norm_verts,
                "center": norm_center,
                "rotation": tile["rotation"],
                "n_sides": tile["n_sides"],
            }

            self.cells[cell_id] = Cell(
                id=cell_id,
                neighbors={},
                row=tile["grid_row"],
                col=tile["grid_col"],
                position_hint=norm_center,
                tiling_coords=tiling_coords,
            )

        self._cell_list = list(self.cells.keys())

        # Build adjacency by shared-edge detection
        vertex_eps = 0.5 / scale

        # Spatial index: bucket vertices
        bucket_resolution = vertex_eps * 2
        vertex_to_cells: dict[tuple[int, int], list[str]] = {}

        for cell_id in self.cells:
            tc = self.cells[cell_id].tiling_coords
            for vx, vy in tc["vertices"]:
                bx = int(round(vx / bucket_resolution))
                by = int(round(vy / bucket_resolution))
                for dbx in [-1, 0, 1]:
                    for dby in [-1, 0, 1]:
                        key = (bx + dbx, by + dby)
                        if key not in vertex_to_cells:
                            vertex_to_cells[key] = []
                        vertex_to_cells[key].append(cell_id)

        # Find candidate neighbor pairs
        candidate_pairs: set[tuple[str, str]] = set()
        for cell_id in self.cells:
            tc = self.cells[cell_id].tiling_coords
            neighbor_candidates: set[str] = set()
            for vx, vy in tc["vertices"]:
                bx = int(round(vx / bucket_resolution))
                by = int(round(vy / bucket_resolution))
                for cid in vertex_to_cells.get((bx, by), []):
                    if cid != cell_id:
                        neighbor_candidates.add(cid)
            for cid in neighbor_candidates:
                pair = (min(cell_id, cid), max(cell_id, cid))
                candidate_pairs.add(pair)

        # Check each candidate pair
        for cid_a, cid_b in candidate_pairs:
            verts_a = self.cells[cid_a].tiling_coords["vertices"]
            verts_b = self.cells[cid_b].tiling_coords["vertices"]
            shared = _shared_vertex_count(verts_a, verts_b, vertex_eps)
            if shared >= 2:
                edge_idx_a = self._find_shared_edge_index(verts_a, verts_b, vertex_eps)
                edge_idx_b = self._find_shared_edge_index(verts_b, verts_a, vertex_eps)

                dir_a = f"edge_{edge_idx_a}"
                dir_b = f"edge_{edge_idx_b}"

                self.cells[cid_a].neighbors[dir_a] = cid_b
                self.cells[cid_b].neighbors[dir_b] = cid_a

        return self.cells

    def _find_shared_edge_index(self, verts_a: list[tuple[float, float]],
                                 verts_b: list[tuple[float, float]],
                                 eps: float) -> int:
        """
        Find which edge index of polygon A is shared with polygon B.
        An edge is (verts_a[i], verts_a[(i+1)%n]). It's shared if both
        endpoints match vertices in verts_b.
        """
        n = len(verts_a)
        for i in range(n):
            v0 = verts_a[i]
            v1 = verts_a[(i + 1) % n]
            match0 = any(_vertices_match(v0, vb, eps) for vb in verts_b)
            match1 = any(_vertices_match(v1, vb, eps) for vb in verts_b)
            if match0 and match1:
                return i
        return 0  # fallback

    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized [0,1] coordinates to nearest cell ID."""
        best_id = self._cell_list[0] if self._cell_list else ""
        best_dist = float("inf")

        for cell_id, cell in self.cells.items():
            cx, cy = cell.position_hint
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_dist:
                best_dist = d
                best_id = cell_id

        return best_id

    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized [0,1] coordinates."""
        if cell_id in self.cells:
            return self.cells[cell_id].position_hint
        return (0.5, 0.5)

    def get_neighbor(self, cell_id: str, direction: str) -> Optional[str]:
        """
        Get neighbor cell ID in given direction, or None.

        Directions beyond the cell's actual edge count return None.
        For example, a square only uses edge_0..edge_3; edge_4..edge_7
        return None.
        """
        cell = self.cells.get(cell_id)
        if cell is None:
            return None
        return cell.neighbors.get(direction)

    def distance(self, cell_a: str, cell_b: str) -> int:
        """Compute graph distance (hops) between two cells using BFS."""
        if cell_a == cell_b:
            return 0
        if cell_a not in self.cells or cell_b not in self.cells:
            return 999

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

        return 999  # unreachable
