# tilings/archimedean_3464.py

"""
Rhombitrihexagonal (3-4-6-4) Archimedean Tiling

This tiling consists of regular triangles, squares, and hexagons meeting at
each vertex in the pattern 3-4-6-4:
  - Each hexagon is surrounded by 6 squares and 6 triangles.
  - Each square is shared between 2 hexagons.
  - Each triangle is shared between 3 hexagons.

Construction:
  1. Place hexagons on a lattice with translation vectors:
       a1 = (1 + sqrt(3), 0) * s
       a2 = ((1 + sqrt(3))/2, (3 + sqrt(3))/2) * s
  2. For each hexagon, compute the 6 outward squares (on each edge) and
     6 equilateral triangles (at each vertex).
  3. Deduplicate tiles that are shared between hexagons using a vertex-
     based key (rounded to a tolerance).
  4. Detect adjacency by shared edges (2 shared vertices).
"""

import math
from collections import deque
from typing import Optional
from ..base import Tiling
from ..core import Cell


# Epsilon for floating-point vertex matching
_EPS = 1e-6

# Rounding precision for deduplication keys
_ROUND_PREC = 5


def _centroid(verts: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute the centroid of a polygon given its vertices."""
    n = len(verts)
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n
    return (cx, cy)


def _vert_key(verts: list[tuple[float, float]]) -> tuple:
    """
    Create a hashable deduplication key from polygon vertices.
    Sorts the rounded vertices so that the same polygon found from
    different hexagons produces the same key.
    """
    rounded = tuple(sorted(
        (round(v[0], _ROUND_PREC), round(v[1], _ROUND_PREC)) for v in verts
    ))
    return rounded


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


def _generate_hex_surround(hc: tuple[float, float], s: float):
    """
    Generate all tiles surrounding one hexagon centered at hc with edge length s.

    Returns lists of (tile_type, vertices) for:
      - 1 hexagon
      - 6 squares (one on each hex edge)
      - 6 triangles (one at each hex vertex)
    """
    hex_R = s  # circumradius of regular hexagon with edge s

    # Pointy-top hexagon: first vertex at top, going clockwise
    hverts = []
    for i in range(6):
        angle = math.pi / 2 - i * math.pi / 3
        hverts.append((hc[0] + hex_R * math.cos(angle),
                        hc[1] + hex_R * math.sin(angle)))

    tiles = []

    # The hexagon itself
    tiles.append(("hexagon", list(hverts)))

    # Squares on each of the 6 edges
    square_list = []
    for i in range(6):
        va = hverts[i]
        vb = hverts[(i + 1) % 6]
        # Edge direction
        ex, ey = vb[0] - va[0], vb[1] - va[1]
        el = math.sqrt(ex * ex + ey * ey)
        ed = (ex / el, ey / el)
        # Two candidate perpendiculars
        p1 = (-ed[1], ed[0])
        p2 = (ed[1], -ed[0])
        # Pick the one pointing outward from hex center
        mid = ((va[0] + vb[0]) / 2 - hc[0], (va[1] + vb[1]) / 2 - hc[1])
        if p1[0] * mid[0] + p1[1] * mid[1] > 0:
            perp = p1
        else:
            perp = p2
        # Square vertices: va, vb, vb + s*perp, va + s*perp
        vc = (vb[0] + s * perp[0], vb[1] + s * perp[1])
        vd = (va[0] + s * perp[0], va[1] + s * perp[1])
        sq_verts = [va, vb, vc, vd]
        tiles.append(("square", sq_verts))
        square_list.append(sq_verts)

    # Triangles at each hex vertex
    for i in range(6):
        prev = (i - 1) % 6
        # Triangle at vertex i uses:
        #   - hex vertex i
        #   - outer vertex of square on edge (i-1), closest to vertex i
        #     = square_list[prev][3] (the vd of that square, which was from va + perp)
        #     Actually: square on edge prev has va=hverts[prev], vb=hverts[i]
        #     Its outer verts are: vc (from vb=hverts[i]), vd (from va=hverts[prev])
        #     So the outer vert near hverts[i] is vc = square_list[prev][2]
        #   - outer vertex of square on edge i, closest to vertex i
        #     = square_list[i][3] (the vd of that square, which was from va=hverts[i])
        tri_verts = [hverts[i], square_list[prev][2], square_list[i][3]]
        tiles.append(("triangle", tri_verts))

    return tiles


class Archimedean3464Tiling(Tiling):
    """
    Rhombitrihexagonal (3-4-6-4) Archimedean tiling.

    Contains triangles (3 neighbors), squares (4 neighbors), and
    hexagons (6 neighbors) arranged so that each vertex is surrounded
    by a triangle, square, hexagon, square in that order.
    """

    # Maximum edge count across all tile types in the tiling
    _MAX_EDGES = 6

    def __init__(self):
        super().__init__()
        self._cell_list: list[str] = []
        self._grid_cols = 0
        self._grid_rows = 0

    @property
    def name(self) -> str:
        return "3464"

    @property
    def directions(self) -> list[str]:
        return [f"edge_{i}" for i in range(self._MAX_EDGES)]

    def generate_graph(self, width: int, height: int, seed: int = 0
                       ) -> dict[str, Cell]:
        """
        Generate the 3-4-6-4 tiling as an adjacency graph.

        Places hexagons on a lattice, generates surrounding squares and
        triangles, deduplicates shared tiles, then detects adjacency by
        shared edges.

        Args:
            width: Number of hexagon columns in the lattice.
            height: Number of hexagon rows in the lattice.
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

        # Translation vectors for the hexagon lattice
        a1 = ((1 + math.sqrt(3)) * s, 0.0)
        a2 = (((1 + math.sqrt(3)) / 2) * s, ((3 + math.sqrt(3)) / 2) * s)

        # Step 1: Generate all tiles from all hexagon positions, with dedup
        # unique_tiles: vert_key -> {tile_type, vertices (raw)}
        unique_tiles: dict[tuple, dict] = {}

        for row in range(height):
            for col in range(width):
                hcx = col * a1[0] + row * a2[0]
                hcy = col * a1[1] + row * a2[1]
                tiles = _generate_hex_surround((hcx, hcy), s)
                for tile_type, verts in tiles:
                    key = _vert_key(verts)
                    if key not in unique_tiles:
                        unique_tiles[key] = {
                            "tile_type": tile_type,
                            "vertices": verts,
                            "n_sides": len(verts),
                        }

        # Step 2: Assign cell IDs and compute raw centers
        tile_list = []
        counters = {"hexagon": 0, "square": 0, "triangle": 0}
        for key, tile in unique_tiles.items():
            tt = tile["tile_type"]
            idx = counters[tt]
            counters[tt] += 1
            cell_id = f"a3464_{tt[0]}_{idx}"  # e.g., a3464_h_0, a3464_s_3, a3464_t_7
            center = _centroid(tile["vertices"])
            tile_list.append((cell_id, tile["tile_type"], tile["vertices"],
                              tile["n_sides"], center))

        # Step 3: Normalize all positions to [0,1]
        all_xs = []
        all_ys = []
        for _, _, verts, _, _ in tile_list:
            for vx, vy in verts:
                all_xs.append(vx)
                all_ys.append(vy)

        min_x, max_x = min(all_xs), max(all_xs)
        min_y, max_y = min(all_ys), max(all_ys)
        range_x = max_x - min_x if max_x > min_x else 1.0
        range_y = max_y - min_y if max_y > min_y else 1.0
        scale = max(range_x, range_y)
        if scale < _EPS:
            scale = 1.0

        def normalize(px, py):
            nx = (px - min_x) / scale
            ny = (py - min_y) / scale
            offset_x = (1.0 - range_x / scale) / 2
            offset_y = (1.0 - range_y / scale) / 2
            return nx + offset_x, ny + offset_y

        for cell_id, tile_type, verts, n_sides, center in tile_list:
            norm_center = normalize(center[0], center[1])
            norm_verts = [normalize(vx, vy) for vx, vy in verts]

            tiling_coords = {
                "tile_type": tile_type,
                "vertices": norm_verts,
                "center": norm_center,
                "rotation": 0.0,
                "n_sides": n_sides,
            }

            self.cells[cell_id] = Cell(
                id=cell_id,
                neighbors={},
                row=0,
                col=0,
                position_hint=norm_center,
                tiling_coords=tiling_coords,
            )

        self._cell_list = list(self.cells.keys())

        # Step 4: Build adjacency by shared-edge detection
        vertex_eps = 0.5 / scale  # scale epsilon to normalized space

        # Spatial index: bucket vertices
        bucket_resolution = vertex_eps * 2
        vertex_to_cells: dict[tuple[int, int], set[str]] = {}

        for cell_id in self.cells:
            tc = self.cells[cell_id].tiling_coords
            for vx, vy in tc["vertices"]:
                bx = int(round(vx / bucket_resolution))
                by = int(round(vy / bucket_resolution))
                for dbx in [-1, 0, 1]:
                    for dby in [-1, 0, 1]:
                        key = (bx + dbx, by + dby)
                        if key not in vertex_to_cells:
                            vertex_to_cells[key] = set()
                        vertex_to_cells[key].add(cell_id)

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

        # Check each candidate pair for shared edge
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
        For example, a triangle only uses edge_0..edge_2; edge_3..edge_5
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
