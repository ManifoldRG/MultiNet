# test_exotic_tilings.py

"""
Tests for Archimedean tilings: 3-4-6-4 (Rhombitrihexagonal) and 4-8-8 (Truncated Square).
"""

import pytest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.tilings.archimedean_3464 import Archimedean3464Tiling
from multigrid.tilings.archimedean_488 import Archimedean488Tiling
from multigrid.env import TilingRegistry


class TestArchimedean3464CellCount:
    """Tests for 3-4-6-4 tiling cell counts.

    The tiling is built by placing hexagons on a lattice and generating
    surrounding squares (6 per hex) and triangles (6 per hex), then
    deduplicating shared tiles. Each hex has exactly width*height hexagons.
    Squares are shared between 2 hexagons and triangles between 3, so
    the total depends on boundary effects. For a 1x1 grid: 1+6+6=13.
    """

    @pytest.mark.parametrize("width,height,expected_hexes", [
        (1, 1, 1),
        (2, 2, 4),
        (3, 3, 9),
        (2, 4, 8),
        (4, 2, 8),
    ])
    def test_hex_count(self, width, height, expected_hexes):
        """Number of hexagons equals width * height."""
        tiling = Archimedean3464Tiling()
        cells = tiling.generate_graph(width, height, seed=42)
        hex_count = sum(
            1 for c in cells.values()
            if c.tiling_coords["tile_type"] == "hexagon"
        )
        assert hex_count == expected_hexes, (
            f"Expected {expected_hexes} hexagons for {width}x{height} grid, "
            f"got {hex_count}"
        )

    @pytest.mark.parametrize("width,height", [
        (1, 1),
        (2, 2),
        (3, 3),
        (2, 4),
        (4, 2),
    ])
    def test_total_cell_count_positive(self, width, height):
        """Total cell count is greater than number of hexagons."""
        tiling = Archimedean3464Tiling()
        cells = tiling.generate_graph(width, height, seed=42)
        n_hex = width * height
        assert len(cells) > n_hex, (
            f"Total cells ({len(cells)}) should exceed hex count ({n_hex})"
        )


class TestArchimedean488CellCount:
    """Tests for 4-8-8 tiling cell counts."""

    @pytest.mark.parametrize("width,height", [
        (2, 2),
        (3, 3),
        (4, 4),
        (3, 5),
        (5, 3),
    ])
    def test_cell_count(self, width, height):
        """Cell count equals width * height (one tile per grid position)."""
        tiling = Archimedean488Tiling()
        cells = tiling.generate_graph(width, height, seed=42)
        expected = width * height
        assert len(cells) == expected, (
            f"Expected {expected} cells for {width}x{height} grid, got {len(cells)}"
        )


class TestAdjacencySymmetry:
    """If A neighbors B, then B must neighbor A."""

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_adjacency_symmetry(self, tiling_class):
        """Adjacency relation is symmetric."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=3, height=3, seed=0)

        for cell_id, cell in cells.items():
            for direction, neighbor_id in cell.neighbors.items():
                assert neighbor_id in cells, (
                    f"Neighbor {neighbor_id} of {cell_id} not in cells"
                )
                neighbor = cells[neighbor_id]
                assert cell_id in neighbor.neighbors.values(), (
                    f"Asymmetric adjacency: {cell_id} -> {neighbor_id} "
                    f"via {direction}, but {neighbor_id} does not neighbor "
                    f"{cell_id}. {neighbor_id} neighbors: {neighbor.neighbors}"
                )


class TestVariableNeighborCounts:
    """Tiles have the correct number of neighbors based on their polygon type."""

    def test_3464_neighbor_counts(self):
        """3-4-6-4: triangles have <=3, squares <=4, hexagons <=6 neighbors."""
        tiling = Archimedean3464Tiling()
        # Use larger grid so interior cells have full neighbor sets
        cells = tiling.generate_graph(width=4, height=4, seed=0)

        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            tile_type = tc["tile_type"]
            n_neighbors = len(cell.neighbors)

            if tile_type == "triangle":
                assert n_neighbors <= 3, (
                    f"Triangle {cell_id} has {n_neighbors} neighbors (max 3)"
                )
            elif tile_type == "square":
                assert n_neighbors <= 4, (
                    f"Square {cell_id} has {n_neighbors} neighbors (max 4)"
                )
            elif tile_type == "hexagon":
                assert n_neighbors <= 6, (
                    f"Hexagon {cell_id} has {n_neighbors} neighbors (max 6)"
                )

    def test_3464_has_all_tile_types(self):
        """3-4-6-4 tiling contains triangles, squares, and hexagons."""
        tiling = Archimedean3464Tiling()
        cells = tiling.generate_graph(width=2, height=2, seed=0)

        tile_types = set()
        for cell in cells.values():
            tile_types.add(cell.tiling_coords["tile_type"])

        assert "triangle" in tile_types, "Missing triangles in 3-4-6-4 tiling"
        assert "square" in tile_types, "Missing squares in 3-4-6-4 tiling"
        assert "hexagon" in tile_types, "Missing hexagons in 3-4-6-4 tiling"

    def test_488_neighbor_counts(self):
        """4-8-8: squares have <=4, octagons have <=8 neighbors."""
        tiling = Archimedean488Tiling()
        cells = tiling.generate_graph(width=5, height=5, seed=0)

        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            tile_type = tc["tile_type"]
            n_neighbors = len(cell.neighbors)

            if tile_type == "square":
                assert n_neighbors <= 4, (
                    f"Square {cell_id} has {n_neighbors} neighbors (max 4)"
                )
            elif tile_type == "octagon":
                assert n_neighbors <= 8, (
                    f"Octagon {cell_id} has {n_neighbors} neighbors (max 8)"
                )

    def test_488_has_both_tile_types(self):
        """4-8-8 tiling contains both squares and octagons."""
        tiling = Archimedean488Tiling()
        cells = tiling.generate_graph(width=3, height=3, seed=0)

        tile_types = set()
        for cell in cells.values():
            tile_types.add(cell.tiling_coords["tile_type"])

        assert "square" in tile_types, "Missing squares in 4-8-8 tiling"
        assert "octagon" in tile_types, "Missing octagons in 4-8-8 tiling"

    def test_488_interior_octagons_have_8_neighbors(self):
        """Interior octagons in a large-enough grid should have 8 neighbors."""
        tiling = Archimedean488Tiling()
        cells = tiling.generate_graph(width=7, height=7, seed=0)

        # Check interior cells (not on boundary rows/cols)
        found_full_octagon = False
        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            if tc["tile_type"] == "octagon":
                row, col = cell.row, cell.col
                if 1 <= row <= 5 and 1 <= col <= 5:
                    n = len(cell.neighbors)
                    if n == 8:
                        found_full_octagon = True

        assert found_full_octagon, (
            "No interior octagon found with full 8 neighbors in 7x7 grid"
        )

    def test_488_interior_squares_have_4_neighbors(self):
        """Interior squares should have 4 neighbors."""
        tiling = Archimedean488Tiling()
        cells = tiling.generate_graph(width=7, height=7, seed=0)

        found_full_square = False
        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            if tc["tile_type"] == "square":
                row, col = cell.row, cell.col
                if 1 <= row <= 5 and 1 <= col <= 5:
                    n = len(cell.neighbors)
                    if n == 4:
                        found_full_square = True

        assert found_full_square, (
            "No interior square found with full 4 neighbors in 7x7 grid"
        )


class TestCanonicalCoordinates:
    """All canonical coordinates should be in [0,1]."""

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_canonical_in_unit_interval(self, tiling_class):
        """All cell positions (position_hint) are in [0,1]."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=4, height=4, seed=42)

        for cell_id, cell in cells.items():
            x, y = cell.position_hint
            assert 0.0 <= x <= 1.0, (
                f"Cell {cell_id} x={x} out of [0,1]"
            )
            assert 0.0 <= y <= 1.0, (
                f"Cell {cell_id} y={y} out of [0,1]"
            )

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_cell_to_canonical_matches_hint(self, tiling_class):
        """cell_to_canonical returns the same as position_hint."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=3, height=3, seed=0)

        for cell_id, cell in cells.items():
            pos = tiling.cell_to_canonical(cell_id)
            assert abs(pos[0] - cell.position_hint[0]) < 1e-10
            assert abs(pos[1] - cell.position_hint[1]) < 1e-10

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_canonical_to_cell_roundtrip(self, tiling_class):
        """canonical_to_cell(cell_to_canonical(id)) should return the same cell."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=3, height=3, seed=0)

        for cell_id in cells:
            x, y = tiling.cell_to_canonical(cell_id)
            recovered = tiling.canonical_to_cell(x, y)
            assert recovered == cell_id, (
                f"Roundtrip failed for {cell_id}: "
                f"({x:.4f}, {y:.4f}) -> {recovered}"
            )

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_all_vertices_in_unit_interval(self, tiling_class):
        """All polygon vertices should be in [0,1]."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=3, height=3, seed=0)

        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            for vx, vy in tc["vertices"]:
                assert -0.01 <= vx <= 1.01, (
                    f"Cell {cell_id} vertex x={vx} out of range"
                )
                assert -0.01 <= vy <= 1.01, (
                    f"Cell {cell_id} vertex y={vy} out of range"
                )


class TestRendering:
    """Test that rendering produces valid, non-zero RGB arrays."""

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_rendering_produces_nonzero_image(self, tiling_class):
        """Rendering should produce a non-zero RGB array."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=3, height=3, seed=0)

        # Import rendering
        from multigrid.rendering import render_multigrid, MinimalRenderer

        # We need a minimal WorldState-like object for rendering
        # Create a simple stub
        class StubAgent:
            cell_id = list(cells.keys())[0]
            facing = 0
            holding = None

        class StubState:
            agent = StubAgent()
            objects = {}
            goal = None

        frame = render_multigrid(StubState(), tiling, width=256, height=256)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8
        # Should not be all-black (background is light gray)
        assert frame.sum() > 0, "Rendered frame is all black"
        # Should have some variation (not a solid color)
        assert frame.std() > 0, "Rendered frame has no variation"

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_rendering_different_sizes(self, tiling_class):
        """Rendering at different resolutions should all produce valid images."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=2, height=2, seed=0)

        from multigrid.rendering import render_multigrid

        class StubAgent:
            cell_id = list(cells.keys())[0]
            facing = 0
            holding = None

        class StubState:
            agent = StubAgent()
            objects = {}
            goal = None

        for size in [64, 128, 512]:
            frame = render_multigrid(StubState(), tiling, width=size, height=size)
            assert frame.shape == (size, size, 3)
            assert frame.sum() > 0


class TestSeedDeterminism:
    """Same seed should produce identical graphs."""

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_seed_determinism(self, tiling_class):
        """Same seed produces identical graph."""
        tiling1 = tiling_class()
        tiling2 = tiling_class()

        cells1 = tiling1.generate_graph(3, 3, seed=12345)
        cells2 = tiling2.generate_graph(3, 3, seed=12345)

        assert set(cells1.keys()) == set(cells2.keys()), (
            "Cell ID sets differ between identical seeds"
        )

        for cell_id in cells1:
            assert cells1[cell_id].neighbors == cells2[cell_id].neighbors, (
                f"Neighbors differ for {cell_id}"
            )
            pos1 = cells1[cell_id].position_hint
            pos2 = cells2[cell_id].position_hint
            assert abs(pos1[0] - pos2[0]) < 1e-12
            assert abs(pos1[1] - pos2[1]) < 1e-12

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_different_seeds_same_result(self, tiling_class):
        """Since these tilings are deterministic, different seeds should
        still produce the same graph (seed is unused)."""
        tiling1 = tiling_class()
        tiling2 = tiling_class()

        cells1 = tiling1.generate_graph(3, 3, seed=0)
        cells2 = tiling2.generate_graph(3, 3, seed=99999)

        assert set(cells1.keys()) == set(cells2.keys())
        for cell_id in cells1:
            assert cells1[cell_id].neighbors == cells2[cell_id].neighbors


class TestDistance:
    """Graph distance (BFS) computation tests."""

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_distance_self_is_zero(self, tiling_class):
        """Distance from a cell to itself is 0."""
        tiling = tiling_class()
        cells = tiling.generate_graph(3, 3, seed=0)

        for cell_id in list(cells.keys())[:5]:
            assert tiling.distance(cell_id, cell_id) == 0

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_distance_neighbors_is_one(self, tiling_class):
        """Distance between direct neighbors is 1."""
        tiling = tiling_class()
        cells = tiling.generate_graph(3, 3, seed=0)

        for cell_id, cell in list(cells.items())[:5]:
            for neighbor_id in cell.neighbors.values():
                assert tiling.distance(cell_id, neighbor_id) == 1

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_distance_symmetry(self, tiling_class):
        """Distance(A, B) == Distance(B, A)."""
        tiling = tiling_class()
        cells = tiling.generate_graph(3, 3, seed=0)

        cell_ids = list(cells.keys())
        for i in range(min(5, len(cell_ids))):
            for j in range(i + 1, min(5, len(cell_ids))):
                d1 = tiling.distance(cell_ids[i], cell_ids[j])
                d2 = tiling.distance(cell_ids[j], cell_ids[i])
                assert d1 == d2, (
                    f"Asymmetric distance: {cell_ids[i]}<->{cell_ids[j]}: "
                    f"{d1} vs {d2}"
                )


class TestGetNeighborBeyondEdgeCount:
    """get_neighbor returns None for directions beyond cell's edge count."""

    def test_3464_triangle_extra_directions(self):
        """Triangles in 3-4-6-4 should return None for edge_3..edge_5."""
        tiling = Archimedean3464Tiling()
        cells = tiling.generate_graph(3, 3, seed=0)

        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            if tc["tile_type"] == "triangle":
                n_sides = tc["n_sides"]
                # Directions beyond actual edge count should be None
                for i in range(n_sides, 6):
                    result = tiling.get_neighbor(cell_id, f"edge_{i}")
                    assert result is None, (
                        f"Triangle {cell_id} edge_{i} should be None, got {result}"
                    )
                break  # Only need to test one triangle

    def test_488_square_extra_directions(self):
        """Squares in 4-8-8 should return None for edge_4..edge_7."""
        tiling = Archimedean488Tiling()
        cells = tiling.generate_graph(4, 4, seed=0)

        for cell_id, cell in cells.items():
            tc = cell.tiling_coords
            if tc["tile_type"] == "square":
                n_sides = tc["n_sides"]
                for i in range(n_sides, 8):
                    result = tiling.get_neighbor(cell_id, f"edge_{i}")
                    assert result is None, (
                        f"Square {cell_id} edge_{i} should be None, got {result}"
                    )
                break  # Only need to test one square


class TestTilingRegistry:
    """Test that new tilings are registered properly."""

    def test_3464_registered(self):
        """3-4-6-4 tiling can be obtained from registry."""
        tiling = TilingRegistry.get("3464")
        assert tiling.name == "3464"
        assert isinstance(tiling, Archimedean3464Tiling)

    def test_488_registered(self):
        """4-8-8 tiling can be obtained from registry."""
        tiling = TilingRegistry.get("488")
        assert tiling.name == "488"
        assert isinstance(tiling, Archimedean488Tiling)


class TestConnectivity:
    """Test that the tilings produce connected graphs."""

    @pytest.mark.parametrize("tiling_class", [
        Archimedean3464Tiling,
        Archimedean488Tiling,
    ])
    def test_graph_is_connected(self, tiling_class):
        """All cells should be reachable from any starting cell (connected graph)."""
        tiling = tiling_class()
        cells = tiling.generate_graph(3, 3, seed=0)

        if len(cells) == 0:
            return

        # BFS from first cell
        start = next(iter(cells))
        visited = {start}
        from collections import deque
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor_id in cells[current].neighbors.values():
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)

        assert len(visited) == len(cells), (
            f"Graph is not connected: visited {len(visited)} of {len(cells)} cells"
        )
