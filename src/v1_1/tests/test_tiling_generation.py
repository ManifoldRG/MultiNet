# test_tiling_generation.py

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.tilings.square import SquareTiling
from multigrid.tilings.hex import HexTiling
from multigrid.tilings.triangle import TriangleTiling


class TestTilingGeneration:
    """Tests for tiling graph generation."""

    @pytest.mark.parametrize("tiling_class,expected_dirs", [
        (SquareTiling, 4),
        (HexTiling, 6),
        (TriangleTiling, 3),
    ])
    def test_direction_count(self, tiling_class, expected_dirs):
        """Each tiling type has correct number of directions."""
        tiling = tiling_class()
        assert len(tiling.directions) == expected_dirs

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_cell_count(self, tiling_class):
        """Grid generates expected number of cells."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=10, height=8, seed=42)

        if tiling_class == SquareTiling:
            assert len(cells) == 80  # 10 * 8
        elif tiling_class == HexTiling:
            assert len(cells) == 80  # Rectangular hex grid
        elif tiling_class == TriangleTiling:
            assert len(cells) == 480  # 10 * 8 * 6 (each hex subdivided into 6 triangles)

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_boundary_cells_have_fewer_neighbors(self, tiling_class):
        """Cells at grid boundary have fewer neighbors than interior."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=5, height=5, seed=0)

        # Corner cells should have minimum neighbors
        # Interior cells should have maximum neighbors
        neighbor_counts = [len(c.neighbors) for c in cells.values()]

        assert min(neighbor_counts) < max(neighbor_counts)

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_adjacency_symmetry(self, tiling_class):
        """If A neighbors B, then B neighbors A."""
        tiling = tiling_class()
        cells = tiling.generate_graph(width=5, height=5, seed=0)

        for cell_id, cell in cells.items():
            for direction, neighbor_id in cell.neighbors.items():
                neighbor = cells[neighbor_id]
                # Neighbor should have some direction pointing back
                assert cell_id in neighbor.neighbors.values(), \
                    f"Asymmetric: {cell_id} -> {neighbor_id} but not reverse"

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_seed_determinism(self, tiling_class):
        """Same seed produces identical graph."""
        tiling1 = tiling_class()
        tiling2 = tiling_class()

        cells1 = tiling1.generate_graph(10, 10, seed=12345)
        cells2 = tiling2.generate_graph(10, 10, seed=12345)

        assert set(cells1.keys()) == set(cells2.keys())
        for cell_id in cells1:
            assert cells1[cell_id].neighbors == cells2[cell_id].neighbors
