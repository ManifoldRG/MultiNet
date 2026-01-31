# test_distance.py

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.tilings.square import SquareTiling
from multigrid.tilings.hex import HexTiling
from multigrid.tilings.triangle import TriangleTiling


class TestDistance:
    """Tests for distance computation."""

    def test_square_manhattan_distance(self):
        """Square grid distance equals Manhattan distance."""
        tiling = SquareTiling()
        tiling.generate_graph(10, 10, seed=0)

        # Cells 3 apart horizontally
        d = tiling.distance("sq_5_2", "sq_5_5")
        assert d == 3

        # Cells 2 apart vertically
        d = tiling.distance("sq_3_5", "sq_5_5")
        assert d == 2

        # Diagonal: Manhattan = 4
        d = tiling.distance("sq_3_3", "sq_5_5")
        assert d == 4

    def test_hex_distance(self):
        """Hex grid distance uses hex metric."""
        tiling = HexTiling()
        tiling.generate_graph(10, 10, seed=0)

        # Adjacent cells are distance 1
        for cell_id, cell in list(tiling.cells.items())[:10]:  # Test first 10 cells
            for neighbor_id in cell.neighbors.values():
                assert tiling.distance(cell_id, neighbor_id) == 1

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_distance_zero_to_self(self, tiling_class):
        """Distance from cell to itself is 0."""
        tiling = tiling_class()
        tiling.generate_graph(5, 5, seed=0)

        for cell_id in list(tiling.cells.keys())[:10]:  # Test first 10 cells
            assert tiling.distance(cell_id, cell_id) == 0

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_distance_symmetry(self, tiling_class):
        """Distance is symmetric."""
        tiling = tiling_class()
        cells = tiling.generate_graph(5, 5, seed=0)

        cell_ids = list(cells.keys())[:10]  # Sample 10 cells
        for i, id1 in enumerate(cell_ids):
            for id2 in cell_ids[i+1:]:
                assert tiling.distance(id1, id2) == tiling.distance(id2, id1)
