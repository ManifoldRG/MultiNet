# test_coordinates.py

import pytest
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.tilings.square import SquareTiling
from multigrid.tilings.hex import HexTiling
from multigrid.tilings.triangle import TriangleTiling


class TestCoordinateConversion:
    """Tests for canonical <-> cell coordinate conversion."""

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_canonical_roundtrip_center(self, tiling_class):
        """Converting to cell and back gives approximately same position."""
        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        # Test center of grid
        x, y = 0.5, 0.5
        cell_id = tiling.canonical_to_cell(x, y)
        x2, y2 = tiling.cell_to_canonical(cell_id)

        # Should be within half a cell width
        assert abs(x - x2) < 0.15
        assert abs(y - y2) < 0.15

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_canonical_corners(self, tiling_class):
        """Corner positions map to boundary cells."""
        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        corners = [(0.01, 0.01), (0.99, 0.01), (0.01, 0.99), (0.99, 0.99)]

        for x, y in corners:
            cell_id = tiling.canonical_to_cell(x, y)
            assert cell_id in tiling.cells, f"Corner ({x},{y}) mapped to invalid cell"

    @pytest.mark.parametrize("tiling_class", [
        SquareTiling, HexTiling, TriangleTiling
    ])
    def test_cell_positions_unique(self, tiling_class):
        """Each cell has a unique canonical position."""
        tiling = tiling_class()
        tiling.generate_graph(10, 10, seed=0)

        positions = set()
        for cell_id in tiling.cells:
            pos = tiling.cell_to_canonical(cell_id)
            # Round to avoid floating point issues
            pos_rounded = (round(pos[0], 6), round(pos[1], 6))
            assert pos_rounded not in positions, f"Duplicate position for {cell_id}"
            positions.add(pos_rounded)
