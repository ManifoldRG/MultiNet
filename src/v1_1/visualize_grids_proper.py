#!/usr/bin/env python3
"""
Proper grid visualization showing actual tiled patterns.
"""

import sys
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Circle, RegularPolygon
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling


def visualize_square_grid(width=10, height=10):
    """Visualize square grid with proper tiling."""
    tiling = SquareTiling()
    tiling.generate_graph(width, height, seed=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Square Tiling ({width}×{height} cells, 4 directions per cell)", fontsize=14)

    cell_size = 1.0 / width

    # Draw all cells
    for cell_id, cell in tiling.cells.items():
        x_norm, y_norm = cell.position_hint

        # Draw square
        square = mpatches.Rectangle(
            (x_norm - cell_size/2, y_norm - cell_size/2),
            cell_size, cell_size,
            fill=True,
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=0.5
        )
        ax.add_patch(square)

        # Draw cell center
        ax.plot(x_norm, y_norm, 'k.', markersize=1)

    # Highlight a sample cell and its neighbors
    sample_cell_id = f"sq_5_5"
    if sample_cell_id in tiling.cells:
        cell = tiling.cells[sample_cell_id]
        x, y = cell.position_hint

        # Highlight center cell
        square = mpatches.Rectangle(
            (x - cell_size/2, y - cell_size/2),
            cell_size, cell_size,
            fill=True,
            facecolor='yellow',
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(square)

        # Highlight neighbors
        for direction, neighbor_id in cell.neighbors.items():
            neighbor = tiling.cells[neighbor_id]
            nx, ny = neighbor.position_hint
            square = mpatches.Rectangle(
                (nx - cell_size/2, ny - cell_size/2),
                cell_size, cell_size,
                fill=True,
                facecolor='lightgreen',
                edgecolor='green',
                linewidth=1.5
            )
            ax.add_patch(square)

    plt.savefig('square_grid_proper.png', dpi=150, bbox_inches='tight')
    print("Saved square_grid_proper.png")
    plt.close()


def visualize_hex_grid(width=10, height=10):
    """Visualize hexagonal grid with proper tiling."""
    tiling = HexTiling()
    tiling.generate_graph(width, height, seed=0)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Hexagonal Tiling ({width}×{height} cells, 6 directions per cell)", fontsize=14)

    # Calculate hex size based on grid dimensions
    hex_width_units = width * math.sqrt(3)
    hex_height_units = height * 1.5 + 0.5
    size = min(1.0 / hex_width_units, 1.0 / hex_height_units)

    # Draw all hexagons
    for cell_id, cell in tiling.cells.items():
        x_norm, y_norm = cell.position_hint

        # Create hexagon vertices
        hexagon = RegularPolygon(
            (x_norm, y_norm),
            numVertices=6,
            radius=size * 0.98,  # Slightly smaller to see edges
            orientation=math.pi / 2,  # Point top
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=0.5
        )
        ax.add_patch(hexagon)

        # Draw cell center
        ax.plot(x_norm, y_norm, 'k.', markersize=1)

    # Highlight a sample cell in the middle and its neighbors
    mid_cells = [c for c in tiling.cells.values() if 0.4 < c.position_hint[0] < 0.6 and 0.4 < c.position_hint[1] < 0.6]
    if mid_cells:
        cell = mid_cells[0]
        x, y = cell.position_hint

        # Highlight center cell
        hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=size * 0.98,
            orientation=math.pi / 2,
            facecolor='yellow',
            edgecolor='red',
            linewidth=2
        )
        ax.add_patch(hexagon)

        # Highlight neighbors
        for direction, neighbor_id in cell.neighbors.items():
            neighbor = tiling.cells[neighbor_id]
            nx, ny = neighbor.position_hint
            hexagon = RegularPolygon(
                (nx, ny),
                numVertices=6,
                radius=size * 0.98,
                orientation=math.pi / 2,
                facecolor='lightgreen',
                edgecolor='green',
                linewidth=1.5
            )
            ax.add_patch(hexagon)

    plt.savefig('hex_grid_proper.png', dpi=150, bbox_inches='tight')
    print("Saved hex_grid_proper.png")
    plt.close()


def visualize_triangle_grid(width=10, height=10):
    """Visualize triangular grid with proper tiling."""
    tiling = TriangleTiling()
    tiling.generate_graph(width, height, seed=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Triangular Tiling ({width}×{height} cells, 3 edges per cell)", fontsize=14)

    cell_size = 1.0 / width

    # Draw all triangles
    for cell_id, cell in tiling.cells.items():
        x_norm, y_norm = cell.position_hint

        # Determine if triangle points up or down
        pointing_up = (cell.row + cell.col) % 2 == 0

        if pointing_up:
            # Upward pointing triangle
            vertices = [
                (x_norm, y_norm - cell_size * 0.4),
                (x_norm - cell_size * 0.4, y_norm + cell_size * 0.2),
                (x_norm + cell_size * 0.4, y_norm + cell_size * 0.2)
            ]
        else:
            # Downward pointing triangle
            vertices = [
                (x_norm, y_norm + cell_size * 0.4),
                (x_norm - cell_size * 0.4, y_norm - cell_size * 0.2),
                (x_norm + cell_size * 0.4, y_norm - cell_size * 0.2)
            ]

        triangle = Polygon(
            vertices,
            fill=True,
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=0.5
        )
        ax.add_patch(triangle)

        # Draw cell center
        ax.plot(x_norm, y_norm, 'k.', markersize=1)

    plt.savefig('triangle_grid_proper.png', dpi=150, bbox_inches='tight')
    print("Saved triangle_grid_proper.png")
    plt.close()


def create_comparison():
    """Create side-by-side comparison of all three tilings."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    tilings = [
        (SquareTiling(), "Square (4-connected)", 'square_cell'),
        (HexTiling(), "Hexagonal (6-connected)", 'hex_cell'),
        (TriangleTiling(), "Triangular (3-connected)", 'tri_cell')
    ]

    width, height = 8, 8

    for ax, (tiling_obj, title, prefix) in zip(axes, tilings):
        tiling_obj.generate_graph(width, height, seed=0)

        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        if isinstance(tiling_obj, SquareTiling):
            cell_size = 1.0 / width
            for cell in list(tiling_obj.cells.values())[:64]:
                x, y = cell.position_hint
                square = mpatches.Rectangle(
                    (x - cell_size/2, y - cell_size/2),
                    cell_size, cell_size,
                    fill=True,
                    facecolor='lightblue',
                    edgecolor='darkblue',
                    linewidth=0.8
                )
                ax.add_patch(square)

        elif isinstance(tiling_obj, HexTiling):
            hex_width_units = width * math.sqrt(3)
            hex_height_units = height * 1.5 + 0.5
            size = min(1.0 / hex_width_units, 1.0 / hex_height_units)

            for cell in list(tiling_obj.cells.values())[:64]:
                x, y = cell.position_hint
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=size * 0.98,
                    orientation=math.pi / 2,
                    facecolor='lightblue',
                    edgecolor='darkblue',
                    linewidth=0.8
                )
                ax.add_patch(hexagon)

        elif isinstance(tiling_obj, TriangleTiling):
            cell_size = 1.0 / width
            for cell in list(tiling_obj.cells.values())[:64]:
                x, y = cell.position_hint
                pointing_up = (cell.row + cell.col) % 2 == 0

                if pointing_up:
                    vertices = [
                        (x, y - cell_size * 0.4),
                        (x - cell_size * 0.4, y + cell_size * 0.2),
                        (x + cell_size * 0.4, y + cell_size * 0.2)
                    ]
                else:
                    vertices = [
                        (x, y + cell_size * 0.4),
                        (x - cell_size * 0.4, y - cell_size * 0.2),
                        (x + cell_size * 0.4, y - cell_size * 0.2)
                    ]

                triangle = Polygon(
                    vertices,
                    fill=True,
                    facecolor='lightblue',
                    edgecolor='darkblue',
                    linewidth=0.8
                )
                ax.add_patch(triangle)

    plt.tight_layout()
    plt.savefig('tiling_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved tiling_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Generating proper grid visualizations...")
    print("=" * 50)

    visualize_square_grid(10, 10)
    visualize_hex_grid(10, 10)
    visualize_triangle_grid(10, 10)
    create_comparison()

    print("=" * 50)
    print("All visualizations created!")
    print("\nGenerated files:")
    print("  - square_grid_proper.png")
    print("  - hex_grid_proper.png")
    print("  - triangle_grid_proper.png")
    print("  - tiling_comparison.png")
