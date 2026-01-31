#!/usr/bin/env python3
"""
Visualization script for MultiGrid environments.

This script creates a simple grid environment and visualizes it using matplotlib.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multigrid.env import MultiGridEnv, TilingRegistry
from multigrid.tilings import SquareTiling, HexTiling, TriangleTiling
from multigrid.agent import Action


def visualize_grid(tiling_name="square", width=10, height=10):
    """
    Visualize a grid with the specified tiling.

    Args:
        tiling_name: Type of tiling ("square", "hex", or "triangle")
        width: Grid width in cells
        height: Grid height in cells
    """
    # Create tiling
    tiling = TilingRegistry.get(tiling_name)
    cells = tiling.generate_graph(width, height, seed=0)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(f"{tiling_name.capitalize()} Grid ({width}x{height})")

    # Draw cells
    for cell_id, cell in cells.items():
        x, y = cell.position_hint

        # Draw cell based on tiling type
        if tiling_name == "square":
            # Draw square cell
            cell_size = 1.0 / width
            rect = Rectangle(
                (x - cell_size/2, y - cell_size/2),
                cell_size, cell_size,
                facecolor='lightblue',
                edgecolor='darkblue',
                linewidth=0.5
            )
            ax.add_patch(rect)

        elif tiling_name == "hex":
            # Draw hexagon cell with proper sizing to match HexTiling coordinate system
            from matplotlib.patches import RegularPolygon

            # Calculate hex size matching HexTiling._axial_to_normalized()
            width_spacing = (width - 1) if width > 1 else 1
            height_spacing = (height - 1) if height > 1 else 1
            size_from_width = 0.95 / ((width + 0.5) * math.sqrt(3)) if width > 0 else 0.1
            size_from_height = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
            size = min(size_from_width, size_from_height)

            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=size,  # Full size for edge-to-edge tiling
                orientation=math.pi / 2,  # Point top
                facecolor='lightblue',
                edgecolor='darkblue',
                linewidth=0.5
            )
            ax.add_patch(hexagon)

        elif tiling_name == "triangle":
            # Triangles are subdivisions of hexagons
            # Parse triangle ID: tri_hexcol_hexrow_triidx
            parts = cell_id.split("_")
            if len(parts) == 4:
                from multigrid.tilings.hex import OffsetCoord, offset_to_axial
                _, hex_col, hex_row, tri_idx = parts
                tri_idx = int(tri_idx)
                hex_col = int(hex_col)
                hex_row = int(hex_row)

                # Get hex center position
                offset = OffsetCoord(hex_col, hex_row)
                axial = offset_to_axial(offset)

                # Calculate hex size (same as HexTiling)
                width_spacing = (width - 1) if width > 1 else 1
                height_spacing = (height - 1) if height > 1 else 1
                size_from_width = 0.95 / ((width + 0.5) * math.sqrt(3))
                size_from_height = 0.95 / (height_spacing * 1.5)
                hex_size = min(size_from_width, size_from_height)

                # Calculate hex center in normalized coordinates
                col_pos = hex_col * math.sqrt(3) * hex_size
                row_pos = hex_row * 1.5 * hex_size
                if hex_row % 2 == 1:
                    col_pos += math.sqrt(3) / 2 * hex_size

                grid_width = (width + 0.5) * math.sqrt(3) * hex_size
                grid_height = (height - 0.5) * 1.5 * hex_size
                x_offset = (1.0 - grid_width) / 2
                y_offset = (1.0 - grid_height) / 2

                hex_center_x = col_pos + x_offset
                hex_center_y = row_pos + y_offset

                # Calculate the 3 vertices of this triangle
                # Each triangle has apex at a hex vertex and base edges to adjacent vertices
                angle_apex = math.pi / 2 - tri_idx * math.pi / 3
                angle_base1 = math.pi / 2 - ((tri_idx - 1) % 6) * math.pi / 3
                angle_base2 = math.pi / 2 - ((tri_idx + 1) % 6) * math.pi / 3

                # Apex vertex
                apex_x = hex_center_x + hex_size * math.cos(angle_apex)
                apex_y = hex_center_y - hex_size * math.sin(angle_apex)

                # Base vertices (adjacent hex vertices)
                base1_x = hex_center_x + hex_size * math.cos(angle_base1)
                base1_y = hex_center_y - hex_size * math.sin(angle_base1)

                base2_x = hex_center_x + hex_size * math.cos(angle_base2)
                base2_y = hex_center_y - hex_size * math.sin(angle_base2)

                vertices = [
                    (apex_x, apex_y),
                    (base1_x, base1_y),
                    (base2_x, base2_y)
                ]

                triangle = Polygon(
                    vertices,
                    facecolor='lightblue',
                    edgecolor='darkblue',
                    linewidth=0.5
                )
                ax.add_patch(triangle)

        # Draw cell center point
        ax.plot(x, y, 'k.', markersize=1)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='gray', label=f'{len(cells)} cells'),
        mpatches.Patch(facecolor='none', edgecolor='blue', label=f'{len(tiling.directions)} directions per cell')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'grid_visualization_{tiling_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to grid_visualization_{tiling_name}.png")
    plt.close()


def visualize_environment():
    """
    Visualize a complete environment with agent and objects.
    """
    # Create a simple task spec
    task_spec = {
        "task_id": "demo_001",
        "seed": 42,
        "scene": {
            "bounds": {"width": 1.0, "height": 1.0},
            "objects": [
                {
                    "id": "cube_red",
                    "type": "movable",
                    "color": "red",
                    "position": {"x": 0.7, "y": 0.7},
                    "size": 0.1
                },
                {
                    "id": "cube_green",
                    "type": "movable",
                    "color": "green",
                    "position": {"x": 0.3, "y": 0.7},
                    "size": 0.1
                }
            ],
            "agent": {
                "position": {"x": 0.2, "y": 0.2},
                "facing": 0
            }
        },
        "goal": {
            "predicate": "object_in_zone",
            "object_id": "cube_red",
            "zone_id": "zone_blue"
        },
        "limits": {"max_steps": 100},
        "tiling": {"type": "square", "grid_size": {"width": 10, "height": 10}}
    }

    # Create environment
    env = MultiGridEnv(task_spec, tiling="square")
    obs, info = env.reset(seed=42)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    tiling_types = ["square", "hex", "triangle"]

    for idx, tiling_name in enumerate(tiling_types):
        ax = axes[idx]
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"{tiling_name.capitalize()} Tiling (10x10)")

        # Create environment with this tiling
        task_spec["tiling"]["type"] = tiling_name
        env = MultiGridEnv(task_spec, tiling=tiling_name)
        obs, info = env.reset(seed=42)

        # Draw grid
        import math
        from matplotlib.patches import RegularPolygon
        tiling = env.tiling
        cell_size = 1.0 / 10

        # Draw all cells
        for cell_id, cell in tiling.cells.items():
            x, y = cell.position_hint

            if tiling_name == "square":
                rect = Rectangle(
                    (x - cell_size/2, y - cell_size/2),
                    cell_size, cell_size,
                    facecolor='lightgray',
                    edgecolor='gray',
                    linewidth=0.3
                )
                ax.add_patch(rect)
            elif tiling_name == "hex":
                # Calculate proper hex size matching HexTiling coordinate system
                width_spacing = 9  # 10 - 1
                height_spacing = 9  # 10 - 1
                size_from_width = 0.95 / ((10 + 0.5) * math.sqrt(3))
                size_from_height = 0.95 / (height_spacing * 1.5)
                size = min(size_from_width, size_from_height)
                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=size,  # Full size for edge-to-edge
                    orientation=math.pi / 2,
                    facecolor='lightgray',
                    edgecolor='gray',
                    linewidth=0.3
                )
                ax.add_patch(hexagon)
            elif tiling_name == "triangle":
                # Triangles are subdivisions of hexagons
                # Parse triangle ID: tri_hexcol_hexrow_triidx
                parts = cell_id.split("_")
                if len(parts) == 4:
                    from multigrid.tilings.hex import OffsetCoord, offset_to_axial
                    _, hex_col, hex_row, tri_idx = parts
                    tri_idx = int(tri_idx)
                    hex_col = int(hex_col)
                    hex_row = int(hex_row)

                    # Get hex center position
                    offset = OffsetCoord(hex_col, hex_row)
                    axial = offset_to_axial(offset)

                    # Calculate hex size (same as HexTiling)
                    width_spacing = 9  # 10 - 1
                    height_spacing = 9  # 10 - 1
                    size_from_width = 0.95 / ((10 + 0.5) * math.sqrt(3))
                    size_from_height = 0.95 / (height_spacing * 1.5)
                    hex_size = min(size_from_width, size_from_height)

                    # Calculate hex center in normalized coordinates
                    col_pos = hex_col * math.sqrt(3) * hex_size
                    row_pos = hex_row * 1.5 * hex_size
                    if hex_row % 2 == 1:
                        col_pos += math.sqrt(3) / 2 * hex_size

                    grid_width = (10 + 0.5) * math.sqrt(3) * hex_size
                    grid_height = (10 - 0.5) * 1.5 * hex_size
                    x_offset = (1.0 - grid_width) / 2
                    y_offset = (1.0 - grid_height) / 2

                    hex_center_x = col_pos + x_offset
                    hex_center_y = row_pos + y_offset

                    # Calculate the 3 vertices of this triangle
                    angle_apex = math.pi / 2 - tri_idx * math.pi / 3
                    angle_base1 = math.pi / 2 - ((tri_idx - 1) % 6) * math.pi / 3
                    angle_base2 = math.pi / 2 - ((tri_idx + 1) % 6) * math.pi / 3

                    # Apex vertex
                    apex_x = hex_center_x + hex_size * math.cos(angle_apex)
                    apex_y = hex_center_y - hex_size * math.sin(angle_apex)

                    # Base vertices (adjacent hex vertices)
                    base1_x = hex_center_x + hex_size * math.cos(angle_base1)
                    base1_y = hex_center_y - hex_size * math.sin(angle_base1)

                    base2_x = hex_center_x + hex_size * math.cos(angle_base2)
                    base2_y = hex_center_y - hex_size * math.sin(angle_base2)

                    vertices = [
                        (apex_x, apex_y),
                        (base1_x, base1_y),
                        (base2_x, base2_y)
                    ]

                    triangle = Polygon(
                        vertices,
                        facecolor='lightgray',
                        edgecolor='gray',
                        linewidth=0.3
                    )
                    ax.add_patch(triangle)

        # Draw agent
        agent_x, agent_y = tiling.cell_to_canonical(env.state.agent.cell_id)
        ax.plot(agent_x, agent_y, 'bo', markersize=15, label='Agent')

        # Draw objects
        for obj in env.state.objects.values():
            if obj.cell_id:
                obj_x, obj_y = tiling.cell_to_canonical(obj.cell_id)
                color_map = {'red': 'r', 'green': 'g', 'blue': 'b'}
                ax.plot(obj_x, obj_y, f'{color_map.get(obj.color, "k")}s', markersize=10, label=f'{obj.color} cube')

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('environment_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved environment comparison to environment_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("MultiGrid Visualization Script")
    print("=" * 50)

    # Visualize different grid types
    for tiling_name in ["square", "hex", "triangle"]:
        print(f"\nGenerating {tiling_name} grid visualization...")
        visualize_grid(tiling_name, width=10, height=10)

    # Visualize complete environments
    print("\nGenerating environment comparison...")
    visualize_environment()

    print("\n" + "=" * 50)
    print("All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - grid_visualization_square.png")
    print("  - grid_visualization_hex.png")
    print("  - grid_visualization_triangle.png")
    print("  - environment_comparison.png")
