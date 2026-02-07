"""
Visualization script for all MultiGrid tiling types.

Generates PNG images of every tiling supported by the MultiGrid framework:
  1. Square (4-connected)
  2. Hexagonal (6-connected)
  3. Triangular (3-connected)
  4. 3-4-6-4 Rhombitrihexagonal (mixed 3/4/6 connected)
  5. 4-8-8 Truncated Square (mixed 4/8 connected)

Each tiling is rendered with cells colored by polygon type (triangle=red,
square=blue, hexagon=green, octagon=purple). For uniform tilings the polygon
type maps directly to the neighbor count; for Archimedean tilings the actual
tile_type metadata is used so boundary cells are colored correctly. A sample
cell and its neighbors are highlighted in gold, and the title shows cell
count, neighbor count range, and tiling name.

Output files are saved to the current working directory:
  - tiling_square.png
  - tiling_hex.png
  - tiling_triangle.png
  - tiling_3464.png
  - tiling_488.png
  - tiling_comparison.png (all five side-by-side)
"""

import math
import sys
import os

# Add the v1_1 directory to sys.path so multigrid imports resolve
_V1_1_DIR = os.path.dirname(os.path.abspath(__file__))
if _V1_1_DIR not in sys.path:
    sys.path.insert(0, _V1_1_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Rectangle, RegularPolygon

from multigrid.tilings import (
    SquareTiling,
    HexTiling,
    TriangleTiling,
    Archimedean3464Tiling,
    Archimedean488Tiling,
)


# ---------------------------------------------------------------------------
# Color palette: maps neighbor count to a distinct color
# ---------------------------------------------------------------------------
NEIGHBOR_COLORS = {
    3: "#E74C3C",   # red for triangles (3 neighbors)
    4: "#3498DB",   # blue for squares (4 neighbors)
    6: "#2ECC71",   # green for hexagons (6 neighbors)
    8: "#9B59B6",   # purple for octagons (8 neighbors)
}

# Colors keyed by tile_type name (used for Archimedean tilings where
# boundary cells may have fewer neighbors than their polygon's edge count)
TILE_TYPE_COLORS = {
    "triangle": "#E74C3C",
    "square":   "#3498DB",
    "hexagon":  "#2ECC71",
    "octagon":  "#9B59B6",
}

# Fallback gradient for any unexpected neighbor counts
_FALLBACK_CMAP = plt.cm.viridis


def _color_for_neighbor_count(count, min_n, max_n):
    """Return a face color based on the number of neighbors a cell has."""
    if count in NEIGHBOR_COLORS:
        return NEIGHBOR_COLORS[count]
    # Fallback: map linearly into viridis
    if max_n == min_n:
        return _FALLBACK_CMAP(0.5)
    t = (count - min_n) / (max_n - min_n)
    return _FALLBACK_CMAP(t)


def _color_for_tile_type(cell):
    """Return a face color based on the tile_type stored in tiling_coords.

    Falls back to neighbor-count coloring if tile_type is not available.
    """
    tc = cell.tiling_coords
    if isinstance(tc, dict) and "tile_type" in tc:
        tile_type = tc["tile_type"]
        if tile_type in TILE_TYPE_COLORS:
            return TILE_TYPE_COLORS[tile_type]
    return _color_for_neighbor_count(len(cell.neighbors), 0, 8)


# ---------------------------------------------------------------------------
# Per-tiling drawing helpers
# ---------------------------------------------------------------------------

def _draw_square_cell(ax, cell, cell_width, cell_height, facecolor, edgecolor,
                      linewidth=0.5, alpha=0.85):
    """Draw a single square cell as a Rectangle patch."""
    cx, cy = cell.position_hint
    rect = Rectangle(
        (cx - cell_width / 2, cy - cell_height / 2),
        cell_width,
        cell_height,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(rect)


def _draw_hex_cell(ax, cell, hex_size, facecolor, edgecolor,
                   linewidth=0.5, alpha=0.85):
    """Draw a single hexagonal cell as a RegularPolygon (pointy-top)."""
    cx, cy = cell.position_hint
    hex_patch = RegularPolygon(
        (cx, cy),
        numVertices=6,
        radius=hex_size,
        orientation=math.pi / 6,  # pointy-top
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(hex_patch)


def _draw_triangle_cell(ax, cell, facecolor, edgecolor,
                        linewidth=0.5, alpha=0.85):
    """Draw a single triangle cell using its hex_center and tri_idx."""
    tc = cell.tiling_coords
    hex_center = tc["hex_center"]
    tri_idx = tc["tri_idx"]
    hex_size = tc["hex_size"]

    # Apex vertex of the triangle is at the hex vertex
    angle_apex = math.pi / 2 - tri_idx * math.pi / 3
    apex_x = hex_center[0] + hex_size * math.cos(angle_apex)
    apex_y = hex_center[1] - hex_size * math.sin(angle_apex)

    # Two base vertices are the adjacent hex vertices
    angle_left = math.pi / 2 - ((tri_idx + 1) % 6) * math.pi / 3
    left_x = hex_center[0] + hex_size * math.cos(angle_left)
    left_y = hex_center[1] - hex_size * math.sin(angle_left)

    angle_right = math.pi / 2 - ((tri_idx - 1) % 6) * math.pi / 3
    right_x = hex_center[0] + hex_size * math.cos(angle_right)
    right_y = hex_center[1] - hex_size * math.sin(angle_right)

    # The triangle spans from the hex center to two adjacent hex vertices.
    # Actually the triangle is: center -> vertex[tri_idx] edge to vertex[tri_idx+1].
    # But the tiling splits each hexagon into 6 triangles from center to each edge.
    # So the vertices are: hex_center, hex_vertex[tri_idx], hex_vertex[(tri_idx+1)%6].
    v0 = hex_center
    v1 = (apex_x, apex_y)
    v2 = (left_x, left_y)

    tri_patch = MplPolygon(
        [v0, v1, v2],
        closed=True,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(tri_patch)


def _draw_archimedean_cell(ax, cell, facecolor, edgecolor,
                           linewidth=0.5, alpha=0.85):
    """Draw an Archimedean tiling cell using its pre-computed vertices."""
    verts = cell.tiling_coords["vertices"]
    poly = MplPolygon(
        verts,
        closed=True,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
    )
    ax.add_patch(poly)


# ---------------------------------------------------------------------------
# Tiling rendering
# ---------------------------------------------------------------------------

def _pick_sample_cell(cells):
    """Pick a sample cell that is well-connected (not on the boundary).

    Prefers cells near the center of the layout that have a high neighbor count
    relative to the maximum possible for the tiling.
    """
    if not cells:
        return None

    # Compute centroid of all cell positions
    xs = [c.position_hint[0] for c in cells.values()]
    ys = [c.position_hint[1] for c in cells.values()]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    # Find the maximum neighbor count across all cells
    max_neighbors = max(len(c.neighbors) for c in cells.values())

    # Score each cell: prefer central cells with many neighbors
    best_id = None
    best_score = float("inf")
    for cell_id, cell in cells.items():
        dist_to_center = (cell.position_hint[0] - cx) ** 2 + (cell.position_hint[1] - cy) ** 2
        # Penalize cells with fewer neighbors (boundary cells)
        neighbor_penalty = (max_neighbors - len(cell.neighbors)) * 0.5
        score = dist_to_center + neighbor_penalty
        if score < best_score:
            best_score = score
            best_id = cell_id

    return best_id


def _compute_stats(cells):
    """Compute cell count and neighbor count range."""
    if not cells:
        return 0, 0, 0
    neighbor_counts = [len(c.neighbors) for c in cells.values()]
    return len(cells), min(neighbor_counts), max(neighbor_counts)


def render_square_tiling(ax, title_extra=""):
    """Render the square tiling onto the given axes."""
    tiling = SquareTiling()
    width, height = 8, 6
    cells = tiling.generate_graph(width, height, seed=0)

    cell_count, min_n, max_n = _compute_stats(cells)
    cell_w = 1.0 / width
    cell_h = 1.0 / height

    sample_id = _pick_sample_cell(cells)
    sample_neighbors = set(cells[sample_id].neighbors.values()) if sample_id else set()

    for cell_id, cell in cells.items():
        n_count = len(cell.neighbors)
        if cell_id == sample_id:
            fc = "#F39C12"  # gold for sample cell
            ec = "#E67E22"
            lw = 2.0
        elif cell_id in sample_neighbors:
            fc = "#F5B041"  # light gold for neighbors
            ec = "#E67E22"
            lw = 1.5
        else:
            fc = _color_for_neighbor_count(n_count, min_n, max_n)
            ec = "#2C3E50"
            lw = 0.5
        _draw_square_cell(ax, cell, cell_w * 0.95, cell_h * 0.95, fc, ec, lw)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(
        f"Square Tiling (4-connected){title_extra}\n"
        f"{cell_count} cells, {min_n}-{max_n} neighbors per cell",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_hex_tiling(ax, title_extra=""):
    """Render the hexagonal tiling onto the given axes."""
    tiling = HexTiling()
    width, height = 6, 5
    cells = tiling.generate_graph(width, height, seed=0)

    cell_count, min_n, max_n = _compute_stats(cells)

    # Compute hex size for rendering (same logic as in HexTiling)
    height_spacing = (height - 1) if height > 1 else 1
    size_from_w = 0.95 / ((width + 0.5) * math.sqrt(3)) if width > 0 else 0.1
    size_from_h = 0.95 / (height_spacing * 1.5) if height_spacing > 0 else 0.1
    hex_size = min(size_from_w, size_from_h)

    sample_id = _pick_sample_cell(cells)
    sample_neighbors = set(cells[sample_id].neighbors.values()) if sample_id else set()

    for cell_id, cell in cells.items():
        n_count = len(cell.neighbors)
        if cell_id == sample_id:
            fc = "#F39C12"
            ec = "#E67E22"
            lw = 2.0
        elif cell_id in sample_neighbors:
            fc = "#F5B041"
            ec = "#E67E22"
            lw = 1.5
        else:
            fc = _color_for_neighbor_count(n_count, min_n, max_n)
            ec = "#2C3E50"
            lw = 0.5
        _draw_hex_cell(ax, cell, hex_size, fc, ec, lw)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(
        f"Hexagonal Tiling (6-connected){title_extra}\n"
        f"{cell_count} cells, {min_n}-{max_n} neighbors per cell",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_triangle_tiling(ax, title_extra=""):
    """Render the triangular tiling onto the given axes."""
    tiling = TriangleTiling()
    width, height = 4, 3
    cells = tiling.generate_graph(width, height, seed=0)

    cell_count, min_n, max_n = _compute_stats(cells)

    sample_id = _pick_sample_cell(cells)
    sample_neighbors = set(cells[sample_id].neighbors.values()) if sample_id else set()

    for cell_id, cell in cells.items():
        n_count = len(cell.neighbors)
        if cell_id == sample_id:
            fc = "#F39C12"
            ec = "#E67E22"
            lw = 2.0
        elif cell_id in sample_neighbors:
            fc = "#F5B041"
            ec = "#E67E22"
            lw = 1.5
        else:
            fc = _color_for_neighbor_count(n_count, min_n, max_n)
            ec = "#2C3E50"
            lw = 0.5
        _draw_triangle_cell(ax, cell, fc, ec, lw)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(
        f"Triangular Tiling (3-connected){title_extra}\n"
        f"{cell_count} cells, {min_n}-{max_n} neighbors per cell",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_3464_tiling(ax, title_extra=""):
    """Render the 3-4-6-4 rhombitrihexagonal tiling onto the given axes."""
    tiling = Archimedean3464Tiling()
    width, height = 3, 3
    cells = tiling.generate_graph(width, height, seed=0)

    cell_count, min_n, max_n = _compute_stats(cells)

    sample_id = _pick_sample_cell(cells)
    sample_neighbors = set(cells[sample_id].neighbors.values()) if sample_id else set()

    for cell_id, cell in cells.items():
        if cell_id == sample_id:
            fc = "#F39C12"
            ec = "#E67E22"
            lw = 2.0
        elif cell_id in sample_neighbors:
            fc = "#F5B041"
            ec = "#E67E22"
            lw = 1.5
        else:
            fc = _color_for_tile_type(cell)
            ec = "#2C3E50"
            lw = 0.5
        _draw_archimedean_cell(ax, cell, fc, ec, lw)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(
        f"3-4-6-4 Rhombitrihexagonal{title_extra}\n"
        f"{cell_count} cells, {min_n}-{max_n} neighbors per cell",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_488_tiling(ax, title_extra=""):
    """Render the 4-8-8 truncated square tiling onto the given axes."""
    tiling = Archimedean488Tiling()
    width, height = 5, 5
    cells = tiling.generate_graph(width, height, seed=0)

    cell_count, min_n, max_n = _compute_stats(cells)

    sample_id = _pick_sample_cell(cells)
    sample_neighbors = set(cells[sample_id].neighbors.values()) if sample_id else set()

    for cell_id, cell in cells.items():
        if cell_id == sample_id:
            fc = "#F39C12"
            ec = "#E67E22"
            lw = 2.0
        elif cell_id in sample_neighbors:
            fc = "#F5B041"
            ec = "#E67E22"
            lw = 1.5
        else:
            fc = _color_for_tile_type(cell)
            ec = "#2C3E50"
            lw = 0.5
        _draw_archimedean_cell(ax, cell, fc, ec, lw)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title(
        f"4-8-8 Truncated Square{title_extra}\n"
        f"{cell_count} cells, {min_n}-{max_n} neighbors per cell",
        fontsize=10, fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def _add_legend(fig):
    """Add a shared legend showing the color-to-polygon-type mapping."""
    legend_items = [
        mpatches.Patch(facecolor=NEIGHBOR_COLORS[3], edgecolor="#2C3E50",
                       label="Triangle (3 neighbors)"),
        mpatches.Patch(facecolor=NEIGHBOR_COLORS[4], edgecolor="#2C3E50",
                       label="Square (4 neighbors)"),
        mpatches.Patch(facecolor=NEIGHBOR_COLORS[6], edgecolor="#2C3E50",
                       label="Hexagon (6 neighbors)"),
        mpatches.Patch(facecolor=NEIGHBOR_COLORS[8], edgecolor="#2C3E50",
                       label="Octagon (8 neighbors)"),
        mpatches.Patch(facecolor="#F39C12", edgecolor="#E67E22",
                       label="Sample cell (highlighted)"),
        mpatches.Patch(facecolor="#F5B041", edgecolor="#E67E22",
                       label="Neighbors of sample"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderpad=0.8,
    )


# ---------------------------------------------------------------------------
# Individual image generation
# ---------------------------------------------------------------------------

def generate_individual_images():
    """Generate a separate PNG for each tiling type."""
    renderers = [
        ("tiling_square.png", render_square_tiling),
        ("tiling_hex.png", render_hex_tiling),
        ("tiling_triangle.png", render_triangle_tiling),
        ("tiling_3464.png", render_3464_tiling),
        ("tiling_488.png", render_488_tiling),
    ]

    for filename, render_fn in renderers:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        render_fn(ax)
        _add_legend(fig)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        filepath = os.path.join(_V1_1_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved {filepath}")


# ---------------------------------------------------------------------------
# Comparison image (all five side-by-side)
# ---------------------------------------------------------------------------

def generate_comparison_image():
    """Generate a single PNG showing all five tilings side-by-side."""
    fig, axes = plt.subplots(1, 5, figsize=(30, 7))

    render_square_tiling(axes[0])
    render_hex_tiling(axes[1])
    render_triangle_tiling(axes[2])
    render_3464_tiling(axes[3])
    render_488_tiling(axes[4])

    fig.suptitle(
        "MultiGrid Tiling Types -- Cells colored by polygon type",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    _add_legend(fig)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    filepath = os.path.join(_V1_1_DIR, "tiling_comparison.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all tiling visualizations."""
    print("Generating individual tiling images...")
    generate_individual_images()
    print()
    print("Generating comparison image...")
    generate_comparison_image()
    print()
    print("Done. All images saved to:", _V1_1_DIR)


if __name__ == "__main__":
    main()
