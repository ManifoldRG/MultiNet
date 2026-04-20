# multigrid/core.py

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Cell:
    """A single cell in the grid."""
    id: str                                    # Unique identifier (e.g., "cell_0_0")
    neighbors: dict[str, str] = field(default_factory=dict)  # direction -> neighbor_cell_id
    contents: Optional[Any] = None             # Object occupying this cell
    position_hint: tuple[float, float] = (0.0, 0.0)  # Rendering position (normalized 0-1)
    tiling_coords: Any = None                  # Tiling-specific coordinates (for math)
    row: int = 0                               # Grid row (for offset/storage)
    col: int = 0                               # Grid column (for offset/storage)


@dataclass
class TilingGraph:
    """Adjacency graph representing the world topology."""
    cells: dict[str, Cell] = field(default_factory=dict)  # cell_id -> Cell
    boundary_cells: set[str] = field(default_factory=set)  # IDs of cells at world boundary
    directions: list[str] = field(default_factory=list)    # Valid direction labels for this tiling
