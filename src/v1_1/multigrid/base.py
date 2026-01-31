# multigrid/base.py

from abc import ABC, abstractmethod
from typing import Optional
from .core import Cell, TilingGraph


class Tiling(ABC):
    """Abstract base for all tiling types."""

    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells: dict[str, Cell] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Tiling identifier (e.g., 'square', 'hex', 'triangle')."""
        pass

    @property
    @abstractmethod
    def directions(self) -> list[str]:
        """List of valid movement directions."""
        pass

    @abstractmethod
    def generate_graph(self, width: int, height: int, seed: int) -> dict[str, Cell]:
        """Generate the adjacency graph for a world of given size."""
        pass

    @abstractmethod
    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized [0,1] coordinates to cell ID."""
        pass

    @abstractmethod
    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized [0,1] coordinates."""
        pass

    @abstractmethod
    def get_neighbor(self, cell_id: str, direction: str) -> Optional[str]:
        """Get neighbor cell ID in given direction, or None if blocked/boundary."""
        pass

    @abstractmethod
    def distance(self, cell_a: str, cell_b: str) -> int:
        """Compute graph distance (hops) between two cells."""
        pass

    def render_cell(self, cell: Cell, renderer) -> None:
        """Render a single cell using the provided renderer."""
        # Default implementation - can be overridden
        pass
