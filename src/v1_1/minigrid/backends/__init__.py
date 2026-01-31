"""
Backend Abstraction for Grid Environments

Provides pluggable backend implementations for gridworld environments.

Available Backends:
    MiniGridBackend: Standard MiniGrid (gymnasium) implementation
        - Square grid only
        - Full MiniGrid feature set (keys, doors, switches, gates, hazards)
        - Partial observability support
        - No zones or teleporters

    MultiGridBackend: Custom multigrid with exotic tilings
        - Square, hexagonal, and triangle tilings
        - Full mechanism set (keys, doors, switches, gates, hazards, teleporters, zones)
        - No partial observability yet

Feature Comparison (see base.py for full table):
    - MiniGrid: Best for standard square grid tasks, more mature/tested
    - MultiGrid: Required for hex/triangle tilings or zones/teleporters

Usage:
    from minigrid.backends import get_backend

    # Standard square grid
    backend = get_backend("minigrid", render_mode="rgb_array")

    # Exotic tilings (hex, triangle)
    backend = get_backend("multigrid", tiling="triangle", render_mode="rgb_array")
"""

from .base import AbstractGridBackend, GridState
from .minigrid_backend import MiniGridBackend

# MultiGridBackend is optional - requires multigrid module
try:
    from .multigrid_backend import MultiGridBackend
    _MULTIGRID_AVAILABLE = True
except ImportError:
    MultiGridBackend = None
    _MULTIGRID_AVAILABLE = False

__all__ = [
    "AbstractGridBackend",
    "GridState",
    "MiniGridBackend",
    "MultiGridBackend",
]


def get_backend(name: str, **kwargs) -> AbstractGridBackend:
    """
    Get a backend instance by name.

    Args:
        name: Backend name ("minigrid" or "multigrid")
        **kwargs: Arguments passed to backend constructor

    Returns:
        Backend instance

    Raises:
        ValueError: If backend name is unknown or unavailable
    """
    if name == "minigrid":
        return MiniGridBackend(**kwargs)
    elif name == "multigrid":
        if not _MULTIGRID_AVAILABLE:
            raise ValueError(
                "MultiGridBackend not available. "
                "Ensure multigrid module is accessible."
            )
        return MultiGridBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")
