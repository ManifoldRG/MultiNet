# multigrid/visibility.py

"""
BFS-based visibility computation for MultiGrid partial observability.

Supports two modes:
  - Omnidirectional (fog_of_war): all cells within radius are visible
  - Directional (view_cone): only cells within a facing-angle cone are visible

Walls, closed doors, and closed gates block visibility propagation.
"""

import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Tiling
    from .world import WorldState


def compute_visible_cells(
    agent_cell_id: str,
    tiling: "Tiling",
    world_state: "WorldState",
    radius: int,
    facing: Optional[int] = None,
    cone_half_angle: float = math.pi / 2,
) -> set[str]:
    """
    Compute the set of cell IDs visible from the agent's position.

    Uses BFS on the adjacency graph, stopping at blocking cells (walls,
    closed doors, closed gates). If facing is provided, an angular cone
    filter is applied.

    Args:
        agent_cell_id: The agent's current cell ID.
        tiling: The tiling graph.
        world_state: Current world state (used to check blocking objects).
        radius: Maximum BFS hop distance.
        facing: Agent facing index (None = omnidirectional / fog_of_war).
        cone_half_angle: Half-angle of the view cone in radians (default 90 deg).

    Returns:
        Set of visible cell IDs.
    """
    visible = {agent_cell_id}

    # BFS frontier: (cell_id, hops_so_far)
    frontier = [(agent_cell_id, 0)]
    visited = {agent_cell_id}

    # Pre-compute agent position and facing angle for cone filtering
    agent_pos = None
    facing_angle = None
    if facing is not None:
        agent_pos = tiling.cells[agent_cell_id].position_hint
        facing_angle = _facing_to_angle(facing, tiling)

    while frontier:
        next_frontier = []
        for cell_id, hops in frontier:
            if hops >= radius:
                continue

            cell = tiling.cells.get(cell_id)
            if cell is None:
                continue

            for _direction, neighbor_id in cell.neighbors.items():
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                # Check if neighbor blocks visibility
                blocking = _is_cell_blocking(neighbor_id, world_state)

                # Apply cone filter if directional
                if facing is not None and agent_pos is not None:
                    neighbor_pos = tiling.cells[neighbor_id].position_hint
                    if not _is_in_view_cone(agent_pos, neighbor_pos, facing_angle, cone_half_angle):
                        continue

                # The cell is visible (even blocking cells are visible themselves)
                visible.add(neighbor_id)

                # But don't propagate BFS through blocking cells
                if not blocking:
                    next_frontier.append((neighbor_id, hops + 1))

        frontier = next_frontier

    return visible


def _facing_to_angle(facing: int, tiling: "Tiling") -> float:
    """
    Convert a facing direction index to an angle in radians.

    Angle convention: 0 = right (+x), pi/2 = down (+y).
    This matches the rendering coordinate system.

    For square tilings: 0=N(-pi/2), 1=E(0), 2=S(pi/2), 3=W(pi)
    For hex tilings: 0=N(-pi/2), then 60-degree increments clockwise
    """
    num_dirs = len(tiling.directions)
    tiling_name = tiling.name

    if tiling_name == "square":
        # Square: 0=N, 1=E, 2=S, 3=W
        angle_map = {0: -math.pi / 2, 1: 0.0, 2: math.pi / 2, 3: math.pi}
        return angle_map.get(facing, 0.0)
    elif tiling_name == "hex":
        # Hex: 0=N, then 60-degree clockwise increments
        return -math.pi / 2 + facing * (math.pi / 3)
    else:
        # Generic: evenly spaced, starting from up
        return -math.pi / 2 + facing * (2 * math.pi / num_dirs)


def _is_in_view_cone(
    agent_pos: tuple[float, float],
    cell_pos: tuple[float, float],
    facing_angle: float,
    half_angle: float,
) -> bool:
    """
    Check whether cell_pos is within the view cone of the agent.

    Uses canonical (normalized) coordinates for the angle check.
    """
    dx = cell_pos[0] - agent_pos[0]
    dy = cell_pos[1] - agent_pos[1]

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return True  # Same position

    angle_to_cell = math.atan2(dy, dx)
    angle_diff = abs(_normalize_angle(angle_to_cell - facing_angle))

    return angle_diff <= half_angle


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _is_cell_blocking(cell_id: str, world_state: "WorldState") -> bool:
    """
    Check if a cell contains an object that blocks visibility.

    Blocking objects: walls, closed doors, closed gates.
    """
    for obj in world_state.get_all_objects_at(cell_id):
        if obj.obj_type == "wall":
            return True
        if obj.obj_type == "door" and not getattr(obj, "is_open", False):
            return True
        if obj.obj_type == "gate" and not getattr(obj, "is_open", False):
            return True
    return False
