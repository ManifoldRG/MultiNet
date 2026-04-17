from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

from .models import Coord, MazeInstance, MazeLayout
from .generators import in_bounds, neighbors4


def solve_navigation_only(layout: MazeLayout) -> dict:
    start, goal = layout.start, layout.goal
    blocked = layout.walls
    pq: List[Tuple[int, Coord]] = [(0, start)]
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    dist: Dict[Coord, int] = {start: 0}

    while pq:
        d, node = heappop(pq)
        if node == goal:
            break
        if d != dist[node]:
            continue
        for nb in neighbors4(node):
            if not in_bounds(nb, layout.width, layout.height) or nb in blocked:
                continue
            nd = d + 1
            if nb not in dist or nd < dist[nb]:
                dist[nb] = nd
                parent[nb] = node
                heappush(pq, (nd, nb))

    if goal not in dist:
        return {"is_solvable": False, "optimal_cost": None, "path": []}

    path: List[Coord] = []
    cur: Optional[Coord] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return {"is_solvable": True, "optimal_cost": len(path) - 1, "path": path}


def count_shortest_paths(layout: MazeLayout, max_count: int = 3) -> int:
    start, goal = layout.start, layout.goal
    blocked = layout.walls
    dist: Dict[Coord, int] = {start: 0}
    count: Dict[Coord, int] = {start: 1}
    pq: List[Tuple[int, Coord]] = [(0, start)]

    while pq:
        d, node = heappop(pq)
        if d != dist[node]:
            continue
        for nb in neighbors4(node):
            if not in_bounds(nb, layout.width, layout.height) or nb in blocked:
                continue
            nd = d + 1
            if nb not in dist:
                dist[nb] = nd
                count[nb] = count[node]
                heappush(pq, (nd, nb))
            elif nd == dist[nb]:
                count[nb] = min(max_count, count[nb] + count[node])

    return count.get(goal, 0)

def _maze_lookup_tables(maze: MazeInstance) -> dict:
    return {
        "key_at": {k.position: k for k in maze.keys},
        "door_at": {d.position: d for d in maze.doors},
        "switch_at": {s.position: s for s in maze.switches},
        "gate_at": {g.position: g for g in maze.gates},
        "gate_to_switches": {
            g.id: [s.id for s in maze.switches if g.id in s.controls]
            for g in maze.gates
        },
    }



def _normalize_state(
    pos: Coord,
    inventory: frozenset[str],
    opened_doors: frozenset[str],
    switch_states: frozenset[str],
) -> Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]]:
    return (pos, inventory, opened_doors, switch_states)



def _apply_cell_effects(
    maze: MazeInstance,
    pos: Coord,
    inventory: frozenset[str],
    opened_doors: frozenset[str],
    switch_states: frozenset[str],
    lookups: dict,
) -> Tuple[frozenset[str], frozenset[str], frozenset[str], List[str]]:
    inventory_set = set(inventory)
    opened_set = set(opened_doors)
    switch_set = set(switch_states)
    interactions: List[str] = []

    key = lookups["key_at"].get(pos)
    if key is not None and key.color not in inventory_set:
        inventory_set.add(key.color)
        interactions.append(f"pickup:{key.id}")

    sw = lookups["switch_at"].get(pos)
    if sw is not None and sw.id not in switch_set:
        # V1 behavior: activate once and keep on.
        switch_set.add(sw.id)
        interactions.append(f"toggle:{sw.id}")

    return frozenset(inventory_set), frozenset(opened_set), frozenset(switch_set), interactions



def _can_enter_cell(
    maze: MazeInstance,
    pos: Coord,
    inventory: frozenset[str],
    opened_doors: frozenset[str],
    switch_states: frozenset[str],
    lookups: dict,
) -> Tuple[bool, frozenset[str], frozenset[str], List[str]]:
    inventory_set = set(inventory)
    opened_set = set(opened_doors)
    interactions: List[str] = []

    door = lookups["door_at"].get(pos)
    if door is not None and door.id not in opened_set:
        if door.requires_key not in inventory_set:
            return False, inventory, opened_doors, []
        inventory_set.remove(door.requires_key)
        opened_set.add(door.id)
        interactions.append(f"open:{door.id}")

    gate = lookups["gate_at"].get(pos)
    if gate is not None:
        controllers = lookups["gate_to_switches"].get(gate.id, [])
        is_open = any(sw_id in switch_states for sw_id in controllers)
        if not is_open:
            return False, inventory, opened_doors, []
        interactions.append(f"cross:{gate.id}")

    return True, frozenset(inventory_set), frozenset(opened_set), interactions



def solve_maze(maze: MazeInstance) -> dict:
    """
    Solve a maze using shortest-path search over full agent state.

    This solver supports movement plus the current mechanism semantics:
    - keys are picked up on entry to their cell
    - doors require a matching key color and consume that key on first use
    - switches activate on first visit and remain on
    - gates are traversable when any controlling switch is on
    """
    lookups = _maze_lookup_tables(maze)

    start_inventory, start_opened, start_switches, start_interactions = _apply_cell_effects(
        maze,
        maze.start,
        frozenset(),
        frozenset(),
        frozenset(),
        lookups,
    )
    start_state = _normalize_state(maze.start, start_inventory, start_opened, start_switches)

    queue = deque([start_state])
    parent: Dict[Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]], Optional[Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]]]] = {
        start_state: None
    }
    action_taken: Dict[Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]], Tuple[str, List[str]]] = {
        start_state: ("START", start_interactions)
    }
    dist: Dict[Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]], int] = {start_state: 0}

    goal_state: Optional[Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]]] = None

    while queue:
        state = queue.popleft()
        pos, inventory, opened_doors, switch_states = state
        if pos == maze.goal:
            goal_state = state
            break

        for nb in neighbors4(pos):
            if not in_bounds(nb, maze.width, maze.height) or nb in maze.walls:
                continue

            allowed, inventory_after_entry, opened_after_entry, entry_interactions = _can_enter_cell(
                maze, nb, inventory, opened_doors, switch_states, lookups
            )
            if not allowed:
                continue

            final_inventory, final_opened, final_switches, cell_interactions = _apply_cell_effects(
                maze,
                nb,
                inventory_after_entry,
                opened_after_entry,
                switch_states,
                lookups,
            )
            next_state = _normalize_state(nb, final_inventory, final_opened, final_switches)
            if next_state in dist:
                continue

            dist[next_state] = dist[state] + 1
            parent[next_state] = state
            action_taken[next_state] = (
                f"MOVE_TO:{nb[0]},{nb[1]}",
                entry_interactions + cell_interactions,
            )
            queue.append(next_state)

    if goal_state is None:
        return {
            "is_solvable": False,
            "optimal_cost": None,
            "path": [],
            "action_sequence": [],
            "interactions": [],
            "final_inventory": [],
            "final_opened_doors": [],
            "active_switches": [],
        }

    states_path: List[Tuple[Coord, frozenset[str], frozenset[str], frozenset[str]]] = []
    cur = goal_state
    while cur is not None:
        states_path.append(cur)
        cur = parent[cur]
    states_path.reverse()

    path = [s[0] for s in states_path]
    action_sequence: List[str] = []
    interactions: List[str] = []
    for st in states_path[1:]:
        move_action, side_effects = action_taken[st]
        action_sequence.append(move_action)
        interactions.extend(side_effects)

    _, final_inventory, final_opened, final_switches = goal_state
    return {
        "is_solvable": True,
        "optimal_cost": len(path) - 1,
        "path": path,
        "action_sequence": action_sequence,
        "interactions": interactions,
        "final_inventory": sorted(final_inventory),
        "final_opened_doors": sorted(final_opened),
        "active_switches": sorted(final_switches),
    }

