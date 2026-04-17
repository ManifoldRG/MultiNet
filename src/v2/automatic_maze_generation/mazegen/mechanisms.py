from __future__ import annotations

from typing import List

import random

from .models import Door, Gate, Key, MazeInstance, MazeLayout, Switch, Coord
from .generators import in_bounds, neighbors4
from .solver import solve_maze


def place_key_door_pair(layout: MazeLayout, color: str = "red") -> MazeInstance:
    key_pos = layout.slots.get("pickup_1_candidates", [None])[0]
    door_pos = layout.slots.get("blocker_1_candidates", [None])[0]
    if key_pos is None or door_pos is None:
        raise ValueError("Missing pickup_1 or blocker_1 slot")
    return MazeInstance(
        width=layout.width,
        height=layout.height,
        walls=set(layout.walls),
        start=layout.start,
        goal=layout.goal,
        keys=[Key(id=f"k{color[0].upper()}", position=key_pos, color=color)],
        doors=[Door(id=f"D{color[0].upper()}", position=door_pos, requires_key=color)],
        metadata=layout.metadata | {"chain_pattern": "kd"},
    )


def place_switch_gate_pair(layout: MazeLayout, switch_id: str = "1") -> MazeInstance:
    switch_pos = layout.slots.get("pickup_1_candidates", [None])[0]
    gate_pos = layout.slots.get("blocker_1_candidates", [None])[0]
    if switch_pos is None or gate_pos is None:
        raise ValueError("Missing pickup_1 or blocker_1 slot")
    return MazeInstance(
        width=layout.width,
        height=layout.height,
        walls=set(layout.walls),
        start=layout.start,
        goal=layout.goal,
        switches=[Switch(id=f"s{switch_id}", position=switch_pos, controls=[f"g{switch_id}"])],
        gates=[Gate(id=f"g{switch_id}", position=gate_pos)],
        metadata=layout.metadata | {"chain_pattern": "sg"},
    )


def place_sequential_chain(layout: MazeLayout, chain_type: str, palette: dict) -> MazeInstance:
    p1 = layout.slots.get("pickup_1_candidates", [None])[0]
    b1 = layout.slots.get("blocker_1_candidates", [None])[0]
    p2 = layout.slots.get("pickup_2_candidates", [None])[0]
    b2 = layout.slots.get("blocker_2_candidates", [None])[0]
    if None in {p1, b1, p2, b2}:
        raise ValueError("Missing sequential chain slots")

    maze = MazeInstance(
        width=layout.width,
        height=layout.height,
        walls=set(layout.walls),
        start=layout.start,
        goal=layout.goal,
        metadata=layout.metadata | {"chain_pattern": chain_type},
    )

    if chain_type == "ks":
        key_color = palette.get("keys", ["red"])[0]
        switch_id = palette.get("switches", ["1"])[0]
        maze.keys.append(Key(id=f"k{key_color[0].upper()}", position=p1, color=key_color))
        maze.doors.append(Door(id=f"D{key_color[0].upper()}", position=b1, requires_key=key_color))
        maze.switches.append(Switch(id=f"s{switch_id}", position=p2, controls=[f"g{switch_id}"]))
        maze.gates.append(Gate(id=f"g{switch_id}", position=b2))
    elif chain_type == "sk":
        key_color = palette.get("keys", ["red"])[0]
        switch_id = palette.get("switches", ["1"])[0]
        maze.switches.append(Switch(id=f"s{switch_id}", position=p1, controls=[f"g{switch_id}"]))
        maze.gates.append(Gate(id=f"g{switch_id}", position=b1))
        maze.keys.append(Key(id=f"k{key_color[0].upper()}", position=p2, color=key_color))
        maze.doors.append(Door(id=f"D{key_color[0].upper()}", position=b2, requires_key=key_color))
    elif chain_type == "kk":
        colors = palette.get("keys", ["red", "blue"])
        c1, c2 = colors[0], colors[1]
        maze.keys.append(Key(id=f"k{c1[0].upper()}", position=p1, color=c1))
        maze.doors.append(Door(id=f"D{c1[0].upper()}", position=b1, requires_key=c1))
        maze.keys.append(Key(id=f"k{c2[0].upper()}", position=p2, color=c2))
        maze.doors.append(Door(id=f"D{c2[0].upper()}", position=b2, requires_key=c2))
    else:
        raise ValueError(f"Unsupported chain_type: {chain_type}")

    return maze


# --- distractor utilities + helpers  ---



def _open_cells_from_maze(maze: MazeInstance) -> set[Coord]:
    return {
        (x, y)
        for x in range(maze.width)
        for y in range(maze.height)
        if (x, y) not in maze.walls
    }


def _occupied_cells(maze: MazeInstance) -> set[Coord]:
    return {
        maze.start,
        maze.goal,
        *[k.position for k in maze.keys],
        *[d.position for d in maze.doors],
        *[s.position for s in maze.switches],
        *[g.position for g in maze.gates],
    }


def _find_free_open_cells(maze: MazeInstance) -> list[Coord]:
    open_cells = _open_cells_from_maze(maze)
    occupied = _occupied_cells(maze)
    return [cell for cell in open_cells if cell not in occupied]


def _find_dead_end_attachment_candidates(maze: MazeInstance) -> list[Coord]:
    open_cells = _open_cells_from_maze(maze)
    occupied = _occupied_cells(maze)

    candidates = []
    for cell in open_cells:
        if cell in occupied:
            continue

        open_nbs = [nb for nb in neighbors4(cell) if nb in open_cells]
        wall_nbs = [
            nb
            for nb in neighbors4(cell)
            if in_bounds(nb, maze.width, maze.height) and nb in maze.walls
        ]

        # A reasonable place to attach a side stub.
        if len(open_nbs) >= 1 and len(wall_nbs) >= 1:
            candidates.append(cell)

    return candidates


def _carve_dead_end_branch(maze: MazeInstance, attach: Coord, length: int = 2) -> bool:
    """
    Carve a straight dead-end branch outward from an existing open cell.

    Returns True if successful, False otherwise.
    """
    open_cells = _open_cells_from_maze(maze)
    x, y = attach

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        branch = []
        cx, cy = x, y

        ok = True
        for _ in range(length):
            cx += dx
            cy += dy
            cell = (cx, cy)

            if not in_bounds(cell, maze.width, maze.height):
                ok = False
                break
            if cell in open_cells:
                ok = False
                break

            branch.append(cell)

        if ok:
            for cell in branch:
                maze.walls.discard(cell)
            return True

    return False


def add_dead_end_distractors(
    maze: MazeInstance,
    count: int = 1,
    branch_length: int = 2,
) -> MazeInstance:
    """
    Add simple empty dead-end branches to increase navigation ambiguity.
    """
    rng = random.Random(maze.metadata.get("seed", 0) + 101)
    candidates = _find_dead_end_attachment_candidates(maze)
    rng.shuffle(candidates)

    added = 0
    for attach in candidates:
        if added >= count:
            break
        if _carve_dead_end_branch(maze, attach, length=branch_length):
            added += 1

    maze.metadata["dead_end_distractors"] = {
        "count": added,
        "branch_length": branch_length,
    }
    return maze


def add_wrong_key_distractors(maze: MazeInstance, colors: List[str]) -> MazeInstance:
    """
    Add irrelevant keys whose colors do not match any real required door.

    Preference:
    - place them on free open cells
    - avoid the current optimal path so they act like actual distractors
    """
    rng = random.Random(maze.metadata.get("seed", 0) + 117)

    # Compute current optimal path before adding distractors
    solver_result = solve_maze(maze)
    optimal_path = {tuple(p) for p in solver_result.get("path", [])}

    candidates = _find_free_open_cells(maze)
    candidates = [cell for cell in candidates if cell not in optimal_path]
    rng.shuffle(candidates)

    real_required_colors = {d.requires_key for d in maze.doors}
    distractor_colors = [c for c in colors if c not in real_required_colors]

    added_colors: list[str] = []
    for color, cell in zip(distractor_colors, candidates):
        maze.keys.append(Key(id=f"k{color[0].upper()}", position=cell, color=color))
        added_colors.append(color)

    maze.metadata["wrong_key_distractors"] = added_colors
    return maze


def add_distractor_chain(
    maze: MazeInstance,
    chain_type: str = "kd",
    color: str = "green",
) -> MazeInstance:
    """
    Add a small plausible but irrelevant subchain.

    V1 supports only a key-door distractor chain:
        key -> door -> useless dead-end
    """
    if chain_type != "kd":
        raise ValueError("V1 distractor_chain only supports chain_type='kd'")

    rng = random.Random(maze.metadata.get("seed", 0) + 129)
    candidates = _find_dead_end_attachment_candidates(maze)
    rng.shuffle(candidates)

    real_colors = {d.requires_key for d in maze.doors}
    if color in real_colors:
        return maze

    for attach in candidates:
        x, y = attach

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            c1 = (x + dx, y + dy)
            c2 = (x + 2 * dx, y + 2 * dy)
            c3 = (x + 3 * dx, y + 3 * dy)

            branch = [c1, c2, c3]
            if not all(in_bounds(c, maze.width, maze.height) for c in branch):
                continue
            if any(c not in maze.walls for c in branch):
                continue

            # Carve the branch.
            for c in branch:
                maze.walls.discard(c)

            # Safer V1 layout:
            # key first, then matching door, then useless terminal cell.
            maze.keys.append(Key(id=f"k{color[0].upper()}", position=c1, color=color))
            maze.doors.append(Door(id=f"D{color[0].upper()}", position=c2, requires_key=color))

            maze.metadata["distractor_chain"] = {
                "type": "kd",
                "color": color,
                "cells": [c1, c2, c3],
            }
            return maze

    return maze