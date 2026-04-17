from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from .models import MazeInstance, MazeLayout, ValidationParams
from .solver import count_shortest_paths, solve_maze, solve_navigation_only


def validate_navigation_layout(layout: MazeLayout, params: ValidationParams) -> dict:
    result = solve_navigation_only(layout)
    reasons: List[str] = []
    if params.require_solvable and not result["is_solvable"]:
        reasons.append("maze is not solvable")
    if params.require_unique_shortest_path and result["is_solvable"]:
        nsp = count_shortest_paths(layout)
        if nsp != 1:
            reasons.append(f"expected unique shortest path, found {nsp}")
    return {
        "is_valid": len(reasons) == 0,
        "reasons": reasons,
        "solver_result": result,
    }









def _clone_maze(maze: MazeInstance) -> MazeInstance:
    return MazeInstance(
        width=maze.width,
        height=maze.height,
        walls=set(maze.walls),
        start=maze.start,
        goal=maze.goal,
        keys=[replace(k) for k in maze.keys],
        doors=[replace(d) for d in maze.doors],
        switches=[replace(s, controls=list(s.controls)) for s in maze.switches],
        gates=[replace(g) for g in maze.gates],
        metadata=dict(maze.metadata),
    )



def _remove_mechanism_by_id(maze: MazeInstance, mech_id: str) -> MazeInstance:
    new_maze = _clone_maze(maze)
    new_maze.keys = [k for k in new_maze.keys if k.id != mech_id]
    new_maze.doors = [d for d in new_maze.doors if d.id != mech_id]
    new_maze.switches = [s for s in new_maze.switches if s.id != mech_id]
    new_maze.gates = [g for g in new_maze.gates if g.id != mech_id]

    for sw in new_maze.switches:
        sw.controls = [gid for gid in sw.controls if gid != mech_id]
    return new_maze

def _extract_required_ids(maze: MazeInstance, expected_logic: Optional[str]) -> List[str]:
    if expected_logic is None:
        return []

    if expected_logic == "kd":
        return [maze.keys[0].id] if maze.keys else []

    if expected_logic == "sg":
        return [maze.switches[0].id] if maze.switches else []

    if expected_logic == "ks":
        ids = []
        if maze.keys:
            ids.append(maze.keys[0].id)
        if maze.switches:
            ids.append(maze.switches[0].id)
        return ids

    if expected_logic == "sk":
        ids = []
        if maze.switches:
            ids.append(maze.switches[0].id)
        if maze.keys:
            ids.append(maze.keys[0].id)
        return ids

    if expected_logic == "kk":
        return [k.id for k in maze.keys[:2]]

    return []



def _run_ablation_checks(maze: MazeInstance, expected_logic: Optional[str]) -> List[str]:
    reasons: List[str] = []
    for mech_id in _extract_required_ids(maze, expected_logic):
        ablated = _remove_mechanism_by_id(maze, mech_id)
        result = solve_maze(ablated)
        if result["is_solvable"]:
            reasons.append(f"mechanism {mech_id} is not necessary under ablation")
    return reasons



def validate_maze(maze: MazeInstance, expected_logic: Optional[str] = None) -> dict:
    solver_result = solve_maze(maze)
    reasons: List[str] = []
    if not solver_result["is_solvable"]:
        reasons.append("maze is not solvable")

    chain_pattern = maze.metadata.get("chain_pattern")
    if expected_logic is not None and chain_pattern not in {expected_logic, None}:
        reasons.append("chain pattern metadata does not match expected logic")

    interactions = solver_result.get("interactions", [])
    if expected_logic == "kd":
        if not any(x.startswith("pickup:k") for x in interactions):
            reasons.append("expected kd maze to require a key pickup")
        if not any(x.startswith("open:D") for x in interactions):
            reasons.append("expected kd maze to require opening a door")
    elif expected_logic == "sg":
        if not any(x.startswith("toggle:s") for x in interactions):
            reasons.append("expected sg maze to require activating a switch")
        if not any(x.startswith("cross:g") for x in interactions):
            reasons.append("expected sg maze to require crossing a gate")
    elif expected_logic in {"ks", "sk", "kk"}:
        required_prefixes = {
            "ks": ["pickup:k", "open:D", "toggle:s", "cross:g"],
            "sk": ["toggle:s", "cross:g", "pickup:k", "open:D"],
            "kk": ["pickup:k", "open:D", "pickup:k", "open:D"],
        }[expected_logic]
        idx = 0
        for interaction in interactions:
            if interaction.startswith(required_prefixes[idx]):
                idx += 1
                if idx == len(required_prefixes):
                    break
        if idx < len(required_prefixes):
            reasons.append(f"expected ordered chain {expected_logic} was not observed in solver interactions")

    if solver_result["is_solvable"] and expected_logic is not None:
        reasons.extend(_run_ablation_checks(maze, expected_logic))

    return {
        "is_valid": len(reasons) == 0,
        "reasons": reasons,
        "solver_result": solver_result,
    }

