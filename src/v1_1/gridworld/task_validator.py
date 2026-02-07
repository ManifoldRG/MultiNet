"""
Task Validator - Beatable Path Checker

Uses BFS to verify that a task specification has at least one valid
solution path from start to goal, considering mechanism dependencies
(keys -> doors, switches -> gates, block pushes).

State space: (agent_pos, agent_dir, frozenset(inventory), frozenset(active_switches),
              frozenset(open_gates), frozenset(block_positions))
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .task_spec import TaskSpecification, Position


@dataclass(frozen=True)
class ValidatorState:
    """Immutable state for BFS search."""
    agent_pos: tuple[int, int]
    inventory: frozenset  # set of key colors held
    active_switches: frozenset  # set of switch ids that are on
    open_gates: frozenset  # set of gate ids that are open
    open_doors: frozenset  # set of door ids that are open
    block_positions: frozenset  # frozenset of (block_id, x, y) tuples


class TaskValidator:
    """
    Validates that a task is beatable by exhaustive BFS.

    Checks:
    1. Goal is reachable from start
    2. All mechanism dependencies are satisfiable
    3. Block push constraints don't create deadlocks on the solution path

    Note: This explores state space ignoring agent direction since the agent
    can always turn in place. We only need to check reachability in the
    grid graph with mechanism state transitions.
    """

    def __init__(self, spec: TaskSpecification):
        self.spec = spec
        self.width, self.height = spec.maze.dimensions

        # Build wall set for fast lookup
        self.walls: set[tuple[int, int]] = set()
        for wall in spec.maze.walls:
            self.walls.add((wall.x, wall.y))
        # Border walls
        for x in range(self.width):
            self.walls.add((x, 0))
            self.walls.add((x, self.height - 1))
        for y in range(self.height):
            self.walls.add((0, y))
            self.walls.add((self.width - 1, y))

        # Build mechanism lookups
        self.doors: dict[tuple[int, int], dict] = {}
        for door in spec.mechanisms.doors:
            self.doors[(door.position.x, door.position.y)] = {
                "id": door.id,
                "color": door.requires_key,
                "locked": door.initial_state == "locked",
            }

        self.gates: dict[tuple[int, int], str] = {}
        for gate in spec.mechanisms.gates:
            self.gates[(gate.position.x, gate.position.y)] = gate.id

        self.gate_initial_open: set[str] = set()
        for gate in spec.mechanisms.gates:
            if gate.initial_state == "open":
                self.gate_initial_open.add(gate.id)

        self.switches: dict[tuple[int, int], dict] = {}
        for switch in spec.mechanisms.switches:
            self.switches[(switch.position.x, switch.position.y)] = {
                "id": switch.id,
                "controls": switch.controls,
            }

        self.keys: dict[tuple[int, int], str] = {}
        for key in spec.mechanisms.keys:
            self.keys[(key.position.x, key.position.y)] = key.color

        self.blocks: dict[tuple[int, int], str] = {}
        for block in spec.mechanisms.blocks:
            self.blocks[(block.position.x, block.position.y)] = block.id

        self.hazards: set[tuple[int, int]] = set()
        for hazard in spec.mechanisms.hazards:
            self.hazards.add((hazard.position.x, hazard.position.y))

        self.teleporter_map: dict[tuple[int, int], tuple[int, int]] = {}
        for tp in spec.mechanisms.teleporters:
            a = (tp.position_a.x, tp.position_a.y)
            b = (tp.position_b.x, tp.position_b.y)
            self.teleporter_map[a] = b
            if tp.bidirectional:
                self.teleporter_map[b] = a

        self.goal = (spec.maze.goal.x, spec.maze.goal.y)
        self.start = (spec.maze.start.x, spec.maze.start.y)
        self.key_consumption = spec.rules.key_consumption

    def validate(self, max_states: int = 500_000) -> tuple[bool, Optional[list[tuple[int, int]]], str]:
        """
        Check if the task is beatable.

        Returns:
            (is_beatable, solution_path_or_None, message)
            solution_path is a list of (x, y) positions if beatable.
        """
        initial_block_pos = frozenset(
            (bid, pos[0], pos[1]) for pos, bid in self.blocks.items()
        )

        initial_open_doors = frozenset(
            d["id"] for pos, d in self.doors.items() if not d["locked"]
        )

        initial_state = ValidatorState(
            agent_pos=self.start,
            inventory=frozenset(),
            active_switches=frozenset(),
            open_gates=frozenset(self.gate_initial_open),
            open_doors=initial_open_doors,
            block_positions=initial_block_pos,
        )

        # BFS
        queue = deque()
        queue.append((initial_state, [self.start]))
        visited: set[ValidatorState] = {initial_state}
        states_explored = 0

        while queue:
            if states_explored >= max_states:
                return False, None, f"State space exceeded {max_states} states without finding solution"

            state, path = queue.popleft()
            states_explored += 1

            # Check goal
            if state.agent_pos == self.goal:
                return True, path, f"Solution found in {len(path)} steps ({states_explored} states explored)"

            # Generate successor states by moving in 4 directions
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = state.agent_pos[0] + dx, state.agent_pos[1] + dy

                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                next_pos = (nx, ny)

                # Can't walk into walls
                if next_pos in self.walls:
                    continue

                # Can't walk into hazards
                if next_pos in self.hazards:
                    continue

                # Current block positions as a dict for lookup
                block_dict = {(bx, by): bid for bid, bx, by in state.block_positions}

                # Check door
                new_inventory = state.inventory
                new_open_doors = state.open_doors
                if next_pos in self.doors:
                    door_info = self.doors[next_pos]
                    if door_info["id"] not in state.open_doors:
                        # Door is closed/locked - need matching key
                        if door_info["color"] in state.inventory:
                            # Open the door, optionally consume key
                            new_open_doors = state.open_doors | {door_info["id"]}
                            if self.key_consumption:
                                # Remove one key of this color
                                inv_list = list(state.inventory)
                                inv_list.remove(door_info["color"])
                                new_inventory = frozenset(inv_list)
                        else:
                            continue  # Can't pass

                # Check gate
                if next_pos in self.gates:
                    gate_id = self.gates[next_pos]
                    if gate_id not in state.open_gates:
                        continue  # Closed gate, can't pass

                # Check block at next_pos
                new_block_positions = state.block_positions
                if next_pos in block_dict:
                    # Try to push block
                    push_x, push_y = nx + dx, ny + dy
                    push_pos = (push_x, push_y)
                    # Block can't be pushed into walls, other blocks, doors, gates, hazards
                    if (push_pos in self.walls or push_pos in block_dict or
                            push_pos in self.doors or push_pos in self.gates or
                            push_pos in self.hazards or
                            not (0 <= push_x < self.width and 0 <= push_y < self.height)):
                        continue  # Can't push
                    bid = block_dict[next_pos]
                    new_block_positions = (
                        state.block_positions - {(bid, nx, ny)} | {(bid, push_x, push_y)}
                    )

                # Pick up key if present (and not already picked up - keys are on the grid)
                if next_pos in self.keys:
                    key_color = self.keys[next_pos]
                    # Simple model: keys are auto-collected when walked over
                    # (In actual MiniGrid, pickup is explicit, but for reachability this is equivalent
                    #  since a rational agent would always pick up keys they encounter)
                    new_inventory = new_inventory | {key_color}

                # Toggle switch if present (walk onto switch cell)
                new_active = state.active_switches
                new_open_gates = state.open_gates
                if next_pos in self.switches:
                    sw = self.switches[next_pos]
                    sw_id = sw["id"]
                    if sw_id in state.active_switches:
                        new_active = state.active_switches - {sw_id}
                        # Close controlled gates
                        new_open_gates = state.open_gates - frozenset(sw["controls"])
                    else:
                        new_active = state.active_switches | {sw_id}
                        # Open controlled gates
                        new_open_gates = state.open_gates | frozenset(sw["controls"])

                # Handle teleporter
                actual_pos = next_pos
                if next_pos in self.teleporter_map:
                    actual_pos = self.teleporter_map[next_pos]

                new_state = ValidatorState(
                    agent_pos=actual_pos,
                    inventory=new_inventory,
                    active_switches=new_active,
                    open_gates=new_open_gates,
                    open_doors=new_open_doors,
                    block_positions=new_block_positions,
                )

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [actual_pos]))

        return False, None, f"No solution found ({states_explored} states explored, all reachable states checked)"


@dataclass
class DifficultyReport:
    """Difficulty metrics for a task."""
    task_id: str
    tier: int
    is_beatable: bool
    optimal_steps: int  # BFS shortest path length (0 if unbeatable)
    states_explored: int  # BFS search space size
    mechanism_count: int  # total interactive objects
    mechanism_types: int  # number of distinct mechanism categories used
    dependency_depth: int  # longest chain: key->door, switch->gate, etc.
    grid_area: int  # width * height
    difficulty_score: float  # composite score

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "is_beatable": self.is_beatable,
            "optimal_steps": self.optimal_steps,
            "states_explored": self.states_explored,
            "mechanism_count": self.mechanism_count,
            "mechanism_types": self.mechanism_types,
            "dependency_depth": self.dependency_depth,
            "grid_area": self.grid_area,
            "difficulty_score": round(self.difficulty_score, 2),
        }


def compute_difficulty(spec: TaskSpecification) -> DifficultyReport:
    """Compute difficulty metrics for a task specification."""
    validator = TaskValidator(spec)
    is_beatable, solution, message = validator.validate()

    optimal_steps = len(solution) - 1 if solution else 0  # -1 because path includes start
    # Extract states_explored from message
    import re
    match = re.search(r"(\d+) states explored", message)
    states_explored = int(match.group(1)) if match else 0

    # Count mechanisms
    m = spec.mechanisms
    keys_count = len(m.keys)
    doors_count = len(m.doors)
    switches_count = len(m.switches)
    gates_count = len(m.gates)
    blocks_count = len(m.blocks)
    teleporters_count = len(m.teleporters)
    hazards_count = len(m.hazards)
    mechanism_count = (keys_count + doors_count + switches_count +
                       gates_count + blocks_count + teleporters_count + hazards_count)

    # Count distinct mechanism types used
    type_flags = [
        keys_count > 0,
        doors_count > 0,
        switches_count > 0,
        gates_count > 0,
        blocks_count > 0,
        teleporters_count > 0,
        hazards_count > 0,
    ]
    mechanism_types = sum(type_flags)

    # Compute dependency depth (longest chain)
    # key -> door = depth 1, switch -> gate = depth 1
    # key + switch -> gate -> door = depth 2
    depth = 0
    if doors_count > 0 and keys_count > 0:
        depth = max(depth, 1)
    if gates_count > 0 and switches_count > 0:
        depth = max(depth, 1)
    if doors_count > 0 and keys_count > 0 and gates_count > 0 and switches_count > 0:
        depth = max(depth, 2)  # Must handle both systems
    if blocks_count > 0:
        depth = max(depth, 1)
    if teleporters_count > 0:
        depth = max(depth, 1)
    if (teleporters_count > 0 or blocks_count > 0) and (gates_count > 0 or doors_count > 0):
        depth = max(depth, 2)

    w, h = spec.maze.dimensions
    grid_area = w * h

    # Composite difficulty score:
    # Weighted combination of optimal path length, mechanism complexity,
    # state space size, and grid size
    score = (
        optimal_steps * 1.0 +          # path length (primary)
        mechanism_count * 2.0 +         # mechanism density
        mechanism_types * 3.0 +         # variety bonus
        depth * 5.0 +                   # dependency chain bonus
        (states_explored / 100.0) +     # search complexity
        (grid_area / 50.0)              # spatial scale
    )

    return DifficultyReport(
        task_id=spec.task_id,
        tier=spec.difficulty_tier,
        is_beatable=is_beatable,
        optimal_steps=optimal_steps,
        states_explored=states_explored,
        mechanism_count=mechanism_count,
        mechanism_types=mechanism_types,
        dependency_depth=depth,
        grid_area=grid_area,
        difficulty_score=score,
    )


def validate_task_file(path: str, verbose: bool = True) -> bool:
    """Validate a single task file and report difficulty."""
    spec = TaskSpecification.from_json(path)
    report = compute_difficulty(spec)

    if verbose:
        status = "PASS" if report.is_beatable else "FAIL"
        print(f"[{status}] {spec.task_id}: optimal={report.optimal_steps} steps, "
              f"mechanisms={report.mechanism_count} ({report.mechanism_types} types), "
              f"depth={report.dependency_depth}, score={report.difficulty_score}")

    return report.is_beatable


def validate_all_tasks(tasks_dir: str = "gridworld/tasks", verbose: bool = True) -> dict:
    """Validate all task files across all tiers and report difficulty."""
    import json
    from pathlib import Path

    results = {"pass": [], "fail": [], "reports": []}
    tasks_path = Path(tasks_dir)

    for tier in range(1, 6):
        tier_dir = tasks_path / f"tier{tier}"
        if not tier_dir.exists():
            continue

        if verbose:
            print(f"\n=== Tier {tier} ===")

        for task_file in sorted(tier_dir.glob("*.json")):
            spec = TaskSpecification.from_json(str(task_file))
            report = compute_difficulty(spec)
            results["reports"].append(report.to_dict())

            if verbose:
                status = "PASS" if report.is_beatable else "FAIL"
                print(f"  [{status}] {report.task_id}: optimal={report.optimal_steps} steps, "
                      f"mechanisms={report.mechanism_count}, score={report.difficulty_score}")

            if report.is_beatable:
                results["pass"].append(str(task_file))
            else:
                results["fail"].append(str(task_file))

    if verbose:
        total = len(results["pass"]) + len(results["fail"])
        print(f"\n=== Summary: {len(results['pass'])}/{total} tasks beatable ===")
        if results["fail"]:
            print("Failed tasks:")
            for f in results["fail"]:
                print(f"  - {f}")

        # Print difficulty ranking
        print("\n=== Difficulty Ranking ===")
        sorted_reports = sorted(results["reports"], key=lambda r: r["difficulty_score"])
        for r in sorted_reports:
            print(f"  {r['difficulty_score']:6.1f}  T{r['tier']}  {r['task_id']}")

    return results


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    validate_all_tasks()
