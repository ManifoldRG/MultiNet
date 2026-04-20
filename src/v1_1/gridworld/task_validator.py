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
    carrying_key: Optional[str]  # key id currently held
    collected_keys: frozenset  # key ids removed from the map
    active_switches: frozenset  # set of switch ids that are on
    used_switches: frozenset  # one-shot switches already used
    open_gates: frozenset  # set of gate ids that are open
    open_doors: frozenset  # set of door ids that are open
    block_positions: frozenset  # frozenset of (block_id, x, y) tuples


@dataclass(frozen=True)
class SuccessorTransition:
    """One abstract transition in validator state space."""
    next_state: ValidatorState
    next_pos: tuple[int, int]
    action_label: str


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
        self.gate_states: dict[str, bool] = {}
        for gate in spec.mechanisms.gates:
            self.gates[(gate.position.x, gate.position.y)] = gate.id
            self.gate_states[gate.id] = gate.initial_state == "open"

        self.gate_initial_open: set[str] = set()
        for gate in spec.mechanisms.gates:
            if gate.initial_state == "open":
                self.gate_initial_open.add(gate.id)

        self.switches: dict[tuple[int, int], dict] = {}
        for switch in spec.mechanisms.switches:
            self.switches[(switch.position.x, switch.position.y)] = {
                "id": switch.id,
                "controls": switch.controls,
                "switch_type": switch.switch_type,
                "initial_state": switch.initial_state,
            }

        self.switches_by_id: dict[str, dict] = {
            sw["id"]: sw for sw in self.switches.values()
        }

        self.keys: dict[tuple[int, int], dict] = {}
        self.keys_by_id: dict[str, dict] = {}
        for key in spec.mechanisms.keys:
            data = {"id": key.id, "color": key.color, "position": (key.position.x, key.position.y)}
            self.keys[(key.position.x, key.position.y)] = data
            self.keys_by_id[key.id] = data

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

    def _recompute_open_gates(self, active_switches: frozenset) -> frozenset:
        """Recompute gate openness from initial state and current switch activity."""
        open_gates = set(
            gate_id for gate_id, is_open in self.gate_states.items() if is_open
        )
        for sw in self.switches.values():
            if sw["id"] in active_switches:
                open_gates.update(sw["controls"])
        return frozenset(open_gates)

    def _apply_switch_activation(
        self,
        state: ValidatorState,
        switch_info: dict,
    ) -> Optional[tuple[frozenset, frozenset, frozenset]]:
        """Apply switch semantics and return updated (active, used, open_gates)."""
        switch_id = switch_info["id"]
        switch_type = switch_info.get("switch_type", "toggle")
        active = set(state.active_switches)
        used = set(state.used_switches)

        if switch_type == "one_shot":
            if switch_id in used:
                return None
            used.add(switch_id)
            active.add(switch_id)
        elif switch_type == "hold":
            active.add(switch_id)
        else:
            if switch_id in active:
                active.remove(switch_id)
            else:
                active.add(switch_id)

        active_fs = frozenset(active)
        return active_fs, frozenset(used), self._recompute_open_gates(active_fs)

    def _successors(self, state: ValidatorState) -> list[SuccessorTransition]:
        """Generate abstract successor transitions from a validator state."""
        successors: list[SuccessorTransition] = []

        for dx, dy, move_label in [
            (0, -1, "move_up"),
            (0, 1, "move_down"),
            (-1, 0, "move_left"),
            (1, 0, "move_right"),
        ]:
            nx, ny = state.agent_pos[0] + dx, state.agent_pos[1] + dy

            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue

            next_pos = (nx, ny)
            if next_pos in self.walls or next_pos in self.hazards:
                continue

            block_dict = {(bx, by): bid for bid, bx, by in state.block_positions}

            new_carrying_key = state.carrying_key
            new_collected_keys = state.collected_keys
            new_open_doors = state.open_doors
            new_block_positions = state.block_positions
            action_label = move_label

            if next_pos in self.doors:
                door_info = self.doors[next_pos]
                if door_info["id"] not in state.open_doors:
                    held_color = None
                    if state.carrying_key is not None:
                        held_color = self.keys_by_id[state.carrying_key]["color"]
                    if held_color == door_info["color"]:
                        new_open_doors = state.open_doors | {door_info["id"]}
                        action_label = f"open_door:{door_info['id']}"
                        if self.key_consumption:
                            new_carrying_key = None
                    else:
                        continue

            if next_pos in self.gates:
                gate_id = self.gates[next_pos]
                if gate_id not in state.open_gates:
                    continue

            if next_pos in block_dict:
                push_x, push_y = nx + dx, ny + dy
                push_pos = (push_x, push_y)
                if (
                    push_pos in self.walls
                    or push_pos in block_dict
                    or push_pos in self.doors
                    or push_pos in self.gates
                    or push_pos in self.hazards
                    or not (0 <= push_x < self.width and 0 <= push_y < self.height)
                ):
                    continue
                bid = block_dict[next_pos]
                new_block_positions = (
                    state.block_positions - {(bid, nx, ny)} | {(bid, push_x, push_y)}
                )
                action_label = f"push:{bid}:{push_x},{push_y}"

            actual_pos = next_pos
            if next_pos in self.teleporter_map:
                actual_pos = self.teleporter_map[next_pos]
                action_label = f"teleport:{next_pos}->{actual_pos}"

            successor_variants = [
                (new_carrying_key, new_collected_keys, action_label)
            ]
            if next_pos in self.keys:
                key_info = self.keys[next_pos]
                if key_info["id"] not in state.collected_keys and new_carrying_key is None:
                    successor_variants.append(
                        (
                            key_info["id"],
                            state.collected_keys | {key_info["id"]},
                            f"pickup:{key_info['id']}",
                        )
                    )

            for carrying_key, collected_keys, label in successor_variants:
                successors.append(
                    SuccessorTransition(
                        next_state=ValidatorState(
                            agent_pos=actual_pos,
                            carrying_key=carrying_key,
                            collected_keys=collected_keys,
                            active_switches=state.active_switches,
                            used_switches=state.used_switches,
                            open_gates=state.open_gates,
                            open_doors=new_open_doors,
                            block_positions=new_block_positions,
                        ),
                        next_pos=actual_pos,
                        action_label=label,
                    )
                )

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            target_pos = (state.agent_pos[0] + dx, state.agent_pos[1] + dy)
            if target_pos not in self.switches:
                continue
            switch_info = self.switches[target_pos]
            result = self._apply_switch_activation(state, switch_info)
            if result is None:
                continue
            new_active, new_used_switches, new_open_gates = result
            successors.append(
                SuccessorTransition(
                    next_state=ValidatorState(
                        agent_pos=state.agent_pos,
                        carrying_key=state.carrying_key,
                        collected_keys=state.collected_keys,
                        active_switches=new_active,
                        used_switches=new_used_switches,
                        open_gates=new_open_gates,
                        open_doors=state.open_doors,
                        block_positions=state.block_positions,
                    ),
                    next_pos=state.agent_pos,
                    action_label=f"toggle:{switch_info['id']}",
                )
            )

        return successors

    def _find_solution(
        self,
        initial_state: ValidatorState,
        goal: Optional[tuple[int, int]] = None,
        max_states: int = 500_000,
    ) -> tuple[bool, Optional[list[tuple[int, int]]], int]:
        """Run BFS from an arbitrary validator state."""
        target = self.goal if goal is None else goal
        queue = deque([(initial_state, [initial_state.agent_pos])])
        visited: set[ValidatorState] = {initial_state}
        states_explored = 0

        while queue:
            if states_explored >= max_states:
                return False, None, states_explored

            state, path = queue.popleft()
            states_explored += 1
            if state.agent_pos == target:
                return True, path, states_explored

            for transition in self._successors(state):
                if transition.next_state not in visited:
                    visited.add(transition.next_state)
                    queue.append((transition.next_state, path + [transition.next_pos]))

        return False, None, states_explored

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

        initial_active_switches = frozenset(
            sw["id"] for sw in self.switches.values() if sw.get("initial_state") == "on"
        )
        initial_used_switches = frozenset(
            sw["id"]
            for sw in self.switches.values()
            if sw.get("initial_state") == "on" and sw.get("switch_type") == "one_shot"
        )
        initial_state = ValidatorState(
            agent_pos=self.start,
            carrying_key=None,
            collected_keys=frozenset(),
            active_switches=initial_active_switches,
            used_switches=initial_used_switches,
            open_gates=self._recompute_open_gates(initial_active_switches),
            open_doors=initial_open_doors,
            block_positions=initial_block_pos,
        )

        beatable, path, states_explored = self._find_solution(initial_state, max_states=max_states)
        if beatable:
            return True, path, f"Solution found in {len(path)} steps ({states_explored} states explored)"
        if states_explored >= max_states:
            return False, None, f"State space exceeded {max_states} states without finding solution"
        return False, None, f"No solution found ({states_explored} states explored, all reachable states checked)"

    def _spec_without_mechanism(self, mechanism_id: str) -> TaskSpecification:
        """Return a copy of the spec with a single mechanism removed by id."""
        data = self.spec.to_dict()
        mechanisms = data.get("mechanisms", {})
        for key in ("keys", "doors", "switches", "gates", "blocks", "teleporters", "hazards"):
            mechanisms[key] = [
                item for item in mechanisms.get(key, [])
                if item.get("id") != mechanism_id
            ]
        if data.get("dependency_chain"):
            data["dependency_chain"]["sequence"] = [
                step for step in data["dependency_chain"].get("sequence", [])
                if step.get("element") != mechanism_id and step.get("unlocks") != mechanism_id
            ]
            data["dependency_chain"]["depth"] = len(data["dependency_chain"]["sequence"])
        return TaskSpecification.from_dict(data)

    def validate_mechanism_necessity(self) -> list[str]:
        """Report mechanisms whose removal still leaves the task solvable."""
        if self.spec.dependency_chain is not None:
            mechanism_ids = [step.element for step in self.spec.dependency_chain.sequence]
        else:
            mechanism_ids = [
                obj.id
                for group in (
                    self.spec.mechanisms.keys,
                    self.spec.mechanisms.doors,
                    self.spec.mechanisms.switches,
                    self.spec.mechanisms.gates,
                    self.spec.mechanisms.blocks,
                    self.spec.mechanisms.teleporters,
                    self.spec.mechanisms.hazards,
                )
                for obj in group
            ]

        violations = []
        for mechanism_id in dict.fromkeys(mechanism_ids):
            stripped_spec = self._spec_without_mechanism(mechanism_id)
            beatable, _, _ = TaskValidator(stripped_spec).validate()
            if beatable:
                violations.append(f"Mechanism {mechanism_id} is not necessary")
        return violations

    def _spec_with_steps_triggered(self, steps: list) -> TaskSpecification:
        """Return a copy of the spec with the provided dependency steps pre-triggered."""
        data = self.spec.to_dict()
        mechanisms = data.get("mechanisms", {})

        for step in steps:
            if step.type == "key-door":
                for door in mechanisms.get("doors", []):
                    if door.get("id") == step.unlocks:
                        door["initial_state"] = "open"
            elif step.type == "switch-gate":
                for switch in mechanisms.get("switches", []):
                    if switch.get("id") == step.element:
                        switch["initial_state"] = "on"
                for gate in mechanisms.get("gates", []):
                    if gate.get("id") == step.unlocks:
                        gate["initial_state"] = "open"
        return TaskSpecification.from_dict(data)

    def _get_element_position(self, element_id: str) -> Optional[tuple[int, int]]:
        """Locate a mechanism by id and return its grid position."""
        for group in (
            self.spec.mechanisms.keys,
            self.spec.mechanisms.doors,
            self.spec.mechanisms.switches,
            self.spec.mechanisms.gates,
            self.spec.mechanisms.blocks,
            self.spec.mechanisms.hazards,
        ):
            for obj in group:
                if obj.id == element_id:
                    return obj.position.to_tuple()
        return None

    def validate_chain_ordering(self) -> bool:
        """Verify that each next chain element is unreachable until the prior step is triggered."""
        if self.spec.dependency_chain is None or len(self.spec.dependency_chain.sequence) <= 1:
            return True

        sequence = self.spec.dependency_chain.sequence
        for idx in range(len(sequence) - 1):
            current_step = sequence[idx]
            prior_steps = sequence[:idx]
            next_step = sequence[idx + 1]
            next_pos = self._get_element_position(next_step.element)
            if next_pos is None:
                return False
            staged_spec = self._spec_with_steps_triggered(prior_steps)
            staged_spec = TaskValidator(staged_spec)._spec_without_mechanism(current_step.element)
            staged_data = staged_spec.to_dict()
            staged_data["maze"]["goal"] = list(next_pos)
            staged_data["goal"] = {"type": "reach_position", "target": list(next_pos)}
            staged_target_spec = TaskSpecification.from_dict(staged_data)
            beatable, _, _ = TaskValidator(staged_target_spec).validate()
            if beatable:
                return False
        return True

    def validate_distractor_safety(self) -> list[str]:
        """Check whether a single distractor interaction can make the task unsolvable."""
        if not self.spec.distractors:
            return []

        base_beatable, _, _ = self.validate()
        if not base_beatable:
            return ["Base task is not solvable"]

        initial_block_pos = frozenset(
            (bid, pos[0], pos[1]) for pos, bid in self.blocks.items()
        )
        initial_open_doors = frozenset(
            d["id"] for _, d in self.doors.items() if not d["locked"]
        )
        initial_active_switches = frozenset(
            sw["id"] for sw in self.switches.values() if sw.get("initial_state") == "on"
        )
        initial_used_switches = frozenset(
            sw["id"]
            for sw in self.switches.values()
            if sw.get("initial_state") == "on" and sw.get("switch_type") == "one_shot"
        )
        initial_state = ValidatorState(
            agent_pos=self.start,
            carrying_key=None,
            collected_keys=frozenset(),
            active_switches=initial_active_switches,
            used_switches=initial_used_switches,
            open_gates=self._recompute_open_gates(initial_active_switches),
            open_doors=initial_open_doors,
            block_positions=initial_block_pos,
        )

        violations = []
        for distractor in self.spec.distractors:
            relevant_ids = self._distractor_candidate_ids(distractor)
            queue = deque([initial_state])
            visited = {initial_state}
            found_interaction = False
            unsafe = False

            while queue:
                state = queue.popleft()
                for transition in self._successors(state):
                    if transition.next_state not in visited:
                        visited.add(transition.next_state)
                        queue.append(transition.next_state)

                    if not any(
                        self._transition_matches_distractor(transition.action_label, candidate_id)
                        for candidate_id in relevant_ids
                    ):
                        continue

                    found_interaction = True
                    beatable, _, _ = self._find_solution(transition.next_state)
                    if (
                        not beatable
                        and distractor.type == "wrong_color_key"
                        and transition.action_label.startswith("pickup:")
                    ):
                        dropped_state = ValidatorState(
                            agent_pos=transition.next_state.agent_pos,
                            carrying_key=None,
                            collected_keys=transition.next_state.collected_keys,
                            active_switches=transition.next_state.active_switches,
                            used_switches=transition.next_state.used_switches,
                            open_gates=transition.next_state.open_gates,
                            open_doors=transition.next_state.open_doors,
                            block_positions=transition.next_state.block_positions,
                        )
                        beatable, _, _ = self._find_solution(dropped_state)
                    if not beatable:
                        unsafe = True
                        queue.clear()
                        break

                if unsafe:
                    break

            if unsafe or not found_interaction:
                violations.append(f"Distractor {distractor.element_id} can break solvability")

        return violations

    def compute_fragility(self, depth_limit: int = 5) -> "FragilityReport":
        """Bounded BFS over abstract transitions to find the shortest breaking sequence."""
        initial_block_pos = frozenset(
            (bid, pos[0], pos[1]) for pos, bid in self.blocks.items()
        )
        initial_open_doors = frozenset(
            d["id"] for _, d in self.doors.items() if not d["locked"]
        )
        initial_active_switches = frozenset(
            sw["id"] for sw in self.switches.values() if sw.get("initial_state") == "on"
        )
        initial_used_switches = frozenset(
            sw["id"]
            for sw in self.switches.values()
            if sw.get("initial_state") == "on" and sw.get("switch_type") == "one_shot"
        )
        initial_state = ValidatorState(
            agent_pos=self.start,
            carrying_key=None,
            collected_keys=frozenset(),
            active_switches=initial_active_switches,
            used_switches=initial_used_switches,
            open_gates=self._recompute_open_gates(initial_active_switches),
            open_doors=initial_open_doors,
            block_positions=initial_block_pos,
        )

        queue = deque([(initial_state, [])])
        visited: dict[ValidatorState, int] = {initial_state: 0}
        breaking_sequences: list[list[str]] = []
        min_steps_to_break = None

        while queue:
            state, sequence = queue.popleft()
            if min_steps_to_break is not None and len(sequence) >= min_steps_to_break:
                continue
            if len(sequence) >= depth_limit:
                continue

            for transition in self._successors(state):
                next_sequence = list(sequence)
                if self._is_irreversible_transition(state, transition):
                    next_sequence = sequence + [transition.action_label]
                next_irrev = len(next_sequence)
                if next_irrev > depth_limit:
                    continue
                if transition.next_state in visited and visited[transition.next_state] <= next_irrev:
                    continue
                visited[transition.next_state] = next_irrev

                beatable, _, _ = self._find_solution(transition.next_state)
                if not beatable and self._is_irreversible_transition(state, transition):
                    min_steps_to_break = len(next_sequence) if min_steps_to_break is None else min(min_steps_to_break, len(next_sequence))
                    if len(next_sequence) == min_steps_to_break:
                        breaking_sequences.append(next_sequence)
                    continue

                queue.append((transition.next_state, next_sequence))

        if min_steps_to_break is None:
            return FragilityReport(
                min_steps_to_break=-1,
                breaking_sequences=[],
                is_fragile=False,
            )

        return FragilityReport(
            min_steps_to_break=min_steps_to_break,
            breaking_sequences=breaking_sequences[:depth_limit],
            is_fragile=min_steps_to_break <= 3,
        )

    def _transition_matches_distractor(self, action_label: str, element_id: str) -> bool:
        """Check whether an action label interacted with a distractor element."""
        if action_label.startswith(("pickup:", "toggle:", "open_door:")):
            return action_label.split(":", 1)[1] == element_id
        if action_label.startswith("push:"):
            parts = action_label.split(":")
            return len(parts) >= 2 and parts[1] == element_id
        if action_label.startswith("teleport:"):
            return element_id in action_label
        return False

    def _distractor_candidate_ids(self, distractor) -> list[str]:
        """Map a distractor annotation to concrete mechanism ids."""
        if any(
            distractor.element_id == obj.id
            for group in (
                self.spec.mechanisms.keys,
                self.spec.mechanisms.doors,
                self.spec.mechanisms.switches,
                self.spec.mechanisms.gates,
                self.spec.mechanisms.blocks,
                self.spec.mechanisms.teleporters,
                self.spec.mechanisms.hazards,
            )
            for obj in group
        ):
            return [distractor.element_id]

        if distractor.type == "distractor_chain":
            critical_ids = set()
            if self.spec.dependency_chain is not None:
                for step in self.spec.dependency_chain.sequence:
                    critical_ids.add(step.element)
                    critical_ids.add(step.unlocks)
            candidate_ids = [
                obj.id
                for group in (
                    self.spec.mechanisms.keys,
                    self.spec.mechanisms.doors,
                    self.spec.mechanisms.switches,
                    self.spec.mechanisms.gates,
                )
                for obj in group
                if obj.id not in critical_ids
            ]
            return candidate_ids or [distractor.element_id]

        return [distractor.element_id]

    def _is_irreversible_transition(self, state: ValidatorState, transition: SuccessorTransition) -> bool:
        """Approximate whether a transition is meaningfully irreversible."""
        label = transition.action_label
        if label.startswith("push:"):
            return True
        if label.startswith("open_door:") and self.key_consumption:
            return True
        if label.startswith("toggle:"):
            switch_id = label.split(":", 1)[1]
            switch_info = self.switches_by_id.get(switch_id, {})
            return switch_info.get("switch_type") == "one_shot"
        if label.startswith("teleport:"):
            return True
        return False


@dataclass
class FragilityReport:
    """Minimum wrong-step analysis for a task."""
    min_steps_to_break: int
    breaking_sequences: list[list[str]]
    is_fragile: bool

    def to_dict(self) -> dict:
        return {
            "min_steps_to_break": self.min_steps_to_break,
            "breaking_sequences": self.breaking_sequences,
            "is_fragile": self.is_fragile,
        }


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
    optimal_path: list[tuple[int, int]]
    backtrack_count: int
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
            "optimal_path": [list(pos) for pos in self.optimal_path],
            "backtrack_count": self.backtrack_count,
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
    seen = set()
    backtrack_count = 0
    for pos in solution or []:
        if pos in seen:
            backtrack_count += 1
        seen.add(pos)

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

    # Prefer explicit dependency chain metadata when present.
    depth = spec.dependency_chain.depth if spec.dependency_chain is not None else 0
    if depth == 0:
        if doors_count > 0 and keys_count > 0:
            depth = max(depth, 1)
        if gates_count > 0 and switches_count > 0:
            depth = max(depth, 1)
        if doors_count > 0 and keys_count > 0 and gates_count > 0 and switches_count > 0:
            depth = max(depth, 2)
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
        backtrack_count * 2.0 +         # path revisits
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
        optimal_path=solution or [],
        backtrack_count=backtrack_count,
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
