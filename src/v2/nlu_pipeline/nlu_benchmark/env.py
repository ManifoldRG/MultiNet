from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional

Pos = Tuple[int, int]

@dataclass
class GridState:
    rows: int
    cols: int
    walls: Set[Pos]
    start: Pos
    goal: Pos
    agent_pos: Pos
    step_count: int = 0
    max_steps: int = 50
    keys: List[Dict[str, Any]] = field(default_factory=list)
    doors: List[Dict[str, Any]] = field(default_factory=list)
    switches: List[Dict[str, Any]] = field(default_factory=list)
    gates: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class StepEvent:
    type: str   # "MOVED", "WALL", "OOB", "DONE"
    message: str


class GridWorldEnv:
    ACTION_DELTAS = {
        "MOVE_NORTH": (-1, 0),
        "MOVE_SOUTH": (1, 0),
        "MOVE_EAST": (0, 1),
        "MOVE_WEST": (0, -1),
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        walls: Set[Pos],
        start: Pos,
        goal: Pos,
        max_steps: int = 50,
        mechanisms: Optional[Dict[str, Any]] = None,
    ):
        mechs = mechanisms or {}
        self.initial = GridState(
            rows=rows,
            cols=cols,
            walls=walls,
            start=start,
            goal=goal,
            agent_pos=start,
            max_steps=max_steps,
            keys=mechs.get("keys", []),
            doors=mechs.get("doors", []),
            switches=mechs.get("switches", []),
            gates=mechs.get("gates", []),
        )
        self.state: Optional[GridState] = None

    def reset(self) -> GridState:
        s = self.initial
        self.state = GridState(
            rows=s.rows,
            cols=s.cols,
            walls=set(s.walls),
            start=s.start,
            goal=s.goal,
            agent_pos=s.start,
            step_count=0,
            max_steps=s.max_steps,
            keys=list(s.keys),
            doors=list(s.doors),
            switches=list(s.switches),
            gates=list(s.gates),
        )
        return self.state

    def step(self, action: str) -> tuple[GridState, StepEvent]:
        assert self.state is not None, "Call reset() first."
        if action not in self.ACTION_DELTAS:
            return self.state, StepEvent("INVALID", f"Unknown action: {action}")

        dr, dc = self.ACTION_DELTAS[action]
        r, c = self.state.agent_pos
        nr, nc = r + dr, c + dc

        if nr < 1 or nr > self.state.rows or nc < 1 or nc > self.state.cols:
            return self.state, StepEvent("OOB", f"{action} would move out of bounds.")
        if (nr, nc) in self.state.walls:
            return self.state, StepEvent("WALL", f"{action} is blocked by a wall at ({nr},{nc}).")

        self.state.agent_pos = (nr, nc)
        self.state.step_count += 1

        if self.state.agent_pos == self.state.goal:
            return self.state, StepEvent("DONE", f"Reached goal at {self.state.goal}.")
        return self.state, StepEvent("MOVED", f"Moved to {self.state.agent_pos}.")