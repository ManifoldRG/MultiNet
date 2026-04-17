from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]


class Backbone(str, Enum):
    WINDING_CORRIDOR = "winding_corridor"
    MULTI_ROUTE = "multi_route"
    SIDE_VAULT = "side_vault"
    SEQUENTIAL_CHAIN = "sequential_chain"
    DENSE_MAZE = "dense_maze"


class LogicChain(str, Enum):
    NONE = "none"
    KD = "kd"
    SG = "sg"
    KS = "ks"
    SK = "sk"
    KK = "kk"


class DistractorMode(str, Enum):
    NONE = "none"
    WRONG_KEYS = "wrong_keys"
    WRONG_SWITCHES = "wrong_switches"
    DEAD_END_ROOMS = "dead_end_rooms"
    DISTRACTOR_CHAIN = "distractor_chain"


@dataclass
class WindingCorridorParams:
    corridor_length: int = 20
    turn_count: int = 4
    segment_min_length: int = 2
    segment_max_length: int = 5
    corridor_width: int = 1
    allow_side_stubs: bool = False
    side_stub_count: int = 0
    start_goal_at_ends: bool = True
    self_proximity_budget: int = 0


@dataclass
class MultiRouteParams:
    num_routes: int = 3
    min_route_length: int = 8
    max_route_length: int = 18
    allow_route_rejoin: bool = True
    route_overlap_budget: int = 1
    route_asymmetry: float = 0.5
    dead_end_branch_count: int = 0
    main_corridor_width: int = 1


@dataclass
class SideVaultParams:
    foyer_size: str = "medium"
    vault_branch_depth: int = 4
    vault_branch_turns: int = 1
    main_route_length_before_blocker: int = 8
    blocker_distance_from_goal: int = 2
    vault_position_mode: str = "random"
    mainline_shape: str = "linear"
    allow_small_dead_ends: bool = False


@dataclass
class SequentialChainParams:
    zone1_size: str = "medium"
    zone2_size: str = "medium"
    choke1_orientation: str = "random"
    choke2_orientation: str = "random"
    pickup1_branch_depth: int = 1
    pickup2_branch_depth: int = 2
    zone2_internal_branches: int = 0
    main_progress_shape: str = "linear"
    allow_local_dead_ends: bool = False



@dataclass
class DenseMazeParams:
    maze_width_cells: int = 7
    maze_height_cells: int = 7
    add_loops: bool = False
    loop_count: int = 0
    pickup1_progress_min: float = 0.20
    pickup1_progress_max: float = 0.40
    blocker1_progress_min: float = 0.45
    blocker1_progress_max: float = 0.65
    pickup2_progress_min: float = 0.60
    pickup2_progress_max: float = 0.80
    blocker2_progress_min: float = 0.80
    blocker2_progress_max: float = 0.92


@dataclass
class ValidationParams:
    require_solvable: bool = True
    require_no_bypass: bool = True
    require_chain_order: bool = True
    require_prerequisite_before_blocker: bool = True
    require_single_main_path: bool = False
    require_unique_shortest_path: bool = False
    min_distinct_solution_routes: int = 1


@dataclass
class MazeGenSpec:
    backbone: Backbone
    logic_chain: LogicChain
    difficulty_tier: int
    grid_width: int
    grid_height: int
    seed: int
    distractor_mode: DistractorMode = DistractorMode.NONE
    max_distractors: int = 0
    backbone_params: object = None
    validation_params: ValidationParams = field(default_factory=ValidationParams)

    def rng(self):
        import random
        return random.Random(self.seed)


@dataclass
class Key:
    id: str
    position: Coord
    color: str


@dataclass
class Door:
    id: str
    position: Coord
    requires_key: str
    initial_state: str = "locked"


@dataclass
class Switch:
    id: str
    position: Coord
    controls: List[str]
    switch_type: str = "toggle"
    initial_state: str = "off"


@dataclass
class Gate:
    id: str
    position: Coord
    initial_state: str = "closed"


@dataclass
class MazeLayout:
    width: int
    height: int
    walls: Set[Coord]
    start: Coord
    goal: Coord
    slots: Dict[str, List[Coord]] = field(default_factory=dict)
    route_cells: List[Set[Coord]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class MazeInstance:
    width: int
    height: int
    walls: Set[Coord]
    start: Coord
    goal: Coord
    keys: List[Key] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    switches: List[Switch] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_json_like(self) -> dict:
        return {
            "maze": {
                "dimensions": [self.width, self.height],
                "walls": sorted([list(w) for w in self.walls]),
                "start": list(self.start),
                "goal": list(self.goal),
            },
            "mechanisms": {
                "keys": [k.__dict__ | {"position": list(k.position)} for k in self.keys],
                "doors": [d.__dict__ | {"position": list(d.position)} for d in self.doors],
                "switches": [s.__dict__ | {"position": list(s.position)} for s in self.switches],
                "gates": [g.__dict__ | {"position": list(g.position)} for g in self.gates],
            },
            "metadata": self.metadata,
        }