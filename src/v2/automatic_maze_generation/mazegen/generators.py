from __future__ import annotations

from typing import Iterable, List, Set, Tuple
from collections import deque

from .models import (
    Backbone,
    Coord,
    MazeGenSpec,
    MazeLayout,
    DenseMazeParams,
    MultiRouteParams,
    SequentialChainParams,
    SideVaultParams,
    WindingCorridorParams,
)


def in_bounds(c: Coord, width: int, height: int) -> bool:
    x, y = c
    return 0 <= x < width and 0 <= y < height


def neighbors4(c: Coord) -> List[Coord]:
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def carve_cells(cells: Iterable[Coord], open_cells: Set[Coord], width: int, height: int, corridor_width: int = 1) -> None:
    for x, y in cells:
        for dx in range(corridor_width):
            for dy in range(corridor_width):
                cc = (x + dx, y + dy)
                if in_bounds(cc, width, height):
                    open_cells.add(cc)


def build_walls_from_open(width: int, height: int, open_cells: Set[Coord]) -> Set[Coord]:
    return {(x, y) for x in range(width) for y in range(height) if (x, y) not in open_cells}


def path_from_points(points: List[Coord]) -> List[Coord]:
    out: List[Coord] = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        out.append((x1, y1))
        if x1 == x2:
            step = 1 if y2 >= y1 else -1
            for y in range(y1 + step, y2 + step, step):
                out.append((x1, y))
        elif y1 == y2:
            step = 1 if x2 >= x1 else -1
            for x in range(x1 + step, x2 + step, step):
                out.append((x, y1))
        else:
            raise ValueError("Consecutive points must align horizontally or vertically")
    if points:
        out.append(points[-1])
    dedup: List[Coord] = []
    seen: Set[Coord] = set()
    for p in out:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
        seen.add(p)
    return dedup

def generate_winding_corridor(spec: MazeGenSpec) -> MazeLayout:
    assert spec.backbone == Backbone.WINDING_CORRIDOR
    rng = spec.rng()
    p: WindingCorridorParams = spec.backbone_params
    width, height = spec.grid_width, spec.grid_height

    x_min, x_max = 1, max(1, width - 2)
    y_min, y_max = 1, max(1, height - 2)

    current = (x_min, y_min)
    points = [current]
    horizontal = True

    for i in range(p.turn_count + 1):
        seg_len = rng.randint(p.segment_min_length, p.segment_max_length)
        x, y = current

        if horizontal:
            target_x = min(x_max, x + seg_len) if i % 2 == 0 else max(x_min, x - seg_len)
            if target_x == x:
                target_x = min(x_max, x + seg_len)
            current = (target_x, y)
        else:
            target_y = min(y_max, y + seg_len) if (i // 2) % 2 == 0 else max(y_min, y - seg_len)
            if target_y == y:
                target_y = min(y_max, y + seg_len)
            current = (x, target_y)

        points.append(current)
        horizontal = not horizontal

    path = path_from_points(points)
    open_cells: Set[Coord] = set()
    carve_cells(path, open_cells, width, height, corridor_width=p.corridor_width)

    if p.allow_side_stubs:
        candidates = path[1:-1]
        rng.shuffle(candidates)
        stubs_added = 0
        for cell in candidates:
            if stubs_added >= p.side_stub_count:
                break
            dirs = neighbors4(cell)
            rng.shuffle(dirs)
            for nb in dirs:
                if in_bounds(nb, width, height) and nb not in open_cells:
                    open_cells.add(nb)
                    stubs_added += 1
                    break

    start = path[0]
    goal = path[-1]
    walls = build_walls_from_open(width, height, open_cells)

    # --- new: expose mechanism slots on the forced path ---
    pickup_idx = max(1, len(path) // 3)
    blocker_idx = min(len(path) - 2, (2 * len(path)) // 3)

    # keep them away from start/goal and distinct
    if blocker_idx <= pickup_idx:
        blocker_idx = min(len(path) - 2, pickup_idx + 2)

    pickup_cell = path[pickup_idx]
    blocker_cell = path[blocker_idx]

    return MazeLayout(
        width=width,
        height=height,
        walls=walls,
        start=start,
        goal=goal,
        slots={
            "pickup_1_candidates": [pickup_cell],
            "blocker_1_candidates": [blocker_cell],
            "distractor_branch_candidates": [],
        },
        route_cells=[set(path)],
        metadata={
            "backbone": spec.backbone.value,
            "logic_chain": spec.logic_chain.value,
            "turn_count": p.turn_count,
        },
    )

def _route_template_cells(width: int, height: int, num_routes: int) -> Tuple[Coord, Coord, List[List[Coord]]]:
    start = (1, height // 2)
    goal = (width - 2, height // 2)
    rows: List[int] = []
    if num_routes == 2:
        rows = [1, height - 2]
    elif num_routes == 3:
        rows = [1, height // 2, height - 2]
    else:
        rows = [1 + i * max(1, (height - 3) // max(1, num_routes - 1)) for i in range(num_routes)]
        rows = [max(1, min(height - 2, r)) for r in rows]

    routes: List[List[Coord]] = []
    for r in rows[:num_routes]:
        points = [start, (2, start[1]), (2, r), (width - 3, r), (width - 3, goal[1]), goal]
        routes.append(path_from_points(points))
    return start, goal, routes


def generate_multi_route(spec: MazeGenSpec) -> MazeLayout:
    assert spec.backbone == Backbone.MULTI_ROUTE
    p: MultiRouteParams = spec.backbone_params
    width, height = spec.grid_width, spec.grid_height
    start, goal, routes = _route_template_cells(width, height, p.num_routes)
    open_cells: Set[Coord] = set()
    route_sets: List[Set[Coord]] = []
    for route in routes:
        carve_cells(route, open_cells, width, height, corridor_width=p.main_corridor_width)
        route_sets.append(set(route))

    walls = build_walls_from_open(width, height, open_cells)
    return MazeLayout(
        width=width,
        height=height,
        walls=walls,
        start=start,
        goal=goal,
        route_cells=route_sets,
        slots={
            "pickup_1_candidates": [c for c in routes[0][2:-2]] if routes else [],
            "blocker_1_candidates": [goal],
            "distractor_branch_candidates": [],
        },
        metadata={
            "backbone": spec.backbone.value,
            "logic_chain": spec.logic_chain.value,
            "num_routes": len(routes),
        },
    )


def generate_side_vault(spec: MazeGenSpec) -> MazeLayout:
    assert spec.backbone == Backbone.SIDE_VAULT
    p: SideVaultParams = spec.backbone_params
    width, height = spec.grid_width, spec.grid_height

    open_cells: Set[Coord] = set()
    main_y = height // 2
    start = (1, main_y)
    goal = (width - 2, main_y)
    main_path = path_from_points([start, (width - 2, main_y)])
    carve_cells(main_path, open_cells, width, height)

    foyer_x = min(width - 4, max(3, width // 3))
    branch_dir = -1 if p.vault_position_mode in {"upper"} else 1
    if p.vault_position_mode == "random":
        branch_dir = -1 if spec.rng().random() < 0.5 else 1
    branch_end_y = max(1, min(height - 2, main_y + branch_dir * p.vault_branch_depth))
    vault_path = path_from_points([(foyer_x, main_y), (foyer_x, branch_end_y), (min(width - 3, foyer_x + 2), branch_end_y)])
    carve_cells(vault_path, open_cells, width, height)

    blocker_x = min(width - 3, max(foyer_x + 2, width - 3 - p.blocker_distance_from_goal))
    walls = build_walls_from_open(width, height, open_cells)
    return MazeLayout(
        width=width,
        height=height,
        walls=walls,
        start=start,
        goal=goal,
        slots={
            "pickup_1_candidates": [vault_path[-1]],
            "blocker_1_candidates": [(blocker_x, main_y)],
            "distractor_branch_candidates": [],
        },
        route_cells=[set(main_path), set(vault_path)],
        metadata={"backbone": spec.backbone.value, "logic_chain": spec.logic_chain.value},
    )


def generate_sequential_chain(spec: MazeGenSpec) -> MazeLayout:
    assert spec.backbone == Backbone.SEQUENTIAL_CHAIN
    p: SequentialChainParams = spec.backbone_params
    width, height = spec.grid_width, spec.grid_height
    open_cells: Set[Coord] = set()

    start = (1, height // 3)
    choke1 = (max(3, width // 3), height // 3)
    zone2_entry = (max(4, width // 3 + 1), 2 * height // 3)
    choke2 = (max(6, 2 * width // 3), 2 * height // 3)
    goal = (width - 2, 2 * height // 3)

    main_points = [start, choke1, (choke1[0], zone2_entry[1]), zone2_entry, choke2, goal]
    main_path = path_from_points(main_points)
    carve_cells(main_path, open_cells, width, height)

    pickup1 = (max(1, choke1[0] - 1), max(1, start[1] - p.pickup1_branch_depth))
    pickup1_path = path_from_points([(choke1[0] - 1, start[1]), (choke1[0] - 1, pickup1[1])])
    carve_cells(pickup1_path, open_cells, width, height)

    pickup2 = (min(width - 2, zone2_entry[0] + p.pickup2_branch_depth), max(1, zone2_entry[1] - 1))
    pickup2_path = path_from_points([zone2_entry, (pickup2[0], zone2_entry[1]), pickup2])
    carve_cells(pickup2_path, open_cells, width, height)

    walls = build_walls_from_open(width, height, open_cells)
    return MazeLayout(
        width=width,
        height=height,
        walls=walls,
        start=start,
        goal=goal,
        slots={
            "pickup_1_candidates": [pickup1_path[-1]],
            "blocker_1_candidates": [choke1],
            "pickup_2_candidates": [pickup2_path[-1]],
            "blocker_2_candidates": [choke2],
            "distractor_branch_candidates": [],
        },
        route_cells=[set(main_path), set(pickup1_path), set(pickup2_path)],
        metadata={"backbone": spec.backbone.value, "logic_chain": spec.logic_chain.value},
    )



def _carve_dense_maze_grid(cell_w: int, cell_h: int, rng) -> tuple[set[Coord], int, int]:
    """
    Return open cells for a classic carved maze on a tile grid of size:
    width = 2*cell_w + 1, height = 2*cell_h + 1
    """
    width = 2 * cell_w + 1
    height = 2 * cell_h + 1

    open_cells: set[Coord] = set()

    # Mark all logical cells as open
    for cx in range(cell_w):
        for cy in range(cell_h):
            open_cells.add((2 * cx + 1, 2 * cy + 1))

    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))

    while stack:
        cx, cy = stack[-1]
        neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cell_w and 0 <= ny < cell_h and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx, dy))

        if not neighbors:
            stack.pop()
            continue

        nx, ny, dx, dy = rng.choice(neighbors)
        # open wall between current cell and next cell
        wall_x = 2 * cx + 1 + dx
        wall_y = 2 * cy + 1 + dy
        open_cells.add((wall_x, wall_y))

        visited.add((nx, ny))
        stack.append((nx, ny))

    return open_cells, width, height


def _add_dense_maze_loops(open_cells: set[Coord], width: int, height: int, rng, loop_count: int) -> None:
    candidates = []
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if (x, y) in open_cells:
                continue
            # candidate interior wall between two open cells
            horiz = (x - 1, y) in open_cells and (x + 1, y) in open_cells
            vert = (x, y - 1) in open_cells and (x, y + 1) in open_cells
            if horiz or vert:
                candidates.append((x, y))

    rng.shuffle(candidates)
    for c in candidates[:loop_count]:
        open_cells.add(c)


def _shortest_path_on_open_cells(start: Coord, goal: Coord, open_cells: set[Coord], width: int, height: int) -> list[Coord]:
    q = deque([start])
    parent = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in neighbors4(cur):
            if not in_bounds(nb, width, height):
                continue
            if nb not in open_cells or nb in parent:
                continue
            parent[nb] = cur
            q.append(nb)

    if goal not in parent:
        return []

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path



def _pick_path_cell_by_progress(path: list[Coord], lo: float, hi: float, rng) -> Coord:
    if len(path) < 3:
        raise ValueError("Path too short to sample progress-based slot")

    start_idx = max(1, int(lo * (len(path) - 1)))
    end_idx = min(len(path) - 2, int(hi * (len(path) - 1)))
    if end_idx < start_idx:
        end_idx = start_idx
    idx = rng.randint(start_idx, end_idx)
    return path[idx]



def generate_dense_maze(spec: MazeGenSpec) -> MazeLayout:
    assert spec.backbone == Backbone.DENSE_MAZE
    rng = spec.rng()
    p: DenseMazeParams = spec.backbone_params

    open_cells, width, height = _carve_dense_maze_grid(
        p.maze_width_cells,
        p.maze_height_cells,
        rng,
    )

    if p.add_loops and p.loop_count > 0:
        _add_dense_maze_loops(open_cells, width, height, rng, p.loop_count)

    # pick start/goal from open odd cells, far apart
    candidates = sorted(open_cells)
    best_pair = None
    best_dist = -1
    for a in candidates:
        for b in candidates:
            d = manhattan(a, b)
            if d > best_dist:
                best_dist = d
                best_pair = (a, b)

    if best_pair is None:
        raise ValueError("Could not find start/goal in dense maze")

    start, goal = best_pair
    path = _shortest_path_on_open_cells(start, goal, open_cells, width, height)
    if not path:
        raise ValueError("Dense maze path generation failed")

    pickup1 = _pick_path_cell_by_progress(path, p.pickup1_progress_min, p.pickup1_progress_max, rng)
    blocker1 = _pick_path_cell_by_progress(path, p.blocker1_progress_min, p.blocker1_progress_max, rng)
    pickup2 = _pick_path_cell_by_progress(path, p.pickup2_progress_min, p.pickup2_progress_max, rng)
    blocker2 = _pick_path_cell_by_progress(path, p.blocker2_progress_min, p.blocker2_progress_max, rng)

    # enforce monotonic order along the path
    idx = {cell: i for i, cell in enumerate(path)}
    ordered = sorted([pickup1, blocker1, pickup2, blocker2], key=lambda c: idx[c])
    pickup1, blocker1, pickup2, blocker2 = ordered

    # ensure all 4 are distinct and separated
    dedup = []
    for cell in [pickup1, blocker1, pickup2, blocker2]:
        if cell not in dedup:
            dedup.append(cell)

    if len(dedup) < 4:
        # simple fallback using spaced path indices
        n = len(path)
        pickup1 = path[max(1, n // 5)]
        blocker1 = path[max(2, (2 * n) // 5)]
        pickup2 = path[max(3, (3 * n) // 5)]
        blocker2 = path[max(4, (4 * n) // 5)]

    walls = build_walls_from_open(width, height, open_cells)

    return MazeLayout(
        width=width,
        height=height,
        walls=walls,
        start=start,
        goal=goal,
        slots={
            "pickup_1_candidates": [pickup1],
            "blocker_1_candidates": [blocker1],
            "pickup_2_candidates": [pickup2],
            "blocker_2_candidates": [blocker2],
            "distractor_branch_candidates": [],
        },
        route_cells=[set(path)],
        metadata={
            "backbone": spec.backbone.value,
            "logic_chain": spec.logic_chain.value,
            "dense_maze_cells": [p.maze_width_cells, p.maze_height_cells],
            "solution_path_length": len(path) - 1,
        },
    )


def generate_from_spec(spec: MazeGenSpec) -> MazeLayout:
    if spec.backbone == Backbone.WINDING_CORRIDOR:
        return generate_winding_corridor(spec)
    if spec.backbone == Backbone.MULTI_ROUTE:
        return generate_multi_route(spec)
    if spec.backbone == Backbone.SIDE_VAULT:
        return generate_side_vault(spec)
    if spec.backbone == Backbone.SEQUENTIAL_CHAIN:
        return generate_sequential_chain(spec)
    if spec.backbone == Backbone.DENSE_MAZE:
        return generate_dense_maze(spec)
    raise ValueError(f"Unsupported backbone: {spec.backbone}")

