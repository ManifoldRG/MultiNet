# Gridworld Domain: Backend Reference

This document describes the two gridworld backends available in MultiNet v1.1 for VLM/VLA evaluation on navigation and puzzle-solving tasks.

## Overview

The gridworld domain provides configurable puzzle environments where an agent must navigate, manipulate objects, and achieve goals. Two backend implementations are available:

| Backend | Based On | Best For |
|---------|----------|----------|
| **MiniGridBackend** | gymnasium `minigrid` package | Standard square grid tasks, mature/tested |
| **MultiGridBackend** | Custom implementation | Exotic tilings (hex, triangle), zones, teleporters |

Both backends implement the same `AbstractGridBackend` interface, allowing seamless swapping for evaluation.

---

## MiniGridBackend

### Description

Wraps the gymnasium `minigrid` package (v3.0+), providing access to a mature, well-tested gridworld implementation. Recommended for standard square-grid puzzles.

### Installation

```bash
pip install minigrid gymnasium
```

### Usage

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

# Load task specification
spec = TaskSpecification.from_json("tasks/tier2/single_key_001.json")

# Create and configure backend
backend = MiniGridBackend(render_mode="rgb_array")
backend.configure(spec)

# Run episode
obs, state, info = backend.reset(seed=42)

for step in range(spec.max_steps):
    action = policy(obs)  # Your policy here
    obs, reward, terminated, truncated, state, info = backend.step(action)

    if terminated or truncated:
        break

backend.close()
```

### Supported Features

| Feature | Support | Notes |
|---------|---------|-------|
| **Tilings** | | |
| Square grid | ✓ | Standard 4-connected grid |
| Hexagonal grid | ✗ | Not supported |
| Triangle grid | ✗ | Not supported |
| **Objects** | | |
| Walls | ✓ | Impassable barriers |
| Keys | ✓ | Colored, unlock matching doors |
| Doors | ✓ | Locked/unlocked, colored |
| Switches | ✓ | Via custom implementation |
| Gates | ✓ | Via custom implementation |
| Blocks (pushable) | ✓ | Can be pushed by agent |
| Hazards (lava) | ✓ | Terminates episode |
| Teleporters | ✗ | Not supported |
| Zones | ✗ | Not supported |
| **Features** | | |
| Partial observability | ✓ | Agent sees limited view |
| Full observability | ✓ | Agent sees entire grid |
| Memory tasks | ✓ | Via MiniGrid environments |
| RGB rendering | ✓ | High-quality sprites |

### Action Space

7 discrete actions (MiniGrid standard):

| ID | Action | Description |
|----|--------|-------------|
| 0 | `turn_left` | Rotate 90° counter-clockwise |
| 1 | `turn_right` | Rotate 90° clockwise |
| 2 | `forward` | Move one cell in facing direction |
| 3 | `pickup` | Pick up object in front |
| 4 | `drop` | Drop held object |
| 5 | `toggle` | Interact (open door, press switch) |
| 6 | `done` | No-op / signal completion |

### Rendering

- Default observation: 64x64 RGB (configurable)
- High-res render: Sprite-based, visually detailed
- Partial observability: Shows only visible cells

### Limitations

- Square grids only
- No zone/target area objects
- No teleporter mechanics
- Tied to MiniGrid's object set

---

## MultiGridBackend

### Description

Custom implementation supporting arbitrary grid topologies (square, hexagonal, triangle) with an extended object set. Built on a topology-agnostic adjacency graph that generalizes to any tiling pattern.

### Usage

```python
from gridworld.backends import MultiGridBackend
from gridworld.task_spec import TaskSpecification

# Load task specification
spec = TaskSpecification.from_json("tasks/tier2/single_key_001.json")

# Create with exotic tiling
backend = MultiGridBackend(
    tiling="triangle",  # or "square", "hex"
    render_mode="rgb_array"
)
backend.configure(spec)

# Run episode (same interface as MiniGridBackend)
obs, state, info = backend.reset(seed=42)

for step in range(spec.max_steps):
    action = policy(obs)
    obs, reward, terminated, truncated, state, info = backend.step(action)

    if terminated or truncated:
        break

backend.close()
```

### Supported Features

| Feature | Support | Notes |
|---------|---------|-------|
| **Tilings** | | |
| Square grid | ✓ | 4-connected (N/E/S/W) |
| Hexagonal grid | ✓ | 6-connected (pointy-top) |
| Triangle grid | ✓ | 3-connected (within hex subdivision) |
| **Objects** | | |
| Walls | ✓ | Impassable barriers |
| Keys | ✓ | Colored, unlock matching doors |
| Doors | ✓ | Locked/unlocked, colored |
| Switches | ✓ | Toggle/hold/one-shot modes |
| Gates | ✓ | Controlled by switches |
| Blocks (movable) | ✓ | Can be picked up or pushed |
| Hazards | ✓ | Terminates episode (lava, spikes, etc.) |
| Teleporters | ✓ | Linked pairs, cooldown support |
| Zones | ✓ | Target areas (overlappable) |
| **Features** | | |
| Partial observability | ✗ | Planned for future |
| Full observability | ✓ | Agent sees entire grid |
| RGB rendering | ✓ | Vector-based (PIL) |

### Action Space

9 discrete actions (extended from MiniGrid):

| ID | Action | Description |
|----|--------|-------------|
| 0 | `forward` | Move in facing direction |
| 1 | `backward` | Move opposite to facing |
| 2 | `turn_left` | Rotate counter-clockwise |
| 3 | `turn_right` | Rotate clockwise |
| 4 | `pickup` | Pick up object at/in front of agent |
| 5 | `drop` | Drop held object |
| 6 | `toggle` | Interact (unlock door with key, activate switch) |
| 7 | `push` | Push object in facing direction |
| 8 | `wait` | No-op |

**Note:** When using MultiGridBackend through the standard 7-action interface, actions are mapped:
- MiniGrid action 5 (toggle) → MultiGrid TOGGLE
- MiniGrid action 6 (done) → MultiGrid WAIT

### Tiling Types

#### Square Tiling
```
┌───┬───┬───┐
│   │   │   │
├───┼───┼───┤    4 directions: N, E, S, W
│   │ A │   │    Agent can face/move in 4 directions
├───┼───┼───┤
│   │   │   │
└───┴───┴───┘
```

#### Hexagonal Tiling
```
   ╱╲   ╱╲
  ╱  ╲ ╱  ╲
 │    │    │     6 directions: N, NE, SE, S, SW, NW
 │  A │    │     Agent can face/move in 6 directions
  ╲  ╱ ╲  ╱
   ╲╱   ╲╱
```

#### Triangle Tiling
```
    ╱╲
   ╱  ╲
  ╱ A  ╲         3 directions: edge0, edge1, edge2
 ╱──────╲        Agent can face/move in 3 directions
```

Each hexagon is subdivided into 6 triangles, creating a denser navigation graph.

### Object Types

#### Key
```python
{
    "id": "key_blue",
    "type": "key",
    "color": "blue",
    "position": {"x": 0.3, "y": 0.5}
}
```
- Can be picked up with PICKUP action
- Used to unlock doors of matching color via TOGGLE
- Optionally consumed on use (configurable via `rules.key_consumption`)

#### Door
```python
{
    "id": "door_blue",
    "type": "door",
    "color": "blue",
    "position": {"x": 0.5, "y": 0.5},
    "is_locked": true
}
```
- Blocks movement when locked/closed
- TOGGLE with matching key unlocks
- TOGGLE again opens/closes (when unlocked)

#### Switch
```python
{
    "id": "switch_1",
    "type": "switch",
    "color": "yellow",
    "position": {"x": 0.3, "y": 0.3},
    "switch_type": "toggle",  // "toggle", "hold", or "one_shot"
    "controls": ["gate_1", "gate_2"],
    "initial_state": false
}
```
- **toggle**: Each TOGGLE flips state
- **hold**: Active only while agent stands on switch
- **one_shot**: Can only be activated once

#### Gate
```python
{
    "id": "gate_1",
    "type": "gate",
    "color": "yellow",
    "position": {"x": 0.5, "y": 0.5},
    "is_open": false,
    "controlled_by": ["switch_1"],
    "require_all": false  // true = AND logic, false = OR logic
}
```
- Opens/closes based on controlling switch states
- Blocks movement when closed

#### Hazard
```python
{
    "id": "lava_1",
    "type": "hazard",
    "color": "red",
    "position": {"x": 0.7, "y": 0.7},
    "hazard_type": "lava",  // for rendering
    "damage": 1.0
}
```
- Agent can step on hazards
- Terminates episode immediately

#### Teleporter
```python
{
    "id": "tele_1",
    "type": "teleporter",
    "color": "purple",
    "position": {"x": 0.1, "y": 0.1},
    "linked_to": "tele_2",
    "cooldown": 1
}
```
- Comes in linked pairs
- Agent stepping on teleporter is transported to linked destination
- Cooldown prevents immediate re-teleportation

#### Zone
```python
{
    "id": "target_zone",
    "type": "zone",
    "color": "cyan",
    "position": {"x": 0.9, "y": 0.9},
    "radius_hops": 1
}
```
- Overlappable target area
- Useful for goal regions, spawn areas, etc.

#### Movable (Block/Box)
```python
{
    "id": "box_1",
    "type": "movable",
    "color": "green",
    "position": {"x": 0.5, "y": 0.5}
}
```
- Can be picked up (PICKUP) or pushed (PUSH)
- Blocks movement when in cell

#### Wall
```python
{
    "id": "wall_1",
    "type": "wall",
    "color": "grey",
    "position": {"x": 0.5, "y": 0.5}
}
```
- Impassable barrier
- Cannot be picked up or pushed

### Rendering

- Observation: 64x64 RGB (for VLM input)
- High-res render: 640x640 RGB (for visualization)
- Vector-based rendering using PIL
- Distinct visual for each object type

### Coordinate System

MultiGrid uses **canonical coordinates** (0.0 to 1.0) that map to grid cells:

```python
# Canonical (x, y) → Grid cell
position = {"x": 0.3, "y": 0.5}  # 30% across, 50% down

# The tiling converts this to the nearest cell
cell_id = tiling.canonical_to_cell(0.3, 0.5)  # e.g., "sq_2_1"
```

This allows task specifications to be tiling-agnostic.

---

## Task Specification Format

Both backends use the same JSON task specification format:

```json
{
    "task_id": "puzzle_001",
    "version": "1.0",
    "seed": 42,
    "difficulty_tier": 2,
    "description": "Collect the blue key to unlock the door",

    "maze": {
        "dimensions": [8, 8],
        "walls": [
            {"x": 0, "y": 0}, {"x": 0, "y": 1}, ...
        ],
        "start": {"x": 1, "y": 1},
        "goal": {"x": 6, "y": 6}
    },

    "mechanisms": {
        "keys": [
            {"id": "key_blue", "position": {"x": 3, "y": 4}, "color": "blue"}
        ],
        "doors": [
            {"id": "door_blue", "position": {"x": 5, "y": 5},
             "requires_key": "blue", "initial_state": "locked"}
        ],
        "switches": [],
        "gates": [],
        "blocks": [],
        "hazards": [],
        "teleporters": []
    },

    "rules": {
        "key_consumption": true,
        "switch_type": "toggle"
    },

    "goal": {
        "type": "reach_position",
        "target": {"x": 6, "y": 6}
    },

    "max_steps": 100
}
```

### Goal Types

| Type | Description | Parameters |
|------|-------------|------------|
| `reach_position` | Agent reaches target cell | `target: {x, y}` |
| `collect_all` | Agent collects all specified items | `target_ids: [...]` |
| `push_block_to` | Push blocks to target positions | `target_ids, target_positions` |
| `survive_steps` | Survive for N steps | `steps: N` |

---

## Choosing a Backend

### Use MiniGridBackend when:
- Working with standard square grids
- Need partial observability
- Want mature, well-tested implementation
- Using existing MiniGrid environments
- Don't need zones or teleporters

### Use MultiGridBackend when:
- Need hexagonal or triangle grids
- Need zone/target area objects
- Need teleporter mechanics
- Want extended action space (backward, push)
- Building custom puzzle types

### Factory Function

```python
from gridworld.backends import get_backend

# Standard square grid
backend = get_backend("minigrid", render_mode="rgb_array")

# Custom with exotic tiling
backend = get_backend("multigrid", tiling="hex", render_mode="rgb_array")
```

---

## GridState

Both backends return a `GridState` object providing backend-agnostic state access:

```python
@dataclass
class GridState:
    agent_position: tuple[int, int]  # Grid coordinates
    agent_direction: int             # 0=right, 1=down, 2=left, 3=up
    agent_carrying: Optional[str]    # ID of held object

    step_count: int
    max_steps: int
    terminated: bool
    truncated: bool
    reward: float

    open_doors: set[str]       # IDs of open doors
    collected_keys: set[str]   # IDs of collected keys
    active_switches: set[str]  # IDs of active switches
    open_gates: set[str]       # IDs of open gates
    block_positions: dict[str, tuple[int, int]]

    goal_reached: bool
```

---

## Difficulty Tiers

Tasks are organized into difficulty tiers:

| Tier | Description | Mechanisms |
|------|-------------|------------|
| 1 | Navigation | Walls only, pathfinding |
| 2 | Linear Dependencies | Key → Door |
| 3 | Multi-Mechanism | Keys + Doors + Switches + Gates |
| 4 | Irreversibility | Pushable blocks, consumable items |
| 5 | Hidden Information | Must infer rules, memory tasks |

---

## Example: Running Evaluation

```python
from gridworld.backends import get_backend
from gridworld.task_spec import TaskSpecification
from gridworld.runner import GridRunner

# Load tasks
tasks = [
    TaskSpecification.from_json(f"tasks/tier{i}/puzzle_{j:03d}.json")
    for i in range(1, 6)
    for j in range(1, 4)
]

# Create runner
runner = GridRunner(backend="minigrid", render_mode="rgb_array")

# Evaluate
results = []
for spec in tasks:
    result = runner.run_episode(spec, policy_fn=your_policy, seed=42)
    results.append({
        "task_id": spec.task_id,
        "success": result.success,
        "steps": result.steps_taken,
        "reward": result.total_reward
    })

# Compute metrics
success_rate = sum(r["success"] for r in results) / len(results)
print(f"Success rate: {success_rate:.2%}")
```

---

## Files Reference

```
src/v1_1/gridworld/
├── __init__.py
├── task_spec.py              # TaskSpecification dataclass
├── task_parser.py            # JSON → environment parser
├── actions.py                # Action space definitions
├── custom_env.py             # CustomMiniGridEnv class
├── backends/
│   ├── __init__.py           # get_backend() factory
│   ├── base.py               # AbstractGridBackend interface
│   ├── minigrid_backend.py   # MiniGrid wrapper
│   └── multigrid_backend.py  # MultiGrid adapter
├── runner/
│   └── grid_runner.py        # Episode execution
├── envs/
│   └── tier_envs.py          # Pre-configured environments
└── tasks/                    # Sample task JSON files
    ├── tier1/
    ├── tier2/
    ├── tier3/
    ├── tier4/
    └── tier5/

src/v1_1/multigrid/
├── __init__.py
├── core.py                   # Cell, TilingGraph
├── base.py                   # Tiling base class
├── tilings.py                # Square, Hex, Triangle tilings
├── agent.py                  # AgentState, Action enum
├── world.py                  # WorldState, execute_action()
├── goals.py                  # Goal predicates
├── rendering.py              # PIL-based rendering
├── env.py                    # MultiGridEnv (gymnasium compatible)
└── objects/
    ├── base.py               # WorldObj, ObjectRegistry
    └── builtin.py            # All object types
```
