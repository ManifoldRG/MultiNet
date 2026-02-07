# Technical Design Document: MultiNet v1.1 GridWorld Framework

## Document Overview

This document provides the technical rationale and architectural decisions behind the MultiNet v1.1 GridWorld evaluation framework. It explains why certain technologies were chosen, how components interact, and the forward-looking vision for cross-domain evaluation.

**Target Audience**: Researchers, contributors, and engineers extending the framework

**Last Updated**: 2026-02-06

---

## Table of Contents

1. [Technology Stack and Justification](#1-technology-stack-and-justification)
2. [Why Non-Square Tilings Matter](#2-why-non-square-tilings-matter)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Cross-Domain Vision](#4-cross-domain-vision-forward-looking)
5. [Evaluation Methodology](#5-evaluation-methodology)

---

## 1. Technology Stack and Justification

### 1.1 Why MiniGrid (Farama Foundation)

**MiniGrid** is the production-ready backend for square grid environments, built on the mature Gymnasium (formerly OpenAI Gym) ecosystem.

#### Technical Advantages

**1. Maturity and Stability**
- Actively maintained by Farama Foundation (successor to OpenAI Gym)
- Used in hundreds of RL research papers since 2017
- Battle-tested codebase with well-understood edge cases
- Stable API with semantic versioning

**2. Rich Feature Set**
- 7-action discrete space: turn_left, turn_right, forward, pickup, drop, toggle, done
- Partial observability: Agent has limited field of view (7x7 grid by default)
- Built-in rendering: High-quality RGB visualizations and human-readable text mode
- Standard observation types: Symbolic (grid encoding) and visual (RGB images)

**3. Community and Ecosystem**
- Large user base provides extensive examples and troubleshooting resources
- Compatible with RL libraries: Stable-Baselines3, RLlib, CleanRL
- Well-documented: Official docs at minigrid.farama.org
- Active community on GitHub and Discord

**4. Performance Characteristics**
```
Operation               Time        Memory
------------------------------------------
Environment creation    ~10 ms      ~50 KB
Episode (100 steps)     ~400 ms     ~200 KB
Observation rendering   ~3 ms       ~150 KB (64x64x3)
```

Fast enough for large-scale evaluation (1000s of episodes).

#### Built-in Mechanisms

MiniGrid natively supports:
- **Keys and Doors**: Collectible keys unlock matching color-coded doors
- **Boxes**: Pushable Sokoban-style blocks
- **Lava**: Episode-ending hazard tiles
- **Walls**: Static barriers for maze construction

We extended MiniGrid with:
- **Switches and Gates**: Remote-controlled barriers
- **Goal Markers**: Explicit visual goal positions
- **Teleporters**: Instant transport between positions (v1.2 planned)

#### Limitations

**1. Square-Only Topology**
- Hardcoded 4-connected grid (N/S/E/W movement)
- Agent direction restricted to 4 cardinal directions
- Cannot represent hexagonal or triangular spatial relationships

**2. Rigid Object System**
- Object types are hardcoded Python classes
- Adding new object types requires modifying core MiniGrid code
- Limited extensibility for custom mechanisms

**3. Rendering Pipeline**
- Tile-based rendering assumes square cells
- Cannot easily render non-square tilings
- Sprite system optimized for 90-degree rotations

**4. Distribution Shift Risk**
- Models trained predominantly on MiniGrid may overfit to square-grid patterns
- Success on MiniGrid doesn't guarantee understanding of spatial reasoning (see Section 2)

#### When to Use MiniGrid

**Recommended for:**
- Production evaluation of agents on standard gridworld tasks
- Tasks requiring partial observability and memory
- Benchmarking against existing MiniGrid baselines
- Maximum performance and stability requirements

**Not suitable for:**
- Testing topology invariance
- Exotic tiling research (hex, triangle, Penrose, etc.)
- Tasks requiring novel object types not in MiniGrid

---

### 1.2 Why MultiGrid (Custom Implementation)

**MultiGrid** is an experimental backend designed for research on exotic grid tilings and spatial topology invariance.

#### Core Innovation: Adjacency Graph Architecture

Unlike MiniGrid's hardcoded coordinate system, MultiGrid represents grids as **adjacency graphs**:

```python
# Square tiling: Cell has 4 neighbors
cell_neighbors = {
    "N": cell_id + width,
    "E": cell_id + 1,
    "S": cell_id - width,
    "W": cell_id - 1
}

# Hexagonal tiling: Cell has 6 neighbors
cell_neighbors = {
    "N": ...,   "NE": ...,
    "SE": ...,  "S": ...,
    "SW": ...,  "NW": ...
}

# Triangular tiling: Cell has 3 or 9 neighbors (depends on orientation)
cell_neighbors = {
    "APEX_UP": [...],    # Upward-pointing triangle
    "APEX_DOWN": [...]   # Downward-pointing triangle
}
```

This abstraction enables **tiling-agnostic algorithms**. The same pathfinding or agent logic works on any tiling without code changes.

#### Key Technical Features

**1. Normalized Coordinate System**

All positions are stored in normalized [0,1] × [0,1] space:

```python
# Grid coordinate (3, 5) in 8×8 grid
normalized_pos = (3/8, 5/8) = (0.375, 0.625)
```

**Why normalize?**
- **Cross-tiling compatibility**: Same task specification works on square, hex, and triangle grids
- **Resolution independence**: Tasks scale to different grid sizes without rewriting coordinates
- **Domain transfer**: Same normalized coordinates can map to other domains (see Section 4)

**2. Extensible Object Registry**

MultiGrid uses a registry pattern for objects:

```python
class ObjectRegistry:
    _types = {
        "movable": MovableObject,
        "wall": WallObject,
        "zone": ZoneObject,
        "teleporter": TeleporterObject
    }
```

Adding new object types doesn't require modifying core environment code.

**3. Goal Specification System**

Rich goal types beyond "reach position":

```python
goals = {
    "reach_position": {"target": (0.5, 0.5)},
    "collect_all": {"target_ids": ["key1", "key2"]},
    "push_block_to": {"block_id": "block1", "target": (0.7, 0.7)},
    "survive_steps": {"min_steps": 100},
    "zone_occupation": {"zone_id": "goal_zone", "duration": 10}
}
```

#### Technical Tradeoffs

**Advantages:**
- Arbitrary tilings without code changes
- Normalized coordinates enable cross-domain transfer
- Extensible object and goal systems
- Research-friendly architecture

**Disadvantages:**
- Immature: Fewer users, less tested
- Slower: ~600-900ms per episode (vs 400ms for MiniGrid)
- Incomplete: Switches/gates not yet implemented
- No partial observability yet
- Rendering quality variable for exotic tilings

#### Performance Overhead

```
Operation               MiniGrid    MultiGrid   Overhead
----------------------------------------------------------
Configure task          ~0.1 ms     ~5 ms       50x
Reset environment       ~10 ms      ~15 ms      1.5x
Step execution          ~3 ms       ~5 ms       1.67x
Render                  ~4 ms       ~8 ms       2x
----------------------------------------------------------
100-step episode        ~400 ms     ~600 ms     1.5x
```

**Bottlenecks:**
1. Cell ID ↔ normalized coordinate conversions (happens every step)
2. Neighbor computation for non-square tilings (hexagons have 6 neighbors vs 4 for squares)
3. Rendering complex polygon shapes (triangles, hexagons)

**Optimization opportunities:**
- Cache coordinate conversions
- Precompute neighbor maps
- Vectorize rendering operations

#### When to Use MultiGrid

**Recommended for:**
- Research on topology invariance and spatial reasoning
- Testing agent generalization across grid types
- Exploring novel spatial representations
- Prototyping new object types and mechanisms

**Not suitable for:**
- Production evaluation (use MiniGrid)
- Large-scale benchmarking (too slow)
- Tasks requiring all mechanisms (switches/gates incomplete)
- Time-critical applications

---

### 1.3 Feature Comparison Matrix

| Feature | MiniGrid | MultiGrid | Notes |
|---------|----------|-----------|-------|
| **Status** | Production | Experimental | MiniGrid is battle-tested |
| **Maturity** | High | Low | MultiGrid needs more testing |
| **Tilings** | Square only | Square/Hex/Triangle | MultiGrid's key innovation |
| **Performance** | ~400ms/episode | ~600-900ms/episode | MiniGrid 1.5-2x faster |
| **Mechanisms** | | | |
| - Keys/Doors | ✓ | Partial | Door unlocking incomplete in MultiGrid |
| - Switches/Gates | ✓ | ✗ | Not yet in MultiGrid |
| - Pushable Blocks | ✓ | ✓ | Both support |
| - Hazards (Lava) | ✓ | ✗ | Not yet in MultiGrid |
| - Teleporters | ✗ | ✓ | MultiGrid native support |
| - Zones | ✗ | ✓ | MultiGrid native support |
| **Partial Obs** | ✓ | ✗ | MultiGrid planned v1.2 |
| **Rendering Quality** | High | Variable | Hex/triangle rendering experimental |
| **Community** | Large | Small | MiniGrid has 8+ years community |
| **Documentation** | Extensive | Limited | MiniGrid has official docs |
| **RL Library Support** | Full | Partial | MiniGrid works with SB3, RLlib |
| **Use Case** | Standard eval | Topology research | Choose based on needs |

---

## 2. Why Non-Square Tilings Matter

### 2.1 The Distribution Shift Hypothesis

**Core Hypothesis**: Models trained predominantly on square-grid environments may develop spatial reasoning heuristics that only work on 4-connected grids. Success on square grids could reflect **interface memorization** rather than genuine spatial understanding.

#### Evidence for Distribution Shift

**1. Prevalence of Square Grids in Training Data**

Modern vision-language-action models are trained on:
- **Atari games**: All use square pixel grids with 4-directional movement
- **GridWorld RL environments**: MiniGrid, DeepMind Lab, Procgen all use square grids
- **Video games**: Vast majority use square tile maps (Minecraft, Pokémon, roguelikes)
- **Robot navigation**: Indoor environments often represented as 2D occupancy grids (square cells)

**2. Shortcut Learning Risk**

Models may learn spurious correlations:
- "Moving right twice is equivalent to moving forward twice if I'm facing east"
- "Obstacles are always at Manhattan distance increments"
- "The world has 4 degrees of rotational symmetry"

These heuristics work perfectly on square grids but fail on hexagonal or triangular topologies.

**3. Generalization Failure Example**

Consider a simple navigation task: "Go from position A to position B while avoiding wall at position C."

On a **square grid** (4 neighbors):
```
A . . B
. W . .
. . . .
```
Optimal path length: 3 steps (right, right, up or similar)

On a **hexagonal grid** (6 neighbors):
```
  A   .   B
    .   W   .
  .   .   .
```
Optimal path length: 2 steps (northeast, east or similar)

If a model memorizes "3 steps is optimal for this distance," it fails on the hex grid.

### 2.2 Hexagonal Grids

**Mathematical Properties:**
- **6-connected**: Each cell has 6 neighbors
- **Equidistant neighbors**: All neighbors are the same distance (unlike squares where diagonals are √2x farther)
- **120° rotational symmetry**: Natural for systems with 3-fold or 6-fold symmetry
- **Optimal packing**: Hexagons tile the plane with minimal perimeter for given area

**Real-World Applications:**
- **Strategy games**: Civilization, Catan, Axis & Allies
- **Nature**: Honeycombs, crystal structures, turtle shells
- **Geographic grids**: Some GIS systems use hexagonal cells for regional analysis
- **Path planning**: Hexagonal grids provide smoother diagonal movement

**What Hexagonal Grids Test:**

1. **Direction Concept vs Pattern Matching**
   - Square grid agent might memorize "turn_right = direction + 1 mod 4"
   - Hex grid requires "turn_right = direction + 1 mod 6"
   - Tests whether model understands angular rotation vs memorizes turning mechanics

2. **Distance Computation**
   - Square grids: Manhattan distance (|x1-x2| + |y1-y2|)
   - Hex grids: Cube coordinate distance (different formula)
   - Tests whether model understands proximity vs memorizes step counting

3. **Adjacency Understanding**
   - Square: 4 neighbors (N/E/S/W)
   - Hex: 6 neighbors (N/NE/SE/S/SW/NW)
   - Tests whether model understands "adjacent cell" as a concept vs memorizes 4-directional offsets

**Example Task: Navigation with Obstacle**

```python
# Task specification (normalized coordinates)
task = {
    "agent_start": (0.2, 0.2),
    "goal": (0.8, 0.8),
    "walls": [(0.5, 0.5), (0.5, 0.6)]
}

# Square grid: Agent must go around (6-8 steps)
# Hex grid: Agent can navigate more directly (4-5 steps)
# Model must adapt strategy to topology
```

### 2.3 Triangular Grids

**Mathematical Properties:**
- **3-connected**: Each triangle has 3 edge-adjacent neighbors
- **Variable connectivity**: 9-connected if considering vertex neighbors
- **Minimal connectivity**: Forces longer paths and deeper planning
- **Two orientations**: Upward-pointing (Δ) and downward-pointing (▽) triangles

**What Triangular Grids Test:**

1. **Planning Depth**
   - Fewer neighbors per cell means longer paths
   - Tests whether model can plan ahead multiple steps
   - Exposes greedy policies that don't work with 3-way branching

2. **Orientation Handling**
   - Triangles have different adjacency depending on orientation (Δ vs ▽)
   - Tests whether model can handle position-dependent navigation rules

3. **Minimal Topology**
   - Simplest non-trivial tiling (3 sides per cell)
   - Cleanest test of "can model navigate non-square grids?"

**Example Task: Forced Long Path**

```python
# Same start and goal as hex example
# Triangle grid: ~7-9 steps (fewer branching options)
# Model must commit to longer plans without greedy shortcuts
```

### 2.4 Archimedean Tilings (Future Work)

**Archimedean tilings** use multiple regular polygons. Example: **3-4-6-4 tiling** (triangle-square-hexagon-square pattern).

**Why This Is The Ultimate Test:**

1. **Heterogeneous Neighborhoods**: Some cells have 3 neighbors, others 4, 6, or 8
2. **No Global Patterns**: Model cannot memorize "every cell has N neighbors"
3. **Position-Dependent Rules**: Navigation strategy must adapt per cell
4. **Maximum Adversarial**: Most different from training distribution

**Example: 4-8-8 Tiling**

```
┌─────┬─────┐
│  □  │  ◯  │  □ = square (4 neighbors)
├─────┼─────┤  ◯ = octagon (8 neighbors)
│  ◯  │  □  │
└─────┴─────┘
```

Model navigating this grid must:
- Detect current cell type (square vs octagon)
- Adjust movement strategy dynamically
- Plan paths considering variable branching factor

### 2.5 Contamination Resistance

**Problem**: Modern VLMs are trained on massive web-scale datasets (LAION-5B, Common Crawl, etc.). If MiniGrid environment screenshots appear in training data, models may memorize task solutions rather than learn spatial reasoning.

**Why Exotic Tilings Help:**

1. **Rarity**: Hexagonal and triangular gridworld environments are uncommon in training data
2. **Novel Visuals**: Rendering style differs from typical game screenshots
3. **Controlled Distribution**: We generate tasks programmatically, ensuring no data leakage
4. **Cleaner Signal**: Performance differences between square and hex grids isolate topology understanding

**Evaluation Strategy:**

```python
# Compare same agent on same task across tilings
results = {
    "square": evaluate(agent, task, tiling="square"),
    "hex": evaluate(agent, task, tiling="hex"),
    "triangle": evaluate(agent, task, tiling="triangle")
}

# Generalization gap = performance drop on exotic tilings
gap = results["square"]["success_rate"] - results["hex"]["success_rate"]

# Ideal: gap ≈ 0 (topology-invariant reasoning)
# Reality: gap > 0 (some overfitting to square grids)
```

---

## 3. Architecture Decisions

### 3.1 Why Adjacency Graphs Over Coordinate Grids

**Traditional Approach (MiniGrid):**

```python
# Hardcoded coordinate arithmetic
def move_forward(agent_pos, agent_dir):
    if agent_dir == 0:  # East
        return (agent_pos[0] + 1, agent_pos[1])
    elif agent_dir == 1:  # South
        return (agent_pos[0], agent_pos[1] + 1)
    # ... hardcoded for 4 directions
```

**Problem**: Cannot generalize to 6-directional (hex) or variable-directional (triangle) grids.

**MultiGrid Approach:**

```python
# Tiling-agnostic adjacency graph
class Tiling(ABC):
    def get_neighbors(self, cell_id: int) -> dict[str, int]:
        """Return mapping of direction names to neighbor cell IDs."""
        pass

# Works for any tiling
def move_forward(agent_cell, agent_dir, tiling):
    neighbors = tiling.get_neighbors(agent_cell)
    return neighbors[agent_dir]  # No hardcoded arithmetic!
```

**Advantages:**

1. **Tiling Independence**: Same code works for square, hex, triangle, Penrose, Voronoi, etc.
2. **Extensibility**: Add new tilings without modifying core logic
3. **Correctness**: Neighbor relationships defined once per tiling, not scattered throughout codebase
4. **Testing**: Each tiling has isolated test suite

**Design Pattern: Strategy Pattern**

```python
# Abstract interface
class Tiling(ABC):
    @abstractmethod
    def generate_grid(self, width, height) -> Graph: pass

    @abstractmethod
    def get_neighbors(self, cell_id) -> dict[str, int]: pass

    @abstractmethod
    def cell_to_canonical(self, cell_id) -> tuple[float, float]: pass

# Concrete implementations
class SquareTiling(Tiling): ...
class HexTiling(Tiling): ...
class TriangleTiling(Tiling): ...

# Usage
tiling = HexTiling()
graph = tiling.generate_grid(8, 8)
neighbors = tiling.get_neighbors(cell_id=42)
```

### 3.2 Why Gymnasium API Compatibility Matters

**Gymnasium** (formerly OpenAI Gym) is the de facto standard for RL environments.

**Standard Interface:**

```python
# All Gymnasium environments implement this
env = gym.make("MiniGrid-DoorKey-8x8-v0")
observation, info = env.reset(seed=42)

done = False
while not done:
    action = agent.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

**Why This Matters:**

1. **RL Library Integration**: Stable-Baselines3, RLlib, CleanRL all expect Gymnasium API
2. **Benchmarking**: Papers can directly compare against Gymnasium baselines
3. **Tooling**: Visualization tools, logging, and monitoring assume Gymnasium
4. **Reproducibility**: Standard API reduces implementation variance between research groups

**MultiGrid Compliance:**

```python
class MultiGridEnv(gym.Env):
    """Fully Gymnasium-compatible environment."""

    def reset(self, seed=None, options=None):
        # Standard return: (observation, info)
        return observation, info

    def step(self, action):
        # Standard return: (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info
```

### 3.3 Why Canonical [0,1] Coordinates for Cross-Domain Transfer

**Problem**: Different domains use different coordinate systems.

**Examples:**

| Domain | Coordinate System | Range |
|--------|-------------------|-------|
| GridWorld | Integer cell indices | [0, width) × [0, height) |
| Physics (MuJoCo) | Continuous world space | (-∞, +∞) × (-∞, +∞) |
| Natural Language | No spatial coordinates | N/A |
| GUI (Pygame) | Pixel coordinates | [0, screen_width) × [0, screen_height) |

**Solution: Normalized Canonical Coordinates**

All positions are represented in [0,1] × [0,1] space:

```python
# Task specification (domain-agnostic)
task = {
    "agent_start": (0.2, 0.2),
    "goal": (0.8, 0.8),
    "obstacles": [(0.5, 0.5)]
}

# GridWorld adapter
def to_grid(pos, grid_size):
    return (int(pos[0] * grid_size[0]), int(pos[1] * grid_size[1]))

# Physics adapter (MuJoCo)
def to_physics(pos, world_bounds):
    x = world_bounds[0] + pos[0] * (world_bounds[1] - world_bounds[0])
    y = world_bounds[2] + pos[1] * (world_bounds[3] - world_bounds[2])
    return (x, y)

# GUI adapter (Pygame)
def to_pixels(pos, screen_size):
    return (int(pos[0] * screen_size[0]), int(pos[1] * screen_size[1]))
```

**Advantages:**

1. **Domain Independence**: Same task definition works across all domains
2. **Resolution Independence**: Tasks scale to different grid/screen sizes
3. **Human Interpretability**: Normalized coordinates are intuitive (0.5 = center)
4. **Transfer Learning**: Agents trained on gridworld can be tested on physics sim with same task

**Precision Considerations:**

```python
# Potential precision loss with integer grids
grid_pos = (3, 5) in 8×8 grid
normalized = (0.375, 0.625)
back_to_grid = (int(0.375 * 8), int(0.625 * 8)) = (3, 5) ✓

# Loss with non-power-of-2 dimensions
grid_pos = (3, 5) in 7×7 grid
normalized = (0.428571, 0.714286)
back_to_grid = (int(0.428571 * 7), int(0.714286 * 7)) = (2, 5) ✗
```

**Recommendation**: Use power-of-2 dimensions (8×8, 16×16) for lossless round-tripping.

### 3.4 Action Space Design

**MiniGrid Standard (7 Actions):**

```python
actions = {
    0: "turn_left",    # Rotate counterclockwise
    1: "turn_right",   # Rotate clockwise
    2: "forward",      # Move in facing direction
    3: "pickup",       # Pick up object in front
    4: "drop",         # Drop held object
    5: "toggle",       # Interact (open door, press switch)
    6: "done"          # Signal completion (no-op)
}
```

**MultiGrid Extension (9 Actions):**

```python
actions = {
    0: "FORWARD",      # Move forward
    1: "BACKWARD",     # Move backward (new!)
    2: "TURN_LEFT",    # Rotate CCW
    3: "TURN_RIGHT",   # Rotate CW
    4: "PICKUP",       # Pick up object
    5: "DROP",         # Drop object
    6: "PUSH",         # Push object forward
    7: "WAIT",         # No-op
    8: "TELEPORT"      # Use teleporter (if on one)
}
```

**Action Translation Layer:**

```python
# Backend automatically translates MiniGrid actions to MultiGrid
minigrid_to_multigrid = {
    0: 2,  # turn_left → TURN_LEFT
    1: 3,  # turn_right → TURN_RIGHT
    2: 0,  # forward → FORWARD
    3: 4,  # pickup → PICKUP
    4: 5,  # drop → DROP
    5: 6,  # toggle → PUSH
    6: 7   # done → WAIT
}
```

**Why Translation Matters:**

1. **Policy Reuse**: Same agent code works on both backends
2. **Comparative Evaluation**: Test same policy on MiniGrid and MultiGrid
3. **Backward Compatibility**: Existing MiniGrid agents work on exotic tilings

**Design Tradeoff: Absolute vs Relative Actions**

```python
# Option A: Absolute actions (not used)
actions = ["move_north", "move_east", "move_south", "move_west"]
# Problem: Doesn't work on hex (6 directions) or triangle (variable)

# Option B: Relative actions (chosen)
actions = ["turn_left", "turn_right", "forward"]
# Benefit: Works on any tiling (just adjust turn angle)
```

Relative actions generalize to arbitrary tilings because they're ego-centric.

### 3.5 File-Based Task Interface

**Design Decision**: Tasks are defined in JSON files, not Python code.

**JSON Task Specification:**

```json
{
    "task_id": "tier2_key_door_001",
    "seed": 42,
    "difficulty_tier": 2,
    "max_steps": 100,
    "maze": {
        "dimensions": [8, 8],
        "start": [1, 1],
        "goal": [6, 6],
        "walls": [[3, 3], [3, 4]]
    },
    "mechanisms": {
        "keys": [{"id": "k1", "position": [2, 2], "color": "red"}],
        "doors": [{"id": "d1", "position": [4, 4], "requires_key": "red"}]
    },
    "goal": {"type": "reach_position", "target": [6, 6]}
}
```

**Advantages:**

1. **Language Agnostic**: Can be used from any language (Python, Julia, Rust, etc.)
2. **Version Control**: Git-friendly plain text format
3. **Human Readable**: Non-programmers can create tasks
4. **Programmatic Generation**: Easy to generate task suites with scripts
5. **Validation**: JSON schema validation catches errors early

**Python ABC for Backends:**

```python
class AbstractGridBackend(ABC):
    @abstractmethod
    def configure(self, task_spec: TaskSpecification): pass

    @abstractmethod
    def reset(self, seed: int) -> tuple[np.ndarray, GridState, dict]: pass

    @abstractmethod
    def step(self, action: int) -> tuple[...]: pass
```

This ensures all backends implement the same interface, regardless of internal implementation.

---

## 4. Cross-Domain Vision (Forward-Looking)

### 4.1 The Four Domains

**Goal**: Same task definition works across four different embodiments.

**Domain 1: GridWorld** (Current Implementation)
- Square/hex/triangle tilings
- Discrete cell-based navigation
- Turn-based action execution
- 2D top-down view

**Domain 2: Physics Simulation** (Planned v1.2)
- MuJoCo or PyBullet physics engine
- Continuous 2D or 3D space
- Continuous control (velocity, force)
- Physical collisions and dynamics

**Domain 3: Natural Language** (Planned v1.3)
- Text-based interactive fiction
- Parser-based commands ("go north", "take key")
- ASCII or text descriptions
- Pure language reasoning

**Domain 4: GUI (Pygame)** (Planned v1.4)
- Visual game interface
- Mouse click and keyboard controls
- Real-time or turn-based
- Rich graphics and animations

### 4.2 Canonical Task Specification as Shared Representation

**Core Idea**: A single JSON task specification gets translated to each domain.

**Example Task: Key-Door Puzzle**

```json
{
    "task_id": "cross_domain_001",
    "agent_start": [0.2, 0.2],
    "goal": [0.8, 0.8],
    "objects": [
        {"type": "key", "id": "k1", "position": [0.3, 0.4], "color": "red"},
        {"type": "door", "id": "d1", "position": [0.5, 0.5], "color": "red"}
    ]
}
```

**Domain Translations:**

**GridWorld:**
```python
# 8×8 grid
agent_start = (1, 1)  # 0.2 * 8 = 1.6 → 1
goal = (6, 6)         # 0.8 * 8 = 6.4 → 6
key_pos = (2, 3)
door_pos = (4, 4)
```

**Physics (MuJoCo):**
```python
# 10m × 10m world
agent_start = (2.0, 2.0)  # 0.2 * 10
goal = (8.0, 8.0)
key = PhysicsBody(position=(3.0, 4.0), shape="cube", color="red")
door = PhysicsWall(position=(5.0, 5.0), color="red", passable=False)
```

**Natural Language:**
```
You are in a small room. To the NORTH, you see a RED KEY.
To the EAST, there is a RED DOOR (locked). The goal is to the NORTHEAST.

> take key
You pick up the red key.

> go east
The door is locked. You need a red key.

> unlock door
You unlock the door with the red key. The door opens.

> go east
You reach the goal!
```

**GUI (Pygame):**
```python
# 800×800 pixel window
agent_sprite = Sprite(position=(160, 160))
goal_sprite = Sprite(position=(640, 640), texture="goal.png")
key_sprite = Sprite(position=(240, 320), texture="key_red.png")
door_sprite = Sprite(position=(400, 400), texture="door_red_locked.png")

# Mouse click to move, click key to pick up, click door to unlock
```

### 4.3 Domain Adapters as Thin Translation Layers

**Architecture:**

```python
# Core task specification (domain-agnostic)
task_spec = TaskSpecification.from_json("task.json")

# Domain adapters
gridworld_env = GridWorldAdapter(task_spec, tiling="square")
physics_env = PhysicsAdapter(task_spec, engine="mujoco")
text_env = TextAdapter(task_spec, style="interactive_fiction")
gui_env = GUIAdapter(task_spec, graphics="pygame")

# Same evaluation code
for env in [gridworld_env, physics_env, text_env, gui_env]:
    obs, state, _ = env.reset(seed=42)
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, state, _ = env.step(action)
        done = terminated or truncated
    print(f"Domain: {env.domain_name}, Success: {state.goal_reached}")
```

**Key Insight**: Adapters are thin. Most logic lives in the canonical task specification and shared utility functions.

### 4.4 Mouse Click Support for Domain 4

**Challenge**: GUI domain uses mouse clicks, not discrete actions.

**Solution: Coordinate-Based Action Interface**

```python
# Standard discrete actions (Domains 1-3)
action = 2  # forward

# Coordinate-based actions (Domain 4)
action = {"type": "click", "position": (0.6, 0.5)}
```

**Backend Handling:**

```python
class GUIAdapter(AbstractGridBackend):
    def step(self, action):
        if isinstance(action, int):
            # Discrete action (keyboard shortcut)
            return self._execute_discrete_action(action)
        elif isinstance(action, dict) and action["type"] == "click":
            # Mouse click action
            pixel_pos = self._normalized_to_pixels(action["position"])
            pygame_event = pygame.event.Event(MOUSEBUTTONDOWN, {"pos": pixel_pos})
            return self._inject_event(pygame_event)
```

**Unified Agent Interface:**

```python
# Agent can use either action type
class Agent:
    def predict(self, obs, domain):
        if domain.supports_discrete_actions:
            return self.policy_discrete(obs)
        else:
            # VLM identifies clickable objects in image
            clickable_objects = self.vlm.detect_objects(obs)
            target = self.policy_select_object(clickable_objects)
            return {"type": "click", "position": target.normalized_position}
```

**Example: Clicking a Key to Pick It Up**

```python
# Domain 1 (GridWorld): Discrete action
action = 3  # pickup

# Domain 4 (GUI): Click on key sprite
key_position_pixels = (240, 320)
key_position_normalized = (240/800, 320/800) = (0.3, 0.4)
action = {"type": "click", "position": (0.3, 0.4)}
```

### 4.5 Cross-Domain Evaluation Strategy

**Research Question**: Do agents learn task-solving strategies or domain-specific interfaces?

**Evaluation Protocol:**

1. **Train** on Domain 1 (GridWorld) with square tiling
2. **Test** on:
   - Domain 1 with hex tiling (topology shift)
   - Domain 2 with physics (embodiment shift)
   - Domain 3 with text (modality shift)
   - Domain 4 with GUI (interface shift)

**Metrics:**

```python
results = {
    "gridworld_square": {"success_rate": 0.85, "avg_steps": 12},
    "gridworld_hex": {"success_rate": 0.60, "avg_steps": 15},
    "physics": {"success_rate": 0.45, "avg_steps": 25},
    "text": {"success_rate": 0.30, "avg_steps": 18},
    "gui": {"success_rate": 0.55, "avg_steps": 20}
}

# Generalization gaps
topology_gap = results["gridworld_square"]["success_rate"] - results["gridworld_hex"]["success_rate"]
embodiment_gap = results["gridworld_square"]["success_rate"] - results["physics"]["success_rate"]
modality_gap = results["gridworld_square"]["success_rate"] - results["text"]["success_rate"]
interface_gap = results["gridworld_square"]["success_rate"] - results["gui"]["success_rate"]
```

**Hypothesis**: Current VLMs will show large generalization gaps, indicating domain overfitting.

---

## 5. Evaluation Methodology

### 5.1 Deterministic Seeds for Reproducibility

**Requirement**: All random operations must use explicit seeds.

**Implementation:**

```python
# Task specification includes seed
task_spec = {
    "task_id": "eval_001",
    "seed": 42,  # Default seed for this task
    ...
}

# Evaluation can override seed
for seed in range(10):
    obs, state, _ = backend.reset(seed=seed)
    # Run episode with this seed
```

**Why This Matters:**

1. **Reproducibility**: Other researchers can replicate exact results
2. **Debugging**: Failed episodes can be replayed with same seed
3. **Fair Comparison**: All models see identical task instances
4. **Statistical Power**: Multiple seeds enable significance testing

**Seeding Strategy:**

```python
# Seed controls:
- Environment randomness (object placement if stochastic)
- Agent policy randomness (if stochastic policy)
- Evaluation noise (if added)

# Example
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
agent.reset_rng(seed)
```

### 5.2 Metrics

**Primary Metrics:**

**1. Success Rate**
```python
success_rate = num_episodes_reached_goal / total_episodes
```

Binary: Did the agent reach the goal within max_steps?

**2. Step Efficiency**
```python
step_efficiency = goal_distance / steps_taken
```

How efficiently did the agent solve the task? Lower is better.

**3. Reward (for RL agents)**
```python
total_reward = sum(rewards_per_step)
```

MiniGrid uses time-penalized reward: `reward = 1.0 - 0.9 * (steps / max_steps)`

**Secondary Metrics:**

**4. Mechanism Usage**
- Keys collected: `len(state.collected_keys)`
- Switches activated: `len(state.active_switches)`
- Doors unlocked: `len(state.open_doors)`

**5. Path Quality**
- Path length vs optimal path
- Backtracking steps (revisited cells)

**6. Cross-Domain Generalization Gap**
```python
gap = success_rate_domain_A - success_rate_domain_B
```

### 5.3 Difficulty Tiers

Tasks are organized into 5 tiers based on complexity.

**Tier 1: Pure Navigation**

**What It Tests**: Basic pathfinding, no mechanisms

**Example Task:**
```json
{
    "difficulty_tier": 1,
    "maze": {
        "dimensions": [8, 8],
        "start": [1, 1],
        "goal": [6, 6],
        "walls": []  # Empty maze or simple obstacles
    },
    "mechanisms": {}  # No keys, doors, etc.
}
```

**Skills Required:**
- Spatial awareness (where am I?)
- Goal-directed navigation (move toward goal)
- Obstacle avoidance (go around walls)

**Evaluation:**
- Should have >90% success rate for competent agents
- Baseline for all other tiers

---

**Tier 2: Linear Dependencies**

**What It Tests**: Sequential subtasks (A → B → C)

**Example Task: Key-Door Puzzle**
```json
{
    "difficulty_tier": 2,
    "mechanisms": {
        "keys": [{"id": "k1", "position": [2, 2], "color": "red"}],
        "doors": [{"id": "d1", "position": [4, 4], "requires_key": "red"}]
    }
}
```

**Dependency Chain:**
1. Navigate to key
2. Pick up key
3. Navigate to door
4. Unlock door
5. Navigate to goal

**Skills Required:**
- Subtask decomposition
- Memory (remember where door is after picking up key)
- Action sequencing (pickup, then unlock)

**Common Failure Modes:**
- Forgetting to pick up key
- Trying to unlock door without key
- Navigating to goal before unlocking door

---

**Tier 3: Multi-Mechanism**

**What It Tests**: Parallel dependencies, multiple paths

**Example Task: Multiple Keys and Switches**
```json
{
    "difficulty_tier": 3,
    "mechanisms": {
        "keys": [
            {"id": "k1", "position": [2, 2], "color": "red"},
            {"id": "k2", "position": [5, 1], "color": "blue"}
        ],
        "doors": [
            {"id": "d1", "position": [3, 3], "requires_key": "red"},
            {"id": "d2", "position": [6, 3], "requires_key": "blue"}
        ],
        "switches": [{"id": "sw1", "position": [4, 5], "controls": ["gate1"]}],
        "gates": [{"id": "gate1", "position": [5, 6]}]
    }
}
```

**Skills Required:**
- Planning with multiple subgoals
- Optimal ordering (which key first?)
- Resource management (can only carry one key at a time in some variants)

**Common Failure Modes:**
- Suboptimal ordering (collect far key first)
- Forgetting about mechanisms (activate switch but forget to use gate)

---

**Tier 4: Irreversibility**

**What It Tests**: One-way actions, commitment

**Example Task: Pushable Blocks**
```json
{
    "difficulty_tier": 4,
    "mechanisms": {
        "blocks": [
            {"id": "b1", "position": [3, 3], "color": "grey"},
            {"id": "b2", "position": [4, 5], "color": "grey"}
        ]
    },
    "rules": {
        "blocks_pushable": true,
        "blocks_reversible": false  # Can't pull, only push
    }
}
```

**Irreversible Actions:**
- Pushing blocks (can't unpush)
- Consumable keys (key disappears after use)
- One-shot switches (can only activate once)

**Skills Required:**
- Lookahead planning (will this push block me in?)
- Backtracking avoidance
- Commitment to plans

**Common Failure Modes:**
- Pushing block into corner (deadlock)
- Consuming key prematurely
- Activating one-shot switch before positioning

---

**Tier 5: Hidden Information**

**What It Tests**: Memory, exploration, inference

**Example Task: Hidden Switch**
```json
{
    "difficulty_tier": 5,
    "mechanisms": {
        "switches": [
            {"id": "sw1", "position": [2, 3], "visibility": "hidden"}
        ],
        "gates": [
            {"id": "gate1", "position": [5, 5]}
        ]
    },
    "rules": {
        "partial_observability": true
    }
}
```

**Hidden Information:**
- Hidden switches (invisible until discovered)
- Partial observability (limited vision radius)
- Teleporters (destination unknown until used)
- Color inference (must deduce which key opens which door)

**Skills Required:**
- Exploration (systematic search for hidden objects)
- Memory (remember locations outside current view)
- Inference (deduce rules from observations)

**Common Failure Modes:**
- Incomplete exploration (miss hidden switch)
- Forgetting locations (walk past goal because it's out of view)
- Incorrect inference (wrong key-door pairing)

### 5.4 Live Benchmark Strategy

**Problem**: Fixed benchmarks can be memorized by models trained on leaked data.

**Solution: Procedural Generation + Difficulty Estimation**

**Procedural Generation:**

```python
def generate_task(difficulty_tier, seed):
    """Generate a random task at specified difficulty."""
    rng = np.random.RandomState(seed)

    # Generate maze
    grid_size = 8 + difficulty_tier * 2  # Harder = bigger
    walls = generate_maze(grid_size, density=0.1 + difficulty_tier * 0.05, rng=rng)

    # Add mechanisms based on tier
    if difficulty_tier >= 2:
        num_keys = rng.randint(1, difficulty_tier)
        keys = place_keys(grid_size, num_keys, walls, rng)
        doors = place_doors_for_keys(keys, walls, rng)

    if difficulty_tier >= 3:
        num_switches = rng.randint(1, difficulty_tier - 1)
        switches = place_switches(grid_size, num_switches, walls, rng)
        gates = place_gates_for_switches(switches, walls, rng)

    # ... etc

    return TaskSpecification(...)
```

**Difficulty Estimation:**

After generating a task, estimate its difficulty:

```python
def estimate_difficulty(task_spec):
    """Estimate task difficulty using heuristics."""

    # Heuristics
    optimal_path_length = a_star(task_spec.start, task_spec.goal, task_spec.walls)
    num_mechanisms = count_mechanisms(task_spec)
    dependency_depth = compute_dependency_graph_depth(task_spec)

    # Weighted score
    difficulty_score = (
        0.3 * optimal_path_length +
        0.4 * num_mechanisms +
        0.3 * dependency_depth
    )

    # Verify with expert policy
    expert_success, expert_steps = run_expert(task_spec)
    if not expert_success:
        return "too_hard"  # Discard unsolvable tasks

    if expert_steps < 10:
        return "too_easy"  # Discard trivial tasks

    return difficulty_score
```

**Evaluation Protocol:**

```python
# Generate 1000 tasks at each tier
for tier in range(1, 6):
    tasks = []
    seed = tier * 10000

    while len(tasks) < 1000:
        task = generate_task(tier, seed)
        difficulty = estimate_difficulty(task)

        # Only keep tasks in appropriate difficulty range
        tier_ranges = {1: (1, 5), 2: (5, 15), 3: (15, 30), 4: (30, 50), 5: (50, 100)}
        min_diff, max_diff = tier_ranges[tier]

        if min_diff <= difficulty <= max_diff:
            tasks.append(task)

        seed += 1

    # Evaluate agent
    results = evaluate_agent(agent, tasks)
    print(f"Tier {tier}: Success rate = {results['success_rate']:.2%}")
```

**Advantages:**

1. **Contamination Resistance**: No fixed dataset to memorize
2. **Infinite Evaluation**: Generate fresh tasks for each evaluation
3. **Difficulty Control**: Ensure tasks span appropriate difficulty range
4. **Fair Comparison**: All models see same difficulty distribution

**Validation:**

- Run expert policy (A*) to verify solvability
- Run human players to validate difficulty tiers
- Compare multiple agents to establish baseline difficulty curves

---

## Appendix: Design Alternatives Considered

### A.1 Why Not Use Unity or Unreal for Domain 4?

**Considered**: Use full game engine for GUI domain

**Rejected Because:**
- Heavyweight: Unity/Unreal are multi-GB installs
- Complexity: Steep learning curve for contributors
- Licensing: Unity has runtime fee for certain use cases
- Overkill: Our GUI needs are simple (2D, turn-based)

**Chosen**: Pygame (lightweight, Python-native, MIT license)

### A.2 Why Not Use SMARTS or Habitat for Domain 2?

**Considered**: Use existing robotics simulators

**Rejected Because:**
- Overconstrained: These have specific robot embodiments
- Complex: Hard to match canonical task specifications
- Performance: Slower than MuJoCo for simple 2D tasks

**Chosen**: MuJoCo (faster, more flexible, better documented)

### A.3 Why Not Use Existing Text Adventure Engines (Z-Machine, Inform)?

**Considered**: Use Infocom-style text adventure engines

**Rejected Because:**
- Parser complexity: Natural language parsing is a separate research problem
- Compatibility: Hard to map canonical tasks to text adventure format
- Evaluation: Unclear how to measure spatial reasoning in pure text

**Chosen**: Custom text adapter with simple command set ("go north", "take key")

---

## Document Changelog

### Version 1.0 (2026-02-06)
- Initial technical design document
- Covers technology stack, architecture, cross-domain vision, evaluation methodology
- Written for MultiNet v1.1 release

---

## References

**MiniGrid:**
- Farama Foundation: https://minigrid.farama.org/
- GitHub: https://github.com/Farama-Foundation/Minigrid
- Paper: Chevalier-Boisvert et al. (2018), "Minimalistic Gridworld Environment for OpenAI Gym"

**Hexagonal Grids:**
- Red Blob Games Tutorial: https://www.redblobgames.com/grids/hexagons/
- Birchfield & Tomasi (1998), "Depth Discontinuities by Pixel-to-Pixel Stereo"

**Archimedean Tilings:**
- Grünbaum & Shephard (1987), "Tilings and Patterns"
- Wikipedia: https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons

**Gymnasium API:**
- Documentation: https://gymnasium.farama.org/
- GitHub: https://github.com/Farama-Foundation/Gymnasium

**MuJoCo:**
- Documentation: https://mujoco.readthedocs.io/
- Paper: Todorov et al. (2012), "MuJoCo: A physics engine for model-based control"

---

**End of Technical Design Document**
