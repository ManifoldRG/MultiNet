# MultiGrid Backend Documentation

## Overview

The MultiGrid Backend is an experimental implementation of the `AbstractGridBackend` interface that supports exotic grid tilings (hexagonal and triangular) in addition to standard square grids. It bridges the standard MiniGrid task specification format with a custom MultiGrid environment system designed for research on non-traditional spatial representations.

**Purpose**: Enable research and evaluation on exotic grid tilings while maintaining compatibility with the standard backend interface and task specification format.

**Location**: `/src/v1_1/gridworld/backends/multigrid_backend.py`

**Status**: Experimental - Research use only

**Target Audience**: Researchers investigating how agents generalize across different spatial topologies.

---

## Architecture

### Exotic Tiling Support

The key differentiator of MultiGrid Backend is its support for three tiling types:

1. **Square Tiling** (Standard): 4-connected grid with 90° rotations
2. **Hexagonal Tiling**: 6-connected grid with 60° rotations
3. **Triangular Tiling**: Variable connectivity with complex navigation

```
┌───────────────────────────────────────────────────────────┐
│                   Tiling Types                             │
└───────────────────────────────────────────────────────────┘

SQUARE (4-connected)        HEXAGONAL (6-connected)
┌───┬───┬───┬───┐              ⬡   ⬡   ⬡   ⬡
│   │   │   │   │             ⬡   ⬡   ⬡   ⬡
├───┼───┼───┼───┤              ⬡   ⬡   ⬡   ⬡
│   │ A │   │   │             ⬡   A   ⬡   ⬡
├───┼───┼───┼───┤              ⬡   ⬡   ⬡   ⬡
│   │   │   │   │             ⬡   ⬡   ⬡   ⬡
└───┴───┴───┴───┘

Neighbors: 4 (N/S/E/W)      Neighbors: 6 (all adjacent)

TRIANGULAR (variable)
    △ ▽ △ ▽
    ▽ △ ▽ △
    △ A △ ▽
    ▽ △ ▽ △

Neighbors: 3 or 9 depending on orientation
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│           MultiGrid Backend Architecture                 │
└─────────────────────────────────────────────────────────┘

TaskSpecification (MiniGrid format)
         │
         ▼
┌────────────────────────┐
│ MultiGridBackend       │
│  ._convert_task_spec() │
└───────┬────────────────┘
        │
        ├──► Convert coordinates: integer → normalized [0,1]
        ├──► Convert objects: keys/doors/blocks → unified format
        ├──► Add tiling specification
        │
        ▼
MultiGrid Task Spec (dict)
        │
        ▼
┌────────────────────────┐
│  MultiGridEnv          │
│  (custom environment)  │
└───────┬────────────────┘
        │
        ├──► Tiling: square/hex/triangle
        ├──► Scene: agent + objects + walls
        ├──► Goal: reach/collect/push
        │
        ▼
   GridState (backend-agnostic)
```

### Coordinate System Translation

A major architectural challenge is coordinate system conversion:

**MiniGrid Format** (Integer Grid):
- Position: `(x=3, y=5)` in an 8×8 grid
- Semantics: Absolute grid cell coordinates
- Range: `[0, width)` × `[0, height)`

**MultiGrid Format** (Normalized Continuous):
- Position: `{"x": 0.375, "y": 0.625}`
- Semantics: Normalized position in [0, 1] × [0, 1]
- Calculation: `x_norm = x / width`, `y_norm = y / height`

**Rationale**: Normalized coordinates allow the same task to be rendered on different tilings. A task defined on a square grid can be "ported" to hexagonal by reinterpreting the normalized positions.

---

## Key Components

### MultiGridBackend Class

```python
class MultiGridBackend(AbstractGridBackend):
    """
    Backend adapter for the custom MultiGrid system.
    Supports exotic tilings: square, hex, triangle.
    """

    def __init__(self, tiling="square", render_mode="rgb_array",
                 render_width=640, render_height=640)
    def configure(self, task_spec: TaskSpecification) -> None
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]
    def render(self) -> np.ndarray
    def get_mission_text(self) -> str
    def get_state(self) -> GridState
    def close(self) -> None

    # Internal methods
    def _convert_task_spec(self, spec: TaskSpecification) -> dict
    def _build_grid_state(self) -> GridState
```

### Constructor: `__init__(tiling, render_mode, render_width, render_height)`

**Parameters**:
- `tiling` (str): Tiling type
  - `"square"`: Standard 4-connected grid (default)
  - `"hex"`: Hexagonal 6-connected grid
  - `"triangle"`: Triangular variable-connected grid
- `render_mode` (str): Rendering mode
  - `"rgb_array"`: Returns RGB numpy arrays (recommended)
  - `"human"`: Opens visualization window
- `render_width` (int): Width of rendered images in pixels (default 640)
- `render_height` (int): Height of rendered images in pixels (default 640)

**Example**:
```python
from gridworld.backends import MultiGridBackend

# Standard square tiling (same as MiniGrid)
backend = MultiGridBackend(tiling="square")

# Hexagonal tiling for research
backend = MultiGridBackend(tiling="hex", render_mode="rgb_array")

# Triangle tiling with custom render size
backend = MultiGridBackend(tiling="triangle",
                           render_width=800,
                           render_height=800)
```

**Initialization Details**:
- Stores tiling type and rendering parameters
- Does NOT create environment (lazy initialization on configure)
- Initializes step tracking (`_step_count`, `_max_steps`)

### Method: `configure(task_spec)`

Configures the backend with a task specification and creates the MultiGrid environment.

**Parameters**:
- `task_spec` (TaskSpecification): Task to configure

**Returns**: None

**Side Effects**:
- Converts task spec to MultiGrid format
- Creates `MultiGridEnv` instance
- Sets `_configured` flag

**Example**:
```python
from gridworld.task_spec import TaskSpecification
from gridworld.backends import MultiGridBackend

# Load standard MiniGrid task
spec = TaskSpecification.from_json("task.json")

# Configure with hexagonal tiling
backend = MultiGridBackend(tiling="hex")
backend.configure(spec)

# The same task is now running on a hex grid!
```

**Conversion Process**:

The `_convert_task_spec()` method transforms MiniGrid format → MultiGrid format:

1. **Coordinates**: Integer grid positions → Normalized [0,1] positions
2. **Objects**: Separate mechanism types → Unified objects list
3. **Tiling**: Implicit square → Explicit tiling specification
4. **Goal**: Standard format → MultiGrid goal spec

See "Task Specification Conversion" section for details.

### Method: `reset(seed=None)`

Resets the environment to initial state.

**Parameters**:
- `seed` (int, optional): Random seed for reproducibility

**Returns**:
- `observation` (np.ndarray): RGB image of initial state
- `state` (GridState): Backend-agnostic state
- `info` (dict): Additional information

**Raises**:
- `RuntimeError`: If not configured

**Example**:
```python
obs, state, info = backend.reset(seed=42)
print(f"Observation shape: {obs.shape}")  # (640, 640, 3)
print(f"Agent position: {state.agent_position}")
```

**Note**: Unlike MiniGridBackend, MultiGridBackend does NOT use TaskParser. It directly creates a MultiGridEnv from the converted task spec.

### Method: `step(action)`

Executes one action with automatic action space translation.

**Parameters**:
- `action` (int): MiniGrid action (0-6)

**Returns**:
- `observation`, `reward`, `terminated`, `truncated`, `state`, `info`

**Action Translation**:

MultiGrid uses a different action enumeration than MiniGrid. The backend automatically translates:

| MiniGrid Action | MultiGrid Action | Description |
|-----------------|------------------|-------------|
| 0: turn_left | 2: TURN_LEFT | Rotate counterclockwise |
| 1: turn_right | 3: TURN_RIGHT | Rotate clockwise |
| 2: forward | 0: FORWARD | Move in facing direction |
| 3: pickup | 4: PICKUP | Pick up object in front |
| 4: drop | 5: DROP | Drop held object |
| 5: toggle | 6: PUSH | Interact with object |
| 6: done | 7: WAIT | No-op action |

**Example**:
```python
# Use standard MiniGrid action indices
obs, reward, terminated, truncated, state, info = backend.step(2)  # forward

# Translation happens automatically
# Agent can use same policy on MiniGrid or MultiGrid
```

**Design Rationale**: Action translation enables:
- **Policy Reuse**: Same agent works on both backends
- **Backend Comparison**: Evaluate same policy on square vs hex grids
- **Simplified Evaluation**: Caller doesn't need backend-specific knowledge

### Method: `_convert_task_spec(spec)`

Internal method that converts MiniGrid TaskSpecification to MultiGrid format.

**Parameters**:
- `spec` (TaskSpecification): MiniGrid format task

**Returns**:
- `dict`: MultiGrid format task specification

**Conversion Details**:

```python
# MiniGrid format
{
    "maze": {
        "dimensions": [8, 8],
        "start": [1, 1],
        "goal": [6, 6],
        "walls": [[3, 3], [3, 4]]
    },
    "mechanisms": {
        "keys": [{"id": "key1", "position": [2, 2], "color": "red"}],
        "doors": [{"id": "door1", "position": [4, 4], "requires_key": "red"}],
        "blocks": [{"id": "block1", "position": [3, 5], "color": "grey"}]
    }
}

# Converts to MultiGrid format
{
    "tiling": {
        "type": "hex",  # From backend.tiling_type
        "grid_size": {"width": 8, "height": 8}
    },
    "scene": {
        "agent": {
            "position": {"x": 0.125, "y": 0.125},  # 1/8, 1/8
            "facing": 0
        },
        "objects": [
            {
                "id": "key1",
                "type": "movable",
                "color": "red",
                "position": {"x": 0.25, "y": 0.25}  # 2/8, 2/8
            },
            {
                "id": "door1",
                "type": "wall",
                "color": "red",
                "position": {"x": 0.5, "y": 0.5}  # 4/8, 4/8
            },
            {
                "id": "block1",
                "type": "movable",
                "color": "grey",
                "position": {"x": 0.375, "y": 0.625}  # 3/8, 5/8
            }
        ],
        "walls": [[3, 3], [3, 4]]  # Kept as absolute coordinates
    },
    "goal": {
        "type": "reach_position",
        "target": {"x": 0.75, "y": 0.75}  # 6/8, 6/8
    },
    "limits": {
        "max_steps": 100
    }
}
```

**Object Type Mapping**:
- Keys → `"movable"` (can be picked up)
- Doors → `"wall"` (blocking barrier with color)
- Blocks → `"movable"` (pushable)
- Switches → Not yet supported
- Gates → Not yet supported

**Limitations**:
- Switches and gates not implemented in MultiGrid
- Teleporters not supported
- Hazards not supported
- All mechanisms except reach_position goals are limited

### Method: `_build_grid_state()`

Internal method that extracts GridState from MultiGrid environment.

**Returns**:
- `GridState`: Backend-agnostic state representation

**Extraction Process**:

1. **Agent Position**: Convert from cell_id → normalized coordinates → grid coordinates
2. **Agent Carrying**: Extract from `state.agent.holding`
3. **Block Positions**: Iterate through `state.objects` and convert positions
4. **Goal State**: Check `state.check_goal()`

**Coordinate Conversion**:

```python
# MultiGrid stores positions as cell IDs in the tiling
cell_id = state.agent.cell_id

# Convert to normalized [0,1] coordinates
normalized_pos = tiling.cell_to_canonical(cell_id)
# normalized_pos = (0.375, 0.625)

# Convert to grid coordinates
grid_pos = (
    int(normalized_pos[0] * grid_width),
    int(normalized_pos[1] * grid_height)
)
# grid_pos = (3, 5) for 8×8 grid
```

**Example Output**:
```python
state = backend.get_state()
# GridState(
#     agent_position=(3, 5),
#     agent_direction=2,
#     agent_carrying="key1",
#     step_count=15,
#     max_steps=100,
#     block_positions={"block1": (4, 6)},
#     goal_reached=False
# )
```

---

## Task Specification Conversion

### Coordinate Normalization

**Why Normalize?**

Different tilings have different spatial properties:
- Square: 4 neighbors, regular spacing
- Hex: 6 neighbors, 60° angles
- Triangle: Variable neighbors, complex topology

Normalized coordinates abstract over these differences, allowing the "same" task on different tilings.

**Example**:

```python
# Task: Agent at (2, 3), goal at (6, 7) in 8×8 grid

# Square tiling: 4 steps right, 4 steps down = 8 steps minimum
# Hex tiling: Can move diagonally, ~6 steps minimum
# Triangle tiling: Complex, depends on orientation

# Normalized positions allow all three to work:
# Agent: (0.25, 0.375)
# Goal: (0.75, 0.875)
```

**Normalization Formula**:

```python
x_normalized = x_grid / grid_width
y_normalized = y_grid / grid_height

# Example: Position (3, 5) in 8×8 grid
# x_norm = 3 / 8 = 0.375
# y_norm = 5 / 8 = 0.625
```

**Denormalization** (for GridState extraction):

```python
x_grid = int(x_normalized * grid_width)
y_grid = int(y_normalized * grid_height)

# Example: Normalized (0.375, 0.625) in 8×8 grid
# x_grid = int(0.375 * 8) = 3
# y_grid = int(0.625 * 8) = 5
```

### Object Type Unification

MiniGrid has separate lists for different mechanism types. MultiGrid uses a unified objects list with a `type` field.

**Mapping**:

| MiniGrid Mechanism | MultiGrid Type | Notes |
|--------------------|----------------|-------|
| `keys` | `"movable"` | Can be picked up and carried |
| `doors` | `"wall"` | Blocking barrier (unlock not implemented) |
| `blocks` | `"movable"` | Pushable objects |
| `switches` | N/A | Not yet supported |
| `gates` | N/A | Not yet supported |
| `teleporters` | N/A | Not yet supported |
| `hazards` | N/A | Not yet supported |

**Example Conversion**:

```python
# MiniGrid: Separate lists
"mechanisms": {
    "keys": [
        {"id": "k1", "position": [2, 2], "color": "red"},
        {"id": "k2", "position": [3, 3], "color": "blue"}
    ],
    "doors": [
        {"id": "d1", "position": [5, 5], "requires_key": "red"}
    ],
    "blocks": [
        {"id": "b1", "position": [4, 4], "color": "grey"}
    ]
}

# MultiGrid: Unified objects list
"scene": {
    "objects": [
        {"id": "k1", "type": "movable", "color": "red",
         "position": {"x": 0.25, "y": 0.25}},
        {"id": "k2", "type": "movable", "color": "blue",
         "position": {"x": 0.375, "y": 0.375}},
        {"id": "d1", "type": "wall", "color": "red",
         "position": {"x": 0.625, "y": 0.625}},
        {"id": "b1", "type": "movable", "color": "grey",
         "position": {"x": 0.5, "y": 0.5}}
    ]
}
```

### Goal Specification

MultiGrid supports multiple goal types with slight differences in format.

**Supported Goals**:

1. **Reach Position**:
```python
# MiniGrid
"goal": {
    "goal_type": "reach_position",
    "target": [6, 6]
}

# MultiGrid
"goal": {
    "type": "reach_position",
    "target": {"x": 0.75, "y": 0.75}  # Normalized
}
```

2. **Collect All**:
```python
# MiniGrid
"goal": {
    "goal_type": "collect_all",
    "target_ids": ["key1", "key2"]
}

# MultiGrid
"goal": {
    "type": "collect_all",
    "target_ids": ["key1", "key2"]
}
```

3. **Push Block To**:
```python
# MiniGrid
"goal": {
    "goal_type": "push_block_to",
    "target_ids": ["block1"],
    "target_positions": [[7, 7]]
}

# MultiGrid
"goal": {
    "type": "push_block_to",
    "target_ids": ["block1"],
    "target_positions": [{"x": 0.875, "y": 0.875}]
}
```

---

## Usage Examples

### Example 1: Square vs Hex Comparison

```python
from gridworld.backends import MultiGridBackend
from gridworld.task_spec import TaskSpecification

# Load a navigation task
spec = TaskSpecification.from_json("tasks/navigation_8x8.json")

# Evaluate on square grid
square_backend = MultiGridBackend(tiling="square")
square_backend.configure(spec)
obs, state, info = square_backend.reset(seed=42)

# Count steps to goal
steps_square = 0
done = False
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, state, info = square_backend.step(action)
    steps_square += 1
    done = terminated or truncated

print(f"Square grid: {steps_square} steps")

# Evaluate on hexagonal grid
hex_backend = MultiGridBackend(tiling="hex")
hex_backend.configure(spec)
obs, state, info = hex_backend.reset(seed=42)

steps_hex = 0
done = False
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, state, info = hex_backend.step(action)
    steps_hex += 1
    done = terminated or truncated

print(f"Hexagonal grid: {steps_hex} steps")
print(f"Difference: {abs(steps_square - steps_hex)} steps")
```

### Example 2: Multi-Tiling Evaluation

```python
from gridworld.backends import MultiGridBackend
from gridworld.task_spec import TaskSpecification

def evaluate_across_tilings(policy_fn, task_path, tilings=["square", "hex", "triangle"]):
    """
    Evaluate a policy on the same task across different tilings.
    """
    spec = TaskSpecification.from_json(task_path)

    results = {}
    for tiling_type in tilings:
        backend = MultiGridBackend(tiling=tiling_type)
        backend.configure(spec)

        # Run episode
        obs, state, info = backend.reset(seed=42)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, state, info = backend.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        results[tiling_type] = {
            "success": state.goal_reached,
            "reward": total_reward,
            "steps": steps
        }

        backend.close()

    return results

# Example usage
results = evaluate_across_tilings(my_policy, "task.json")
for tiling, metrics in results.items():
    print(f"{tiling:10s}: success={metrics['success']}, "
          f"steps={metrics['steps']}, reward={metrics['reward']:.3f}")
```

### Example 3: Visualization of Different Tilings

```python
from gridworld.backends import MultiGridBackend
from gridworld.task_spec import TaskSpecification
import matplotlib.pyplot as plt

# Load task
spec = TaskSpecification.from_json("task.json")

# Create backends for each tiling
tilings = ["square", "hex", "triangle"]
backends = {t: MultiGridBackend(tiling=t) for t in tilings}

# Configure and reset
for tiling, backend in backends.items():
    backend.configure(spec)
    backend.reset(seed=42)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, tiling in zip(axes, tilings):
    backend = backends[tiling]
    img = backend.render()
    ax.imshow(img)
    ax.set_title(f"{tiling.capitalize()} Tiling")
    ax.axis('off')

plt.tight_layout()
plt.savefig("tiling_comparison.png")
plt.show()

# Cleanup
for backend in backends.values():
    backend.close()
```

### Example 4: Custom Task on Hex Grid

```python
from gridworld.backends import MultiGridBackend

# Define task programmatically
task_data = {
    "task_id": "hex_navigation",
    "seed": 42,
    "difficulty_tier": 1,
    "max_steps": 50,
    "maze": {
        "dimensions": [8, 8],
        "start": [1, 1],
        "goal": [6, 6],
        "walls": [[3, 3], [3, 4], [4, 3]]  # Small obstacle
    },
    "mechanisms": {
        "keys": [],
        "doors": [],
        "blocks": []
    },
    "goal": {
        "type": "reach_position",
        "target": [6, 6]
    }
}

# Load on hexagonal grid
backend = MultiGridBackend(tiling="hex")
spec = TaskSpecification.from_dict(task_data)
backend.configure(spec)

# Run episode
obs, state, info = backend.reset()
print(f"Mission: {backend.get_mission_text()}")
print(f"Agent starts at: {state.agent_position}")

# Take some actions
for action in [2, 2, 1, 2, 2]:  # forward, forward, turn_right, forward, forward
    obs, reward, terminated, truncated, state, info = backend.step(action)
    print(f"Position: {state.agent_position}, Direction: {state.agent_direction}")

    if terminated:
        if reward > 0:
            print("Goal reached!")
        break

backend.close()
```

### Example 5: Action Space Verification

```python
from gridworld.backends import MiniGridBackend, MultiGridBackend
from gridworld.task_spec import TaskSpecification

# Load task
spec = TaskSpecification.from_json("task.json")

# Create both backends
minigrid = MiniGridBackend()
multigrid = MultiGridBackend(tiling="square")

minigrid.configure(spec)
multigrid.configure(spec)

# Reset with same seed
obs1, state1, _ = minigrid.reset(seed=42)
obs2, state2, _ = multigrid.reset(seed=42)

print("Initial states:")
print(f"  MiniGrid: pos={state1.agent_position}, dir={state1.agent_direction}")
print(f"  MultiGrid: pos={state2.agent_position}, dir={state2.agent_direction}")

# Execute same actions
actions = [2, 2, 1, 2]  # forward, forward, turn_right, forward
for action in actions:
    obs1, r1, t1, tr1, state1, _ = minigrid.step(action)
    obs2, r2, t2, tr2, state2, _ = multigrid.step(action)

    print(f"\nAfter action {action}:")
    print(f"  MiniGrid: pos={state1.agent_position}")
    print(f"  MultiGrid: pos={state2.agent_position}")

    # Positions should match (for square tiling)
    assert state1.agent_position == state2.agent_position, "Position mismatch!"

print("\n✓ Action space translation verified!")

minigrid.close()
multigrid.close()
```

---

## Feature Support and Limitations

### Tiling Support

| Tiling | Status | Notes |
|--------|--------|-------|
| Square | ✓ Full | Same as MiniGrid |
| Hexagonal | ✓ Experimental | 6-connected, 60° angles |
| Triangular | ✓ Experimental | Complex topology, variable connectivity |

### Mechanism Support

| Mechanism | Status | Notes |
|-----------|--------|-------|
| Walls | ✓ Supported | Static barriers |
| Keys | Partial | Can be placed, but pickup may not work correctly |
| Doors | ✗ Limited | Rendered as colored walls, no unlock mechanic |
| Switches | ✗ Not implemented | MultiGrid enhancement needed |
| Gates | ✗ Not implemented | MultiGrid enhancement needed |
| Blocks | Partial | Rendered, but push mechanic unverified |
| Hazards | ✗ Not implemented | No hazard support in MultiGrid |
| Teleporters | ✗ Not implemented | Planned feature |

### Goal Support

| Goal Type | Status | Implementation |
|-----------|--------|----------------|
| Reach Position | ✓ Supported | Fully functional |
| Collect All | ⚠️ Partial | Goal spec converted, checking may not work |
| Push Block To | ⚠️ Partial | Goal spec converted, checking may not work |
| Survive Steps | ⚠️ Partial | Can be specified, but no special handling |

**Legend**: ✓ Full support | ⚠️ Partial support | ✗ Not supported

### Known Limitations

1. **Mechanism Interactivity**: Many mechanisms (doors, switches, gates) are not yet implemented in the underlying MultiGrid environment. They may be converted and placed but won't function.

2. **Coordinate Precision**: Integer-to-normalized conversion can lose precision:
   ```python
   # Original: (3, 5) in 8×8 grid
   # Normalized: (0.375, 0.625)
   # Back to grid: (3, 5)  ✓ OK

   # Original: (7, 7) in 8×8 grid
   # Normalized: (0.875, 0.875)
   # Back to grid: (7, 7)  ✓ OK

   # But for odd dimensions:
   # Original: (3, 5) in 7×7 grid
   # Normalized: (0.428571, 0.714286)
   # Back to grid: (2, 4)  ✗ Precision loss!
   ```
   **Recommendation**: Use power-of-2 dimensions (8×8, 16×16) for exact conversion.

3. **Rendering Quality**: MultiGrid rendering is experimental. Hex and triangle tilings may have visual artifacts.

4. **Performance**: MultiGrid is ~1.5× slower than MiniGrid due to coordinate conversions and less optimized implementation.

5. **Partial Observability**: Not yet implemented. All observations are full-grid.

---

## Performance Characteristics

### Timing Benchmarks (8×8 grid, square tiling)

| Operation | MiniGrid | MultiGrid | Overhead |
|-----------|----------|-----------|----------|
| configure() | ~0.1 ms | ~5 ms | 50× |
| reset() | ~10 ms | ~15 ms | 1.5× |
| step() | ~3 ms | ~5 ms | 1.67× |
| render() | ~4 ms | ~8 ms | 2× |

**Total episode (100 steps)**: ~600-800 ms (vs ~400 ms for MiniGrid)

### Hexagonal and Triangle Tilings

Exotic tilings add additional overhead:

| Tiling | Episode Time | Relative to Square |
|--------|--------------|-------------------|
| Square | ~600 ms | 1.0× |
| Hex | ~750 ms | 1.25× |
| Triangle | ~900 ms | 1.5× |

**Bottlenecks**:
1. Cell ID ↔ normalized coordinate conversion
2. Neighbor computation for non-square tilings
3. Rendering complex tiling shapes

---

## Comparison with MiniGrid Backend

| Aspect | MiniGridBackend | MultiGridBackend |
|--------|-----------------|------------------|
| **Maturity** | Production-ready | Experimental |
| **Tilings** | Square only | Square, hex, triangle |
| **Mechanisms** | Full support | Limited (keys/walls only) |
| **Performance** | Fast (~400ms/episode) | Slower (~600-900ms/episode) |
| **Rendering** | High quality | Experimental quality |
| **Partial Obs** | Supported | Not yet |
| **Backend Source** | Gymnasium MiniGrid | Custom MultiGrid |
| **Use Case** | Standard evaluation | Research on exotic tilings |
| **Stability** | Stable | May have bugs |
| **Documentation** | Comprehensive | Limited |

**When to Use MultiGrid**:
- Research on spatial representation and topology
- Investigating agent generalization across grid types
- Exploring hexagonal or triangular navigation

**When to Use MiniGrid**:
- Production evaluation
- Need full mechanism support
- Performance is critical
- Stability and maturity required

---

## Integration with Evaluation Pipeline

### Standard Evaluation Pattern

```python
from gridworld.backends import MultiGridBackend
from gridworld.task_spec import TaskSpecification

def run_multigrid_evaluation(agent, task_files, tiling="square"):
    """
    Evaluation loop using MultiGrid backend.
    """
    backend = MultiGridBackend(tiling=tiling, render_mode="rgb_array")
    results = {}

    for task_file in task_files:
        spec = TaskSpecification.from_json(task_file)
        backend.configure(spec)

        # Run episode
        obs, state, info = backend.reset(seed=42)
        episode_data = {
            "tiling": tiling,
            "observations": [obs],
            "actions": [],
            "rewards": []
        }

        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, state, info = backend.step(action)

            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            done = terminated or truncated

        episode_data["success"] = state.goal_reached
        episode_data["total_reward"] = sum(episode_data["rewards"])
        episode_data["steps"] = len(episode_data["actions"])

        results[spec.task_id] = episode_data

    backend.close()
    return results
```

### Cross-Backend Comparison

```python
from gridworld.backends import MiniGridBackend, MultiGridBackend

def compare_backends(agent, task_path):
    """
    Compare agent performance on MiniGrid vs MultiGrid (square).
    """
    spec = TaskSpecification.from_json(task_path)

    # MiniGrid
    mg_backend = MiniGridBackend()
    mg_backend.configure(spec)
    obs, state, _ = mg_backend.reset(seed=42)

    mg_steps = 0
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, state, _ = mg_backend.step(action)
        mg_steps += 1
        done = terminated or truncated

    mg_success = state.goal_reached
    mg_backend.close()

    # MultiGrid
    mu_backend = MultiGridBackend(tiling="square")
    mu_backend.configure(spec)
    obs, state, _ = mu_backend.reset(seed=42)

    mu_steps = 0
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, state, _ = mu_backend.step(action)
        mu_steps += 1
        done = terminated or truncated

    mu_success = state.goal_reached
    mu_backend.close()

    return {
        "minigrid": {"success": mg_success, "steps": mg_steps},
        "multigrid": {"success": mu_success, "steps": mu_steps}
    }
```

---

## Troubleshooting

### Issue 1: ImportError for MultiGrid

**Error**: `ModuleNotFoundError: No module named 'multigrid'`

**Cause**: MultiGrid module not in Python path

**Solution**:
```python
# The backend handles this automatically via sys.path manipulation
# But if you see this error, check:
import sys
from pathlib import Path

multigrid_path = Path(__file__).parent.parent.parent / "multigrid"
if str(multigrid_path.parent) not in sys.path:
    sys.path.insert(0, str(multigrid_path.parent))
```

### Issue 2: Coordinate Mismatch

**Symptom**: Agent/objects appear at wrong positions

**Cause**: Coordinate normalization precision loss

**Solution**: Use power-of-2 dimensions (8×8, 16×16, 32×32)

### Issue 3: Mechanisms Not Working

**Symptom**: Keys can't be picked up, doors don't open

**Cause**: Mechanism interaction not yet implemented in MultiGrid

**Solution**: Currently, MultiGrid backend is best for navigation-only tasks. For tasks requiring mechanisms, use MiniGridBackend.

### Issue 4: Rendering Artifacts on Hex/Triangle

**Symptom**: Visual glitches in rendered images

**Cause**: Experimental rendering code

**Solution**: This is a known limitation. For publication-quality visualizations, use square tiling or generate custom renders.

---

## Future Enhancements

### Planned Features

1. **Full Mechanism Support**:
   - Implement switches and gates in MultiGrid
   - Add door unlock mechanic
   - Add hazard tiles

2. **Partial Observability**:
   - Limited agent field of view
   - Fog of war
   - Memory-dependent tasks

3. **Improved Rendering**:
   - High-quality hex/triangle tile graphics
   - Customizable visual themes
   - Animation support

4. **Performance Optimization**:
   - Cache coordinate conversions
   - Optimize neighbor lookups for exotic tilings
   - Vectorized rendering

5. **Additional Tilings**:
   - Octagonal + square (Islamic tiling)
   - Penrose tiling (aperiodic)
   - Voronoi diagrams

### Research Directions

- **Topology Invariance**: Do agents learn topology-invariant navigation strategies?
- **Transfer Learning**: Does training on hex grids improve performance on square grids?
- **Spatial Reasoning**: How do different tilings affect spatial reasoning tasks?

---

## See Also

- [MiniGrid Backend Documentation](./minigrid_backend.md): Production backend for standard tasks
- [Task Parser Documentation](./task_parser.md): How tasks are parsed
- [AbstractGridBackend Interface](../gridworld/backends/base.py): Backend interface specification
- [MultiGrid Environment](../multigrid/env.py): Underlying custom environment
- [Tiling Theory](../../docs/tiling_theory.md): Mathematical background on grid tilings
