# Task Parser Documentation

## Overview

The Task Parser is a critical component of the MiniGrid evaluation framework that transforms declarative JSON task specifications into fully configured, executable MiniGrid environments. It acts as the bridge between high-level task definitions and low-level environment instantiation.

**Purpose**: Enable researchers and evaluators to define gridworld puzzles in a human-readable JSON format without needing to write Python code or understand MiniGrid internals.

**Location**: `/src/v1_1/gridworld/task_parser.py`

**Key Classes**:
- `TaskParser`: Main parser class that orchestrates environment creation
- Helper functions: `load_task_from_file()`, `load_task_from_dict()`

---

## Architecture

### Design Philosophy

The Task Parser follows a three-phase architecture:

1. **Validation Phase**: Verify task specification correctness
2. **Environment Creation Phase**: Instantiate and initialize the base environment
3. **Population Phase**: Add task-specific objects to the grid

This separation ensures that errors are caught early (validation) before expensive environment creation, and that initialization order is handled correctly (creation before population).

### Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                     Task Parser Flow                         │
└─────────────────────────────────────────────────────────────┘

JSON File                    TaskSpecification
   or                               │
Dictionary                          │
    │                               │
    └──────────┬────────────────────┘
               │
               ▼
        ┌─────────────┐
        │TaskParser   │
        │  .parse()   │
        └──────┬──────┘
               │
               ├──► 1. Validate Specification
               │       - Bounds checking
               │       - Dependency validation
               │       - Consistency checks
               │
               ├──► 2. Create Environment
               │       - Instantiate CustomMiniGridEnv
               │       - Call reset() to initialize grid
               │       - Set up border walls
               │
               └──► 3. Populate Grid
                       - Add interior walls
                       - Place goal marker
                       - Add keys (collectible items)
                       - Add doors (barriers)
                       - Add gates (must come before switches!)
                       - Add switches (control gates)
                       - Add blocks (pushable)
                       - Add hazards (lava, pits)
                       - Set agent position (last!)
                       │
                       ▼
              CustomMiniGridEnv
              (Ready for use)
```

### Critical Design Decisions

#### 1. Why Reset Inside Parser?

The `TaskParser.parse()` method calls `env.reset()` internally. This might seem odd since backends also have a `reset()` method. The rationale:

- **Grid Initialization**: MiniGrid requires `reset()` to be called before the grid can be populated. The `_gen_grid()` method (called by `reset()`) creates the grid structure and adds border walls.
- **Single Responsibility**: The parser is responsible for creating a *fully configured* environment. Calling reset outside would require the caller to know about this implementation detail.
- **Avoids Double Reset**: Backend `reset()` methods call `parser.parse()`, which already resets. If the backend also called `env.reset()`, it would wipe out all placed objects.

```python
# WRONG: This would wipe out all objects!
env = parser.parse(task_spec)
env.reset()  # ← Don't do this!

# CORRECT: Parser handles reset internally
env = parser.parse(task_spec)
# Environment is ready to use
```

#### 2. Object Placement Order

The `_populate_grid()` method places objects in a specific order to handle dependencies:

1. **Clear interior** (preserve border walls)
2. **Walls** (static barriers)
3. **Goal** (win condition marker)
4. **Keys** (collectible items)
5. **Doors** (barriers that require keys)
6. **Gates** (barriers controlled by switches) ← Must come before switches
7. **Switches** (controls that toggle gates)
8. **Blocks** (pushable objects)
9. **Hazards** (lava, pits, spikes)
10. **Agent position** (always last to ensure correct spawn)

**Why gates before switches?** Switches store references to gate IDs and validate them during placement. If switches are placed first, they'll fail to find their target gates.

**Why agent position last?** If the task specification accidentally places an object at the agent's start position, placing the agent last ensures it spawns correctly anyway.

---

## Key Components

### TaskParser Class

```python
class TaskParser:
    """
    Parse TaskSpecification and create configured MiniGrid environments.
    """

    def __init__(self, render_mode: Optional[str] = None)
    def parse(self, spec: TaskSpecification, seed: Optional[int] = None) -> CustomMiniGridEnv
    def parse_file(self, path: Union[str, Path]) -> CustomMiniGridEnv
    def parse_dict(self, data: dict) -> CustomMiniGridEnv
    def _populate_grid(self, env: CustomMiniGridEnv, spec: TaskSpecification)
```

#### Constructor: `__init__(render_mode)`

**Parameters**:
- `render_mode` (str, optional): Rendering mode for created environments
  - `"human"`: Opens a window for human viewing
  - `"rgb_array"`: Returns RGB numpy arrays (for headless evaluation)
  - `None`: No rendering (fastest)

**Example**:
```python
# For headless server evaluation
parser = TaskParser(render_mode="rgb_array")

# For interactive debugging
parser = TaskParser(render_mode="human")
```

#### Method: `parse(spec, seed=None)`

The core parsing method. Transforms a TaskSpecification into a configured environment.

**Parameters**:
- `spec` (TaskSpecification): The task to parse
- `seed` (int, optional): Random seed override. If None, uses `spec.seed`

**Returns**:
- `CustomMiniGridEnv`: Configured and ready-to-use environment

**Raises**:
- `ValueError`: If the task specification fails validation

**Example**:
```python
from gridworld.task_spec import TaskSpecification
from gridworld.task_parser import TaskParser

# Load specification
spec = TaskSpecification.from_json("task_001.json")

# Create parser and parse
parser = TaskParser(render_mode="rgb_array")
env = parser.parse(spec, seed=42)

# Environment is ready to use
obs, info = env.reset()
```

#### Method: `parse_file(path)`

Convenience method that loads a JSON file and parses it.

**Parameters**:
- `path` (str or Path): Path to JSON task specification file

**Returns**:
- `CustomMiniGridEnv`: Configured environment

**Example**:
```python
parser = TaskParser()
env = parser.parse_file("tasks/navigation/task_001.json")
```

#### Method: `parse_dict(data)`

Convenience method that parses a dictionary (e.g., loaded from JSON or constructed programmatically).

**Parameters**:
- `data` (dict): Dictionary containing task specification

**Returns**:
- `CustomMiniGridEnv`: Configured environment

**Example**:
```python
import json

with open("task.json") as f:
    data = json.load(f)

parser = TaskParser()
env = parser.parse_dict(data)
```

### Helper Functions

#### `load_task_from_file(path, render_mode=None)`

Top-level convenience function for the most common use case: loading a task from a JSON file.

**Parameters**:
- `path` (str or Path): Path to JSON file
- `render_mode` (str, optional): Rendering mode

**Returns**:
- `CustomMiniGridEnv`: Configured environment

**Example**:
```python
from gridworld.task_parser import load_task_from_file

# One-liner to load and parse
env = load_task_from_file("task.json", render_mode="rgb_array")
```

#### `load_task_from_dict(data, render_mode=None)`

Top-level convenience function for loading from a dictionary.

**Parameters**:
- `data` (dict): Task specification dictionary
- `render_mode` (str, optional): Rendering mode

**Returns**:
- `CustomMiniGridEnv`: Configured environment

---

## Usage Examples

### Example 1: Basic Navigation Task

```python
from gridworld.task_parser import load_task_from_file

# Load a simple navigation task
env = load_task_from_file("tasks/tier1/navigate_8x8.json")

# Run episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    # Simple random policy
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished with reward: {total_reward}")
```

### Example 2: Key-Door Puzzle

```python
from gridworld.task_parser import TaskParser
from gridworld.task_spec import TaskSpecification

# Load task specification
spec = TaskSpecification.from_json("tasks/tier2/key_door_puzzle.json")

# Create parser with rendering for debugging
parser = TaskParser(render_mode="human")

# Parse with specific seed for reproducibility
env = parser.parse(spec, seed=123)

# Environment contains:
# - Keys at specified positions
# - Locked doors matching key colors
# - Agent must collect key, unlock door, reach goal
```

### Example 3: Switch-Gate Mechanism

```python
from gridworld.task_parser import load_task_from_dict

# Programmatically define a task
task_data = {
    "task_id": "custom_switch_gate",
    "seed": 42,
    "difficulty_tier": 3,
    "max_steps": 100,
    "maze": {
        "dimensions": [8, 8],
        "walls": [[3, 3], [3, 4], [3, 5]],
        "start": [1, 1],
        "goal": [6, 6]
    },
    "mechanisms": {
        "switches": [{
            "id": "sw1",
            "position": [2, 4],
            "controls": ["gate1"],
            "switch_type": "toggle"
        }],
        "gates": [{
            "id": "gate1",
            "position": [4, 4],
            "initial_state": "closed"
        }]
    },
    "goal": {
        "type": "reach_position",
        "target": [6, 6]
    }
}

# Load from dictionary
env = load_task_from_dict(task_data, render_mode="rgb_array")

# Agent must toggle switch to open gate, then reach goal
```

### Example 4: Evaluation Loop with Multiple Seeds

```python
from gridworld.task_parser import TaskParser
from gridworld.task_spec import TaskSpecification

# Load task once
spec = TaskSpecification.from_json("task.json")
parser = TaskParser(render_mode="rgb_array")

# Evaluate with multiple seeds
results = []
for seed in range(10):
    env = parser.parse(spec, seed=seed)

    # Run episode
    obs, info = env.reset()
    done = False
    steps = 0
    success = False

    while not done and steps < 100:
        action = my_policy(obs)  # Your agent policy
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
        if terminated and reward > 0:
            success = True

    results.append({
        "seed": seed,
        "success": success,
        "steps": steps
    })

# Analyze results
success_rate = sum(r["success"] for r in results) / len(results)
print(f"Success rate: {success_rate:.1%}")
```

---

## Object Placement Rules

### Walls

- **Type**: Static barriers
- **Placement**: Skip border positions (already have walls from reset)
- **Constraints**: Cannot overlap with start or goal positions (validated by TaskSpecification)

```python
# Walls are added to interior cells only
for wall_pos in spec.maze.walls:
    if 0 < x < width - 1 and 0 < y < height - 1:
        env.place_wall(x, y)
```

### Keys

- **Type**: Collectible items
- **Placement**: Added as pickupable objects on the grid
- **Colors**: "red", "blue", "green", "yellow", "purple", "grey"
- **Mechanics**: Can be picked up and used to unlock matching doors

```python
for key in spec.mechanisms.keys:
    env.place_key(key.position.x, key.position.y, key.color)
```

### Doors

- **Type**: Barriers that require keys to unlock
- **Placement**: Added as locked or unlocked doors
- **Colors**: Must match a key color in the task
- **Mechanics**: Agent with matching key can unlock and open

```python
for door in spec.mechanisms.doors:
    is_locked = door.initial_state == "locked"
    env.place_door(door.position.x, door.position.y,
                   door.requires_key, is_locked)
```

### Gates and Switches

- **Type**: Remote-controlled barriers
- **Placement**: Gates first, then switches (dependency!)
- **Mechanics**: Toggling a switch changes state of all controlled gates
- **Dependency**: Switches reference gate IDs, so gates must exist first

```python
# Place gates first
for gate in spec.mechanisms.gates:
    is_open = gate.initial_state == "open"
    env.place_gate(gate.position.x, gate.position.y, gate.id, is_open)

# Then place switches that control them
for switch in spec.mechanisms.switches:
    env.place_switch(switch.position.x, switch.position.y,
                     switch.id, switch.controls)
```

### Blocks

- **Type**: Pushable objects (Sokoban-style)
- **Placement**: Added as Box objects
- **Mechanics**: Agent can push blocks by moving into them
- **Use Case**: Block puzzles, path creation

```python
for block in spec.mechanisms.blocks:
    env.place_block(block.position.x, block.position.y,
                    block.id, block.color)
```

### Hazards

- **Type**: Dangerous tiles that end the episode
- **Placement**: Added as Lava objects
- **Types**: "lava", "pit", "spike" (all rendered as lava in MiniGrid)
- **Mechanics**: Stepping on a hazard terminates the episode

```python
for hazard in spec.mechanisms.hazards:
    env.place_hazard(hazard.position.x, hazard.position.y,
                     hazard.hazard_type)
```

---

## Validation

The parser validates task specifications before environment creation. Validation catches:

1. **Dimension Checks**: Minimum 3x3 grid size
2. **Bounds Checks**: All positions within grid dimensions
3. **Wall Conflicts**: Start/goal not on walls
4. **Color Consistency**: Doors have matching key colors
5. **ID References**: Switches control valid gate IDs
6. **Tier Validity**: Difficulty tier in range [1, 5]
7. **Max Steps**: Positive step limit

**Example Validation Errors**:

```python
# Task with invalid door (no matching key)
spec = TaskSpecification.from_dict({
    "task_id": "broken",
    "seed": 42,
    "difficulty_tier": 1,
    "max_steps": 100,
    "maze": {
        "dimensions": [8, 8],
        "start": [1, 1],
        "goal": [6, 6],
        "walls": []
    },
    "mechanisms": {
        "doors": [{
            "id": "door1",
            "position": [4, 4],
            "requires_key": "red",  # No red key!
            "initial_state": "locked"
        }],
        "keys": []  # Empty!
    },
    "goal": {"type": "reach_position", "target": [6, 6]}
})

parser = TaskParser()
try:
    env = parser.parse(spec)
except ValueError as e:
    print(e)
    # Output: Invalid task specification: Door door1 requires color 'red'
    #         but no key of that color exists
```

---

## Integration with Backends

The Task Parser is used by backend implementations (MiniGridBackend, MultiGridBackend) to create environments from task specifications.

```python
# Backend usage (simplified)
class MiniGridBackend(AbstractGridBackend):
    def __init__(self, render_mode="rgb_array"):
        self.parser = TaskParser(render_mode=render_mode)

    def configure(self, task_spec: TaskSpecification):
        self.task_spec = task_spec

    def reset(self, seed=None):
        # Parser creates and populates environment
        self.env = self.parser.parse(self.task_spec, seed=seed)
        # Environment is ready to use
        return self.env.render(), self._get_grid_state(), {}
```

---

## Performance Considerations

### Memory Usage

- Each `parse()` call creates a new environment instance
- Environments hold grid state, object references, and render buffers
- For evaluation loops, reuse the parser but create fresh environments per seed

### Computation Time

Parsing is dominated by:
1. **Grid initialization**: O(width × height) to create empty grid
2. **Object placement**: O(num_objects) to place all mechanisms
3. **Validation**: O(num_objects) to check consistency

Typical parse time: **< 10ms** for 8x8 grid with 10-20 objects

### Best Practices

```python
# GOOD: Reuse parser, create fresh environments
parser = TaskParser(render_mode="rgb_array")
for task_file in task_files:
    spec = TaskSpecification.from_json(task_file)
    env = parser.parse(spec)
    # Use environment...
    env.close()

# AVOID: Creating parser per task (unnecessary overhead)
for task_file in task_files:
    parser = TaskParser(render_mode="rgb_array")  # Wasteful!
    env = parser.parse_file(task_file)
    # Use environment...
```

---

## Common Issues and Solutions

### Issue 1: Objects Disappearing After Reset

**Problem**: Objects placed before `reset()` are lost.

**Cause**: MiniGrid's `reset()` method calls `_gen_grid()`, which creates a fresh empty grid.

**Solution**: Always place objects *after* calling `reset()`. The parser handles this correctly.

```python
# WRONG
env = CustomMiniGridEnv(...)
env.place_key(3, 3, "red")  # Placed before reset
env.reset()  # Key is now gone!

# CORRECT (what parser does)
env = CustomMiniGridEnv(...)
env.reset()  # Initialize grid
env.place_key(3, 3, "red")  # Now the key stays
```

### Issue 2: Switch References Invalid Gate

**Problem**: `ValueError` when switch controls non-existent gate.

**Cause**: Gates must exist before switches are placed.

**Solution**: The parser places gates before switches. Ensure your TaskSpecification has matching gate IDs.

```python
# Task spec should have:
"mechanisms": {
    "gates": [{"id": "gate1", ...}],
    "switches": [{"id": "sw1", "controls": ["gate1"], ...}]
}
```

### Issue 3: Agent Spawns in Wrong Position

**Problem**: Agent not at expected start position.

**Cause**: Another object placed at start position.

**Solution**: Parser places agent last to overwrite any conflicts. Check your task specification for position conflicts.

---

## See Also

- [TaskSpecification Schema](../gridworld/task_spec.py): JSON format for tasks
- [CustomMiniGridEnv](../gridworld/custom_env.py): The environment class created by parser
- [MiniGridBackend Documentation](./minigrid_backend.md): Integration with backend system
- [MultiNet Task Generation Guide](../../docs/task_generation.md): Creating evaluation tasks
