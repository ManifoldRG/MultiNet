# MiniGrid Task Framework Documentation

This directory contains comprehensive documentation for the MiniGrid task specification and evaluation framework used in MultiNet.

## Quick Navigation

### Core Components

1. **[Task Parser](./task_parser.md)** - Transforms JSON task specifications into executable environments
2. **[MiniGrid Backend](./minigrid_backend.md)** - Production-ready square grid backend (recommended)
3. **[MultiGrid Backend](./multigrid_backend.md)** - Experimental backend supporting exotic tilings (hex, triangle)

## Overview

The MiniGrid framework provides a complete pipeline for defining, parsing, and evaluating agents on gridworld navigation and puzzle-solving tasks.

```
┌─────────────────────────────────────────────────────────┐
│              Complete Framework Architecture             │
└─────────────────────────────────────────────────────────┘

JSON Task Specification
    │
    ├─ maze: dimensions, walls, start, goal
    ├─ mechanisms: keys, doors, switches, gates, blocks, hazards
    ├─ rules: key consumption, switch types
    └─ goal: reach_position, collect_all, push_block_to
    │
    ▼
TaskSpecification (Python object)
    │
    ▼
TaskParser
    │
    ├─ Validate specification
    ├─ Create CustomMiniGridEnv
    └─ Populate grid with objects
    │
    ▼
Backend (MiniGrid or MultiGrid)
    │
    ├─ configure(task_spec)
    ├─ reset(seed) → observation, state
    ├─ step(action) → observation, reward, terminated, truncated, state
    └─ render() → RGB image
    │
    ▼
Evaluation / Agent Training
```

## Getting Started

### Basic Usage

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

# 1. Load task specification
spec = TaskSpecification.from_json("path/to/task.json")

# 2. Create and configure backend
backend = MiniGridBackend(render_mode="rgb_array")
backend.configure(spec)

# 3. Run episode
obs, state, info = backend.reset(seed=42)
done = False

while not done:
    action = my_policy(obs)  # Your agent
    obs, reward, terminated, truncated, state, info = backend.step(action)
    done = terminated or truncated

# 4. Check results
print(f"Success: {state.goal_reached}")
print(f"Steps: {state.step_count}")
```

### Quick Examples

#### Navigation Task
```python
# Simple navigation from start to goal
from gridworld.task_parser import load_task_from_file

env = load_task_from_file("tasks/tier1/navigation_8x8.json")
obs, info = env.reset()
# ... run episode
```

#### Key-Door Puzzle
```python
# Task requiring key collection and door unlocking
spec = TaskSpecification.from_json("tasks/tier2/key_door_puzzle.json")
backend = MiniGridBackend()
backend.configure(spec)

obs, state, info = backend.reset()
# Agent must: find key → pickup key → unlock door → reach goal
```

#### Switch-Gate Mechanism
```python
# Task with remote-controlled barriers
spec = TaskSpecification.from_json("tasks/tier3/switch_gate.json")
backend = MiniGridBackend()
backend.configure(spec)

obs, state, info = backend.reset()
# Agent must: find switch → toggle switch → pass through gate → reach goal
```

## Documentation Structure

### Task Parser Documentation (`task_parser.md`)

**Topics Covered**:
- Architecture and design philosophy
- Three-phase parsing (validate, create, populate)
- Object placement order and dependencies
- Usage examples and common patterns
- Integration with backends
- Performance considerations
- Troubleshooting guide

**Key Sections**:
- Why reset() is called inside the parser
- Object placement rules (gates before switches!)
- Validation constraints
- Convenience functions

**Best For**: Understanding how JSON tasks become runnable environments

### MiniGrid Backend Documentation (`minigrid_backend.md`)

**Topics Covered**:
- Backend abstraction layer
- GridState extraction
- Complete API reference
- Action space (0-6 actions)
- Reward structure
- Feature support matrix
- Performance benchmarks

**Key Sections**:
- Why we don't call env.reset() in backend.reset()
- GridState extraction algorithm
- Multi-seed evaluation patterns
- Mechanism state tracking
- Video recording

**Best For**: Production evaluation setup, understanding backend interface

### MultiGrid Backend Documentation (`multigrid_backend.md`)

**Topics Covered**:
- Exotic tiling support (hex, triangle)
- Coordinate system translation (integer ↔ normalized)
- Task specification conversion
- Action space translation
- Feature limitations
- Cross-backend comparison

**Key Sections**:
- Why normalize coordinates?
- Object type unification
- Square vs hex vs triangle comparison
- Known limitations and workarounds
- Future enhancements

**Best For**: Research on spatial topology, exotic grid experiments

## Task Specification Format

Tasks are defined in JSON format with the following structure:

```json
{
  "task_id": "unique_identifier",
  "seed": 42,
  "difficulty_tier": 2,
  "max_steps": 100,
  "description": "Human-readable description",

  "maze": {
    "dimensions": [8, 8],
    "start": [1, 1],
    "goal": [6, 6],
    "walls": [[3, 3], [3, 4], [4, 3]]
  },

  "mechanisms": {
    "keys": [
      {"id": "key1", "position": [2, 2], "color": "red"}
    ],
    "doors": [
      {"id": "door1", "position": [4, 4],
       "requires_key": "red", "initial_state": "locked"}
    ],
    "switches": [
      {"id": "sw1", "position": [2, 5],
       "controls": ["gate1"], "switch_type": "toggle"}
    ],
    "gates": [
      {"id": "gate1", "position": [5, 5], "initial_state": "closed"}
    ],
    "blocks": [
      {"id": "block1", "position": [3, 5], "color": "grey"}
    ],
    "hazards": [
      {"id": "lava1", "position": [4, 6], "hazard_type": "lava"}
    ]
  },

  "rules": {
    "key_consumption": true,
    "switch_type": "toggle"
  },

  "goal": {
    "type": "reach_position",
    "target": [6, 6]
  }
}
```

See individual documentation files for detailed schema definitions.

## Difficulty Tiers

Tasks are organized into 5 difficulty tiers based on complexity:

| Tier | Name | Features | Example |
|------|------|----------|---------|
| 1 | Navigation | Basic pathfinding | Empty maze, shortest path |
| 2 | Linear Dependencies | Sequential tasks | Collect key → unlock door → reach goal |
| 3 | Multi-Mechanism | Parallel mechanisms | Multiple keys, switches, gates |
| 4 | Irreversibility | One-way actions | One-shot switches, consumed keys |
| 5 | Hidden Information | Partial observability | Hidden keys, memory requirements |

## Backend Comparison

| Feature | MiniGrid Backend | MultiGrid Backend |
|---------|------------------|-------------------|
| **Status** | Production-ready | Experimental |
| **Tilings** | Square only | Square, hex, triangle |
| **Performance** | Fast (~400ms/episode) | Slower (~600-900ms/episode) |
| **Mechanisms** | Full support | Limited (keys/walls only) |
| **Rendering** | High quality | Experimental |
| **Partial Obs** | Supported | Not yet |
| **Use Case** | Standard evaluation | Research on exotic tilings |

**Recommendation**: Use **MiniGrid Backend** for production evaluation. Use **MultiGrid Backend** only for research requiring non-square tilings.

## Common Patterns

### Pattern 1: Multi-Seed Evaluation

```python
def evaluate_with_seeds(backend, task_spec, num_seeds=10):
    backend.configure(task_spec)
    results = []

    for seed in range(num_seeds):
        obs, state, info = backend.reset(seed=seed)
        # ... run episode
        results.append({"seed": seed, "success": state.goal_reached})

    return results
```

### Pattern 2: Task Suite Evaluation

```python
def evaluate_task_suite(backend, task_dir):
    results = {}

    for task_file in Path(task_dir).glob("*.json"):
        spec = TaskSpecification.from_json(task_file)
        backend.configure(spec)
        # ... run evaluation
        results[spec.task_id] = metrics

    return results
```

### Pattern 3: Observation Collection

```python
def collect_dataset(backend, task_spec, num_episodes=100):
    backend.configure(task_spec)
    dataset = []

    for episode_id in range(num_episodes):
        obs, state, info = backend.reset(seed=episode_id)
        trajectory = {"observations": [obs], "actions": [], "rewards": []}

        done = False
        while not done:
            action = expert_policy(obs)
            obs, reward, terminated, truncated, state, info = backend.step(action)

            trajectory["observations"].append(obs)
            trajectory["actions"].append(action)
            trajectory["rewards"].append(reward)
            done = terminated or truncated

        dataset.append(trajectory)

    return dataset
```

## Performance Tips

### 1. Reuse Parser and Backend
```python
# GOOD: Reuse instances
parser = TaskParser()
backend = MiniGridBackend()

for task_file in task_files:
    spec = TaskSpecification.from_json(task_file)
    backend.configure(spec)
    # ... evaluate

# AVOID: Creating new instances each time
for task_file in task_files:
    parser = TaskParser()  # Wasteful!
    backend = MiniGridBackend()  # Wasteful!
    # ...
```

### 2. Choose Appropriate Render Mode
```python
# For headless evaluation
backend = MiniGridBackend(render_mode="rgb_array")

# For interactive debugging
backend = MiniGridBackend(render_mode="human")

# For fastest execution (no visuals needed)
backend = MiniGridBackend(render_mode=None)
```

### 3. Close Environments
```python
# Always close when done
try:
    backend.reset()
    # ... run episodes
finally:
    backend.close()  # Cleanup resources
```

## Troubleshooting

### Common Issues

1. **RuntimeError: Backend must be configured before reset**
   - Solution: Call `backend.configure(spec)` before `backend.reset()`

2. **Objects not appearing in environment**
   - Check task JSON has mechanisms defined
   - Validate spec: `spec.validate()`

3. **Switch references non-existent gate**
   - Ensure gate IDs in task spec match switch.controls

4. **Agent spawns in wrong position**
   - Check for position conflicts in task spec
   - Parser places agent last to handle conflicts

5. **Unexpected reward values**
   - Check if agent stepped on hazard (reward=0, terminated=True)
   - vs reaching goal (reward>0, terminated=True)

See individual documentation files for detailed troubleshooting guides.

## API Quick Reference

### TaskParser
- `TaskParser(render_mode=None)`: Create parser
- `.parse(spec, seed=None)`: Parse TaskSpecification → environment
- `.parse_file(path)`: Load and parse JSON file
- `.parse_dict(data)`: Parse dictionary

### Backend Interface (MiniGrid and MultiGrid)
- `.__init__(...)`: Initialize backend
- `.configure(task_spec)`: Set task to use
- `.reset(seed=None)`: Reset to initial state
- `.step(action)`: Execute action
- `.render()`: Get RGB image
- `.get_mission_text()`: Get goal description
- `.get_state()`: Get GridState
- `.close()`: Cleanup

### TaskSpecification
- `.from_json(path)`: Load from file
- `.from_dict(data)`: Load from dictionary
- `.validate()`: Check consistency
- `.to_json(path)`: Save to file
- `.get_mission_text()`: Generate description

## File Locations

```
src/v1_1/
├── gridworld/
│   ├── task_spec.py              # TaskSpecification schema
│   ├── task_parser.py            # Parser implementation
│   ├── custom_env.py             # CustomMiniGridEnv
│   └── backends/
│       ├── base.py               # AbstractGridBackend interface
│       ├── minigrid_backend.py   # MiniGrid implementation
│       └── multigrid_backend.py  # MultiGrid implementation
│
├── multigrid/                    # Custom MultiGrid environment
│   └── env.py
│
└── docs/                         # This directory
    ├── README.md                 # This file
    ├── task_parser.md            # Task Parser docs
    ├── minigrid_backend.md       # MiniGrid Backend docs
    └── multigrid_backend.md      # MultiGrid Backend docs
```

## Related Resources

### Code Files
- `gridworld/task_spec.py`: Complete TaskSpecification schema with validation
- `gridworld/custom_env.py`: Custom MiniGrid environment with all mechanisms
- `gridworld/backends/base.py`: Backend interface and GridState definition

### Example Tasks
- `tasks/tier1/`: Navigation tasks
- `tasks/tier2/`: Key-door puzzles
- `tasks/tier3/`: Switch-gate mechanisms
- `tasks/tier4/`: Irreversible actions
- `tasks/tier5/`: Hidden information

### Evaluation Scripts
- `scripts/eval_minigrid.py`: Evaluation runner
- `scripts/generate_tasks.py`: Task generation utilities

## Contributing

When adding new features to the framework:

1. **Update inline documentation**: Add comprehensive docstrings and comments
2. **Update markdown docs**: Reflect changes in relevant .md files
3. **Add examples**: Include usage examples in documentation
4. **Update comparison tables**: Keep feature matrices current
5. **Note limitations**: Document known issues and workarounds

## Version History

- **v1.1**: Current version
  - MiniGrid Backend: Production-ready
  - MultiGrid Backend: Experimental
  - Full mechanism support in MiniGrid
  - Comprehensive documentation

- **v1.0**: Initial release
  - Basic task specification
  - MiniGrid backend only
  - Limited documentation

## Contact and Support

For issues, questions, or contributions:
- See main MultiNet repository README
- Check individual documentation files for detailed troubleshooting
- Review inline code comments for implementation details

---

**Last Updated**: 2026-01-30

**Documentation Status**: Complete and ready for production use
