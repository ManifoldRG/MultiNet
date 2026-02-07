# MiniGrid Backend Documentation

## Overview

The MiniGrid Backend is a production-ready implementation of the `AbstractGridBackend` interface that wraps the gymnasium MiniGrid package. It provides a stable, well-tested foundation for evaluating agents on gridworld navigation and puzzle-solving tasks.

**Purpose**: Enable evaluation of vision-language-action models on standard square-grid environments with comprehensive mechanism support (keys, doors, switches, gates, blocks, hazards).

**Location**: `/src/v1_1/gridworld/backends/minigrid_backend.py`

**Status**: MVP (Minimum Viable Product) - Production ready

---

## Architecture

### Backend Abstraction Layer

The MiniGrid Backend implements the `AbstractGridBackend` interface, which defines a standard API that all grid environment backends must support. This abstraction allows:

- **Backend Swapping**: Switch between MiniGrid and MultiGrid (or future backends) without changing evaluation code
- **Consistent API**: Same methods and return types across all backends
- **Backend-Agnostic State**: GridState representation works with any backend

```
┌───────────────────────────────────────────────────────────┐
│              Backend Abstraction Architecture              │
└───────────────────────────────────────────────────────────┘

    TaskSpecification (JSON)
            │
            ▼
    ┌──────────────────┐
    │AbstractGridBackend│ ◄─── Common interface
    └────────┬──────────┘
         ┌───┴────┐
         ▼        ▼
    ┌─────────┐ ┌──────────────┐
    │MiniGrid │ │  MultiGrid   │
    │Backend  │ │  Backend     │
    │(This)   │ │(Exotic tiles)│
    └────┬────┘ └──────────────┘
         │
         ├──► TaskParser (creates env from spec)
         │
         ├──► CustomMiniGridEnv (gymnasium-based)
         │
         └──► GridState (backend-agnostic state)
```

### Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│              MiniGrid Backend Workflow                   │
└─────────────────────────────────────────────────────────┘

1. CONFIGURATION
   backend.configure(task_spec)
       │
       └──► Store task_spec for later use
            Set _configured = True

2. RESET
   backend.reset(seed=42)
       │
       ├──► parser.parse(task_spec, seed)
       │      │
       │      ├──► Create CustomMiniGridEnv
       │      ├──► env.reset() [initializes grid]
       │      └──► Populate grid with objects
       │
       ├──► env.gen_obs() [symbolic observation]
       ├──► env.render() [RGB image]
       ├──► _get_grid_state() [extract state]
       │
       └──► Return (rgb_obs, state, info)

3. STEP
   backend.step(action)
       │
       ├──► env.step(action) [execute in MiniGrid]
       ├──► env.render() [get new RGB obs]
       ├──► _get_grid_state() [extract new state]
       │
       └──► Return (obs, reward, terminated, truncated, state, info)

4. RENDER
   backend.render()
       │
       └──► env.render() [RGB image of current state]
```

---

## Key Components

### MiniGridBackend Class

```python
class MiniGridBackend(AbstractGridBackend):
    """
    Backend implementation using gymnasium's MiniGrid package.
    """

    def __init__(self, render_mode: Optional[str] = "rgb_array")
    def configure(self, task_spec: TaskSpecification) -> None
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, GridState, dict]
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, GridState, dict]
    def render(self) -> np.ndarray
    def get_mission_text(self) -> str
    def get_state(self) -> GridState
    def close(self) -> None
```

### Constructor: `__init__(render_mode)`

**Parameters**:
- `render_mode` (str, optional): Rendering mode for the environment
  - `"rgb_array"`: Returns RGB numpy arrays (recommended for evaluation)
  - `"human"`: Opens a window for visualization (for debugging)
  - `None`: Minimal rendering (fastest)

**Default**: `"rgb_array"`

**Example**:
```python
from gridworld.backends import MiniGridBackend

# Production evaluation setup
backend = MiniGridBackend(render_mode="rgb_array")

# Interactive debugging
backend = MiniGridBackend(render_mode="human")
```

**Initialization Details**:
- Creates a `TaskParser` instance with the specified render mode
- Initializes `self.env` to None (environment created on reset)
- Sets up observation caching (`_last_obs`)

### Method: `configure(task_spec)`

Configures the backend with a task specification. This is the first method that must be called.

**Parameters**:
- `task_spec` (TaskSpecification): The task definition to use

**Returns**: None

**Side Effects**:
- Stores `task_spec` for use in `reset()`
- Sets `_configured` flag to True

**Example**:
```python
from gridworld.task_spec import TaskSpecification
from gridworld.backends import MiniGridBackend

# Load task specification
spec = TaskSpecification.from_json("task.json")

# Configure backend
backend = MiniGridBackend()
backend.configure(spec)

# Now ready for reset()
```

**Design Note**: Configuration is separate from reset to allow:
1. Pre-validation of task specs before environment creation
2. Reusing the same backend with different tasks
3. Lazy environment creation (only on reset)

### Method: `reset(seed=None)`

Resets the environment to its initial state and returns the starting observation.

**Parameters**:
- `seed` (int, optional): Random seed for reproducibility. If None, uses `task_spec.seed`

**Returns**:
- `observation` (np.ndarray): RGB image of initial state, shape (H, W, 3)
- `state` (GridState): Backend-agnostic state representation
- `info` (dict): Additional information (currently empty)

**Raises**:
- `RuntimeError`: If `configure()` has not been called

**Example**:
```python
# Reset with task's default seed
obs, state, info = backend.reset()

# Reset with specific seed for evaluation
obs, state, info = backend.reset(seed=42)

print(f"Observation shape: {obs.shape}")
print(f"Agent at: {state.agent_position}")
print(f"Agent facing: {state.agent_direction}")
```

**Critical Implementation Detail - Why We Don't Call env.reset() Here**:

The `reset()` method uses `parser.parse()` to create a fresh environment. The parser internally calls `env.reset()` to initialize the grid, then populates it with objects. **We must NOT call `env.reset()` again** in the backend's `reset()` method because:

1. It would wipe out all placed objects (keys, doors, switches, etc.)
2. The grid would be empty except for border walls
3. The task would be unplayable

This is a deliberate architectural choice:
- **TaskParser responsibility**: Create + reset + populate
- **Backend responsibility**: Trigger parser + extract observations

### Method: `step(action)`

Executes one action in the environment and returns the result.

**Parameters**:
- `action` (int): Action to execute (0-6)
  - 0: Turn left
  - 1: Turn right
  - 2: Move forward
  - 3: Pickup object
  - 4: Drop object
  - 5: Toggle/interact
  - 6: Done/wait

**Returns**:
- `observation` (np.ndarray): RGB image of new state
- `reward` (float): Reward for this step
- `terminated` (bool): True if episode ended (goal reached or failure)
- `truncated` (bool): True if episode cut short (max steps reached)
- `state` (GridState): New backend-agnostic state
- `info` (dict): Additional information from environment

**Raises**:
- `RuntimeError`: If `reset()` has not been called

**Example**:
```python
# Execute forward action
obs, reward, terminated, truncated, state, info = backend.step(2)

if terminated:
    if reward > 0:
        print("Goal reached!")
    else:
        print("Episode failed (e.g., stepped on lava)")

if truncated:
    print("Max steps reached without solving")

# Check if agent is carrying something
if state.agent_carrying:
    print(f"Agent holding: {state.agent_carrying}")

# Check mechanism states
print(f"Active switches: {state.active_switches}")
print(f"Open gates: {state.open_gates}")
```

**Reward Structure**:

MiniGrid uses a time-penalized reward:
```python
reward = 1.0 - 0.9 * (step_count / max_steps)
```

- **Goal reached immediately**: reward = 1.0
- **Goal reached at 50% steps**: reward = 0.55
- **Goal reached at max steps**: reward = 0.1
- **Failed or truncated**: reward = 0

This encourages efficient solutions.

### Method: `render()`

Returns an RGB rendering of the current environment state.

**Returns**:
- `np.ndarray`: RGB image, shape (H, W, 3), dtype uint8

**Example**:
```python
import matplotlib.pyplot as plt

# Get current rendering
rgb_image = backend.render()

# Display
plt.imshow(rgb_image)
plt.title("Current Environment State")
plt.axis('off')
plt.show()
```

**Behavior**:
- If `render_mode="rgb_array"`, calls `env.render()`
- If other render mode, returns cached `_last_obs`
- If no observations yet, returns black placeholder

### Method: `get_mission_text()`

Returns the mission/goal description for the current task.

**Returns**:
- `str`: Human-readable mission description

**Example**:
```python
mission = backend.get_mission_text()
print(mission)
# Output: "Navigate to the goal. Keys: 2. Locked doors: 2."
```

**Text Sources** (in order of priority):
1. Environment's mission text (if environment exists)
2. Task spec's mission text (if task configured)
3. Default text: "Navigate to the goal"

### Method: `get_state()`

Returns the current environment state as a GridState object.

**Returns**:
- `GridState`: Backend-agnostic state representation

**Example**:
```python
state = backend.get_state()
print(f"Position: {state.agent_position}")
print(f"Direction: {state.agent_direction}")
print(f"Steps: {state.step_count}/{state.max_steps}")
print(f"Goal reached: {state.goal_reached}")
```

### Method: `close()`

Cleans up resources and closes the environment.

**Example**:
```python
# Done with environment
backend.close()
```

**Best Practice**:
```python
try:
    backend.reset()
    # ... run episode ...
finally:
    backend.close()  # Ensure cleanup
```

---

## GridState Extraction

### The `_get_grid_state()` Method

This internal method converts the MiniGrid environment state into a backend-agnostic `GridState` object. This is crucial for evaluation and backend comparison.

**What It Extracts**:

1. **Agent State**:
   - Position: `(x, y)` tuple
   - Direction: Integer 0-3 (right, down, left, up)
   - Carrying: Color of held object or None

2. **Mechanism States**:
   - Active switches: Set of switch IDs currently toggled on
   - Open gates: Set of gate IDs currently passable
   - Block positions: Dict mapping block_id → (x, y)

3. **Episode State**:
   - Step count: Number of steps taken
   - Max steps: Episode step limit
   - Goal reached: Boolean flag

**Performance Consideration**:

Block position extraction requires a full grid scan (O(width × height) per block). For a typical 8×8 grid with 3 blocks, this is ~192 cell checks per step. Acceptable for evaluation but could be optimized with position caching for larger grids or real-time applications.

**Example Output**:
```python
state = backend.get_state()
# GridState(
#     agent_position=(4, 5),
#     agent_direction=2,  # Facing left
#     agent_carrying="red",  # Holding red key
#     step_count=15,
#     max_steps=100,
#     open_doors=set(),
#     collected_keys=set(),
#     active_switches={'sw1'},  # Switch sw1 is active
#     open_gates={'gate1'},  # Gate gate1 is open
#     block_positions={'block1': (3, 3), 'block2': (5, 6)},
#     goal_reached=False
# )
```

---

## Usage Examples

### Example 1: Basic Episode Execution

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

# Load task
spec = TaskSpecification.from_json("tasks/navigation_8x8.json")

# Create and configure backend
backend = MiniGridBackend(render_mode="rgb_array")
backend.configure(spec)

# Run episode
obs, state, info = backend.reset(seed=42)
done = False
total_reward = 0
step_count = 0

while not done:
    # Random policy (replace with your agent)
    action = np.random.randint(0, 7)

    obs, reward, terminated, truncated, state, info = backend.step(action)
    total_reward += reward
    step_count += 1
    done = terminated or truncated

    print(f"Step {step_count}: pos={state.agent_position}, "
          f"reward={reward:.3f}, done={done}")

print(f"\nEpisode finished:")
print(f"  Total reward: {total_reward:.3f}")
print(f"  Steps taken: {step_count}")
print(f"  Success: {state.goal_reached}")

backend.close()
```

### Example 2: Multi-Seed Evaluation

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

def evaluate_policy(policy_fn, task_path, num_seeds=10):
    """
    Evaluate a policy across multiple seeds.
    """
    spec = TaskSpecification.from_json(task_path)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)

    results = []
    for seed in range(num_seeds):
        obs, state, info = backend.reset(seed=seed)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = policy_fn(obs, state)
            obs, reward, terminated, truncated, state, info = backend.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        results.append({
            "seed": seed,
            "success": state.goal_reached,
            "reward": total_reward,
            "steps": steps
        })

    backend.close()

    # Aggregate results
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "per_seed": results
    }

# Example usage
def random_policy(obs, state):
    return np.random.randint(0, 7)

results = evaluate_policy(random_policy, "task.json", num_seeds=10)
print(f"Success rate: {results['success_rate']:.1%}")
```

### Example 3: Observation and State Comparison

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

# Setup
spec = TaskSpecification.from_json("task.json")
backend = MiniGridBackend(render_mode="rgb_array")
backend.configure(spec)

# Reset
obs, state, info = backend.reset(seed=42)

print("Initial State:")
print(f"  RGB observation shape: {obs.shape}")
print(f"  Agent position: {state.agent_position}")
print(f"  Agent direction: {state.agent_direction}")
print(f"  Mission: {backend.get_mission_text()}")

# Take a few actions
for action in [2, 2, 5]:  # Forward, forward, toggle
    obs, reward, terminated, truncated, state, info = backend.step(action)
    print(f"\nAfter action {action}:")
    print(f"  Position: {state.agent_position}")
    print(f"  Carrying: {state.agent_carrying}")
    print(f"  Active switches: {state.active_switches}")
    print(f"  Reward: {reward}")

backend.close()
```

### Example 4: Mechanism State Tracking

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

# Task with switches and gates
spec = TaskSpecification.from_json("tasks/switch_gate_puzzle.json")
backend = MiniGridBackend()
backend.configure(spec)

obs, state, info = backend.reset()

print("Initial mechanism states:")
print(f"  Active switches: {state.active_switches}")
print(f"  Open gates: {state.open_gates}")

# Agent navigates and toggles a switch
# ... execute actions ...

# After toggling switch
state = backend.get_state()
print("\nAfter toggling switch:")
print(f"  Active switches: {state.active_switches}")
print(f"  Open gates: {state.open_gates}")

# Check if gate is now passable
if 'gate1' in state.open_gates:
    print("Gate 1 is now open and passable!")
```

### Example 5: Video Recording

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification
import imageio

# Setup
spec = TaskSpecification.from_json("task.json")
backend = MiniGridBackend(render_mode="rgb_array")
backend.configure(spec)

# Record episode
frames = []
obs, state, info = backend.reset(seed=42)
frames.append(backend.render())

done = False
while not done:
    action = my_policy(obs)
    obs, reward, terminated, truncated, state, info = backend.step(action)
    frames.append(backend.render())
    done = terminated or truncated

backend.close()

# Save video
imageio.mimsave("episode.mp4", frames, fps=4)
print(f"Saved {len(frames)} frames to episode.mp4")
```

---

## Feature Support

### Supported Mechanisms

| Mechanism | Supported | Notes |
|-----------|-----------|-------|
| Walls | ✓ | Static barriers |
| Keys | ✓ | Collectible items, multiple colors |
| Doors | ✓ | Locked/unlocked, require matching key color |
| Switches | ✓ | Toggle, hold, and one-shot types |
| Gates | ✓ | Controlled by switches |
| Blocks | ✓ | Pushable Sokoban-style |
| Hazards | ✓ | Lava (episode-ending) |
| Teleporters | ✗ | Not implemented in MiniGrid |
| Partial Observability | ✓ | Agent has limited field of view |

### Supported Goal Types

| Goal Type | Supported | Description |
|-----------|-----------|-------------|
| Reach Position | ✓ | Navigate to goal position |
| Collect All | Partial | Can collect keys, but goal checking not fully implemented |
| Push Block To | Partial | Blocks are pushable, but goal checking not fully implemented |
| Survive Steps | ✓ | Don't die until max steps |

**Note**: For full multi-goal support, use the goal specification and implement custom win condition checking in your evaluation code.

### Rendering Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `rgb_array` | Returns RGB numpy arrays | Headless evaluation, ML training |
| `human` | Opens visualization window | Interactive debugging |
| `None` | Minimal rendering | Fastest for non-visual evaluation |

**Recommendation**: Use `"rgb_array"` for all evaluation to ensure consistent observations.

---

## Performance Characteristics

### Timing Benchmarks (8×8 grid, typical task)

| Operation | Time | Notes |
|-----------|------|-------|
| configure() | ~0.1 ms | Just stores task spec |
| reset() | ~8-12 ms | Parser + grid population |
| step() | ~2-4 ms | Action execution + state extraction |
| render() | ~3-5 ms | RGB image generation |
| get_state() | ~1-2 ms | GridState extraction |

**Total episode (100 steps)**: ~400-600 ms

### Memory Usage

- **Backend instance**: ~1 KB (just metadata)
- **Environment instance**: ~50-100 KB (grid, objects, render buffer)
- **RGB observation**: ~150 KB for 64×64×3 uint8 image

**Recommendation**: For large-scale evaluation (1000s of episodes), create environments on-demand and close them when done to avoid memory accumulation.

---

## Integration with Evaluation Pipeline

### Standard Evaluation Pattern

```python
from gridworld.backends import MiniGridBackend
from gridworld.task_spec import TaskSpecification

def run_evaluation(agent, task_files, num_seeds=5):
    """
    Standard evaluation loop using MiniGrid backend.
    """
    backend = MiniGridBackend(render_mode="rgb_array")
    results = {}

    for task_file in task_files:
        spec = TaskSpecification.from_json(task_file)
        backend.configure(spec)

        task_results = []
        for seed in range(num_seeds):
            obs, state, info = backend.reset(seed=seed)

            episode_data = {
                "observations": [obs],
                "states": [state.to_dict()],
                "actions": [],
                "rewards": []
            }

            done = False
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, state, info = backend.step(action)

                episode_data["observations"].append(obs)
                episode_data["states"].append(state.to_dict())
                episode_data["actions"].append(action)
                episode_data["rewards"].append(reward)

                done = terminated or truncated

            episode_data["success"] = state.goal_reached
            episode_data["total_reward"] = sum(episode_data["rewards"])
            task_results.append(episode_data)

        results[spec.task_id] = task_results

    backend.close()
    return results
```

---

## Troubleshooting

### Issue 1: RuntimeError on reset()

**Error**: `RuntimeError: Backend must be configured before reset`

**Cause**: Called `reset()` before `configure()`

**Solution**:
```python
# WRONG
backend = MiniGridBackend()
backend.reset()  # Error!

# CORRECT
backend = MiniGridBackend()
backend.configure(task_spec)
backend.reset()  # Works
```

### Issue 2: Objects Not Appearing

**Symptom**: Environment is empty except for walls

**Cause**: Task specification has no mechanisms, or parser error

**Solution**:
1. Check task JSON has mechanisms defined
2. Validate task spec: `spec.validate()`
3. Check parser logs for errors

### Issue 3: Unexpected Reward Values

**Symptom**: Reward is 0 even though goal reached

**Cause**: Stepped on hazard before reaching goal

**Solution**: Check `state.terminated` to distinguish:
- `terminated=True, reward>0`: Goal reached
- `terminated=True, reward=0`: Failed (hazard, etc.)
- `truncated=True, reward=0`: Max steps reached

### Issue 4: GridState Has Wrong Block Positions

**Symptom**: `state.block_positions` is incorrect

**Cause**: Blocks were pushed but state not updated

**Solution**: This is a known limitation. GridState extraction scans the grid, so it should be accurate. If you're seeing errors, check:
1. Are you using a cached state instead of calling `get_state()` after each step?
2. Are multiple blocks at the same position (invalid task)?

---

## Comparison with MultiGrid Backend

| Feature | MiniGridBackend | MultiGridBackend |
|---------|-----------------|------------------|
| **Tilings** | Square only | Square, hex, triangle |
| **Maturity** | Production-ready | Experimental |
| **Performance** | Fast (~400ms/episode) | Slower (~600ms/episode) |
| **Switches/Gates** | Fully supported | Not yet implemented |
| **Partial Observability** | Supported | Not yet implemented |
| **Render Quality** | High (MiniGrid native) | Variable |
| **Use Case** | Standard evaluation | Research on exotic tilings |

**Recommendation**: Use MiniGridBackend for production evaluation. Use MultiGridBackend only for research requiring non-square tilings.

---

## See Also

- [AbstractGridBackend Interface](../gridworld/backends/base.py): Base interface documentation
- [Task Parser Documentation](./task_parser.md): How tasks are parsed into environments
- [MultiGrid Backend Documentation](./multigrid_backend.md): Alternative backend for exotic tilings
- [TaskSpecification Schema](../gridworld/task_spec.py): JSON format for tasks
- [Evaluation Pipeline Guide](../../docs/evaluation.md): End-to-end evaluation setup
