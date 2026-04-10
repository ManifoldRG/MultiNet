# MultiGrid Core Architecture Specification

**Version:** 1.0-draft
**Date:** 2026-01-20
**Status:** Implementation-Ready (Square, Triangle, Hex); Reference (Exotic)

## 1. Overview

### 1.1 Purpose

MultiGrid is a tiling-agnostic grid environment framework for the Cross-Action Domain Multimodal Game/Puzzle benchmark. It serves as Domain 1 (Discrete Actions) in the MultiNet v1.1 evaluation system.

### 1.2 Design Goals

1. **Contamination Resistance**: Avoid square-grid patterns saturated in AI training data (MiniGrid, NetHack, Pokémon, etc.) by supporting hexagonal, triangular, and exotic tilings
2. **Novel Spatial Reasoning**: Different connectivity patterns require genuinely new navigation strategies, not memorized movement patterns
3. **Cross-Domain Compatibility**: Share canonical task specifications with physics, NL, and GUI domains
4. **Extensibility**: Support arbitrary tilings including semi-regular Archimedean and aperiodic (Penrose) tilings
5. **Gymnasium Compatibility**: Full compatibility with RL training libraries (stable-baselines3, RLlib)

### 1.3 Why Not MiniGrid?

Assessment of [MiniGrid](https://github.com/Farama-Foundation/Minigrid) revealed deep square-grid assumptions:
- Grid stored as flattened 1D array with `j * width + i` indexing
- Movement/visibility uses hardcoded orthogonal directions
- No abstraction layer for geometry
- Refactoring would require rewriting core classes

**Recommendation**: Build custom implementation using adjacency graph architecture.

### 1.4 Implementation Progression

| Phase | Tilings | Status |
|-------|---------|--------|
| 1 | Square | PoC, baseline compatibility |
| 2 | Triangle, Hexagon | Novel mechanics, regular tilings |
| 3 | 3-4-6-4, other Archimedean | Semi-regular tilings |
| 4 | Penrose, custom | Aperiodic and arbitrary tilings |

---

## 2. Core Architecture

### 2.1 Adjacency Graph Foundation

The core data structure is an **adjacency graph** where:
- **Nodes** represent cells (tiles) in the world
- **Edges** represent valid movement connections between cells
- **Node attributes** store cell contents, position metadata, and rendering hints
- **Edge attributes** store movement direction labels

This enables:
- Any tiling topology without coordinate system changes
- Efficient pathfinding using standard graph algorithms
- Clean separation between topology and rendering

```python
@dataclass
class Cell:
    """A single cell in the grid."""
    id: str                          # Unique identifier (e.g., "cell_0_0")
    neighbors: dict[str, str]        # direction -> neighbor_cell_id
    contents: WorldObj | None        # Object occupying this cell
    position_hint: tuple[float, float]  # Rendering position (normalized 0-1)
    tiling_coords: Any               # Tiling-specific coordinates (for math)

class TilingGraph:
    """Adjacency graph representing the world topology."""
    cells: dict[str, Cell]           # cell_id -> Cell
    boundary_cells: set[str]         # IDs of cells at world boundary
    directions: list[str]            # Valid direction labels for this tiling
```

### 2.2 Tiling Abstraction

Each tiling type implements the `Tiling` interface:

```python
from abc import ABC, abstractmethod

class Tiling(ABC):
    """Abstract base for all tiling types."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tiling identifier (e.g., 'square', 'hex', 'triangle')."""
        pass

    @property
    @abstractmethod
    def directions(self) -> list[str]:
        """List of valid movement directions."""
        pass

    @abstractmethod
    def generate_graph(self, width: int, height: int, seed: int) -> TilingGraph:
        """Generate the adjacency graph for a world of given size."""
        pass

    @abstractmethod
    def canonical_to_cell(self, x: float, y: float) -> str:
        """Convert normalized [0,1] coordinates to cell ID."""
        pass

    @abstractmethod
    def cell_to_canonical(self, cell_id: str) -> tuple[float, float]:
        """Convert cell ID to normalized [0,1] coordinates."""
        pass

    @abstractmethod
    def get_neighbor(self, cell_id: str, direction: str) -> str | None:
        """Get neighbor cell ID in given direction, or None if blocked/boundary."""
        pass

    @abstractmethod
    def distance(self, cell_a: str, cell_b: str) -> int:
        """Compute graph distance (hops) between two cells."""
        pass

    @abstractmethod
    def render_cell(self, cell: Cell, renderer: Renderer) -> None:
        """Render a single cell using the provided renderer."""
        pass
```

### 2.3 Canonical Task Specification

Tasks are defined in a domain-agnostic JSON format shared across all four domains:

```json
{
  "task_id": "move_red_cube_001",
  "version": "1.0",
  "seed": 42,

  "scene": {
    "bounds": {"width": 1.0, "height": 1.0},
    "objects": [
      {
        "id": "cube_red",
        "type": "movable",
        "shape": "cube",
        "color": "red",
        "position": {"x": 0.2, "y": 0.3},
        "size": 0.1
      },
      {
        "id": "zone_blue",
        "type": "zone",
        "shape": "circle",
        "color": "blue",
        "position": {"x": 0.8, "y": 0.7},
        "radius_hops": 2
      }
    ],
    "agent": {
      "position": {"x": 0.1, "y": 0.1},
      "facing": 0
    },
    "walls": [
      {"from": {"x": 0.4, "y": 0.0}, "to": {"x": 0.4, "y": 0.5}}
    ],
    "distractors": {
      "count": 3,
      "types": ["cube", "sphere"],
      "colors": ["green", "yellow"],
      "position_variance": 0.1
    }
  },

  "goal": {
    "predicate": "object_in_zone",
    "object_id": "cube_red",
    "zone_id": "zone_blue",
    "consecutive_steps": 1
  },

  "limits": {
    "max_steps": 100,
    "time_limit_seconds": null
  },

  "tiling": {
    "type": "hex",
    "grid_size": {"width": 12, "height": 10}
  }
}
```

**Coordinate System**: All positions use normalized [0,1] coordinates. Each domain maps these to its native representation:
- GridWorld: Discretizes to cell IDs based on tiling
- Physics: Scales to pixel coordinates
- GUI: Scales to screen coordinates

**Zone Representation**: Zones use `center + radius_hops`. The shape emerges from the tiling topology (hexagonal zones on hex grids, square zones on square grids, etc.).

### 2.4 Template System for Procedural Generation

Tasks are generated from templates with randomization ranges:

```json
{
  "template_id": "move_object_to_zone",
  "version": "1.0",

  "scene_template": {
    "objects": [
      {
        "id": "target_object",
        "type": "movable",
        "shape": {"choices": ["cube", "sphere", "pyramid"]},
        "color": {"choices": ["red", "blue", "green"]},
        "position": {"x": {"min": 0.1, "max": 0.4}, "y": {"min": 0.1, "max": 0.4}},
        "size": 0.1
      },
      {
        "id": "goal_zone",
        "type": "zone",
        "color": {"different_from": "target_object.color"},
        "position": {"x": {"min": 0.6, "max": 0.9}, "y": {"min": 0.6, "max": 0.9}},
        "radius_hops": {"min": 1, "max": 3}
      }
    ],
    "agent": {
      "position": {"x": {"min": 0.0, "max": 0.3}, "y": {"min": 0.0, "max": 0.3}}
    },
    "distractors": {
      "count": {"min": 0, "max": 5}
    }
  },

  "goal_template": {
    "predicate": "object_in_zone",
    "object_id": "target_object",
    "zone_id": "goal_zone"
  }
}
```

---

## 3. Object System

### 3.1 Extensible Object Registry

Objects are defined through a registry pattern allowing new types without core changes:

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

class WorldObj(ABC):
    """Base class for all objects in the world."""

    def __init__(self, id: str, color: str):
        self.id = id
        self.color = color
        self.cell_id: str | None = None  # Current location

    @property
    @abstractmethod
    def obj_type(self) -> str:
        """Object type identifier."""
        pass

    @abstractmethod
    def can_overlap(self) -> bool:
        """Whether agent/objects can occupy same cell."""
        pass

    @abstractmethod
    def can_pickup(self) -> bool:
        """Whether agent can pick this up."""
        pass

    @abstractmethod
    def can_push(self) -> bool:
        """Whether agent can push this."""
        pass


class ObjectRegistry:
    """Registry for object types."""
    _types: dict[str, type[WorldObj]] = {}

    @classmethod
    def register(cls, obj_type: str):
        """Decorator to register an object type."""
        def decorator(obj_class: type[WorldObj]):
            cls._types[obj_type] = obj_class
            return obj_class
        return decorator

    @classmethod
    def create(cls, obj_type: str, **kwargs) -> WorldObj:
        """Factory method to create objects."""
        if obj_type not in cls._types:
            raise ValueError(f"Unknown object type: {obj_type}")
        return cls._types[obj_type](**kwargs)


# Built-in object types
@ObjectRegistry.register("movable")
class MovableObj(WorldObj):
    obj_type = "movable"
    def can_overlap(self) -> bool: return False
    def can_pickup(self) -> bool: return True
    def can_push(self) -> bool: return True


@ObjectRegistry.register("wall")
class Wall(WorldObj):
    obj_type = "wall"
    def can_overlap(self) -> bool: return False
    def can_pickup(self) -> bool: return False
    def can_push(self) -> bool: return False


@ObjectRegistry.register("zone")
class Zone(WorldObj):
    """Target zone - agent and objects can occupy."""
    obj_type = "zone"

    def __init__(self, id: str, color: str, radius_hops: int):
        super().__init__(id, color)
        self.radius_hops = radius_hops
        self.covered_cells: set[str] = set()  # Computed from tiling

    def can_overlap(self) -> bool: return True
    def can_pickup(self) -> bool: return False
    def can_push(self) -> bool: return False
```

### 3.2 Physics Interface Stubs

Physics properties are defined but not implemented (for future Domain 2 integration):

```python
@dataclass
class PhysicsProperties:
    """Physics properties for objects (stubbed for future implementation)."""
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.0  # Bounciness

    # Future: momentum, force accumulation, etc.


class WorldObj(ABC):
    # ... existing methods ...

    def get_physics(self) -> PhysicsProperties:
        """Get physics properties. Override in subclasses for custom behavior."""
        return PhysicsProperties()
```

---

## 4. Agent and Actions

### 4.1 Agent State

```python
@dataclass
class AgentState:
    """Complete agent state."""
    cell_id: str                    # Current cell
    facing: int                     # Direction index (0 to num_directions-1)
    holding: WorldObj | None        # Picked up object

    def get_facing_direction(self, tiling: Tiling) -> str:
        """Get direction label agent is facing."""
        return tiling.directions[self.facing]
```

### 4.2 Action Space

Actions are context-sensitive with facing state:

```python
from enum import IntEnum

class Action(IntEnum):
    """Discrete action space."""
    # Movement
    FORWARD = 0       # Move in facing direction
    BACKWARD = 1      # Move opposite to facing direction

    # Rotation
    TURN_LEFT = 2     # Rotate facing counter-clockwise
    TURN_RIGHT = 3    # Rotate facing clockwise

    # Object interaction
    PICKUP = 4        # Pick up object in facing cell
    DROP = 5          # Drop held object in facing cell
    PUSH = 6          # Push object in facing direction

    # No-op
    WAIT = 7


def get_action_space_size(tiling: Tiling) -> int:
    """Action space is fixed regardless of tiling."""
    return len(Action)
```

**Push Semantics**: Push moves the object in the direction the agent is facing. On hex grids, this means 6 possible push directions corresponding to 6 facing states.

### 4.3 Action Execution

```python
def execute_action(
    state: WorldState,
    action: Action,
    tiling: Tiling
) -> tuple[WorldState, bool, dict]:
    """
    Execute action and return (new_state, done, info).

    Returns:
        new_state: Updated world state
        done: Whether episode terminated
        info: Additional information (success, invalid_action, etc.)
    """
    agent = state.agent
    info = {"invalid_action": False, "action_effect": None}

    if action == Action.FORWARD:
        facing_dir = agent.get_facing_direction(tiling)
        next_cell = tiling.get_neighbor(agent.cell_id, facing_dir)
        if next_cell and state.can_move_to(next_cell):
            agent.cell_id = next_cell
            info["action_effect"] = "moved"
        else:
            info["invalid_action"] = True

    elif action == Action.TURN_LEFT:
        num_dirs = len(tiling.directions)
        agent.facing = (agent.facing - 1) % num_dirs
        info["action_effect"] = "turned"

    elif action == Action.TURN_RIGHT:
        num_dirs = len(tiling.directions)
        agent.facing = (agent.facing + 1) % num_dirs
        info["action_effect"] = "turned"

    elif action == Action.PUSH:
        facing_dir = agent.get_facing_direction(tiling)
        target_cell = tiling.get_neighbor(agent.cell_id, facing_dir)
        if target_cell:
            obj = state.get_object_at(target_cell)
            if obj and obj.can_push():
                push_dest = tiling.get_neighbor(target_cell, facing_dir)
                if push_dest and state.can_move_to(push_dest):
                    obj.cell_id = push_dest
                    info["action_effect"] = "pushed"
                else:
                    info["invalid_action"] = True
            else:
                info["invalid_action"] = True
        else:
            info["invalid_action"] = True

    # ... handle other actions ...

    # Check goal
    done = state.check_goal()

    return state, done, info
```

---

## 5. Gymnasium API

### 5.1 Environment Class

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiGridEnv(gym.Env):
    """
    MultiGrid environment with arbitrary tiling support.

    Inherits from gymnasium.Env for full RL library compatibility.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "state_dict"],
        "render_fps": 10,
    }

    def __init__(
        self,
        task_spec: dict | str,           # Task spec dict or path to JSON
        tiling: str | Tiling = "square", # Tiling type or instance
        render_mode: str | None = None,
        render_style: str = "minimal",   # "minimal" or "sprite"
        partial_obs: bool = False,       # Partial observability
        obs_radius: int = 3,             # Vision radius if partial_obs
    ):
        super().__init__()

        # Load task spec
        if isinstance(task_spec, str):
            with open(task_spec) as f:
                task_spec = json.load(f)
        self.task_spec = task_spec

        # Initialize tiling
        if isinstance(tiling, str):
            self.tiling = TilingRegistry.get(tiling)
        else:
            self.tiling = tiling

        self.render_mode = render_mode
        self.render_style = render_style
        self.partial_obs = partial_obs
        self.obs_radius = obs_radius

        # Define action space
        self.action_space = spaces.Discrete(len(Action))

        # Define observation space
        # RGB image observation
        self._obs_shape = self._compute_obs_shape()
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=self._obs_shape,
            dtype=np.uint8
        )

        # State tracking
        self.state: WorldState | None = None
        self.steps: int = 0
        self.renderer: Renderer | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Use task spec seed if not overridden
        actual_seed = seed if seed is not None else self.task_spec.get("seed", 0)

        # Generate world from task spec
        self.state = WorldState.from_task_spec(
            self.task_spec,
            self.tiling,
            seed=actual_seed
        )
        self.steps = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        assert self.state is not None, "Call reset() before step()"

        # Execute action
        self.state, done, action_info = execute_action(
            self.state,
            Action(action),
            self.tiling
        )
        self.steps += 1

        # Compute reward
        reward = self._compute_reward(done, action_info)

        # Check termination conditions
        terminated = done  # Goal achieved
        truncated = self.steps >= self.task_spec["limits"]["max_steps"]

        obs = self._get_obs()
        info = self._get_info()
        info.update(action_info)

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_human()
            return None
        elif self.render_mode == "state_dict":
            return self.get_state_dict()

    def get_state_dict(self) -> dict:
        """Export full state as structured dict for cross-domain verification."""
        return {
            "agent": {
                "cell_id": self.state.agent.cell_id,
                "facing": self.state.agent.facing,
                "facing_direction": self.state.agent.get_facing_direction(self.tiling),
                "holding": self.state.agent.holding.id if self.state.agent.holding else None,
                "position_canonical": self.tiling.cell_to_canonical(self.state.agent.cell_id)
            },
            "objects": {
                obj.id: {
                    "type": obj.obj_type,
                    "cell_id": obj.cell_id,
                    "position_canonical": self.tiling.cell_to_canonical(obj.cell_id) if obj.cell_id else None,
                    "color": obj.color
                }
                for obj in self.state.objects.values()
            },
            "step": self.steps,
            "goal_achieved": self.state.check_goal()
        }

    def _get_obs(self) -> np.ndarray:
        """Get observation based on observability mode."""
        if self.partial_obs:
            return self._render_partial_obs()
        else:
            return self._render_frame()

    def _compute_reward(self, done: bool, action_info: dict) -> float:
        """Compute reward signal."""
        if done:
            return 1.0  # Goal achieved
        elif action_info.get("invalid_action"):
            return -0.01  # Small penalty for invalid actions
        else:
            return 0.0  # Neutral
```

### 5.2 Configurable Observation Modes

```python
class MultiGridEnv(gym.Env):
    # ... existing code ...

    def set_observation_mode(self, mode: str):
        """
        Switch observation mode at runtime.

        Modes:
            - "rgb": Full RGB pixel rendering
            - "rgb_partial": RGB with partial observability
            - "structured": State dict (for debugging/verification)
            - "symbolic": One-hot encoded cell contents
        """
        self._obs_mode = mode
        self._update_observation_space()
```

---

## 6. Rendering System

### 6.1 Renderer Interface

```python
class Renderer(ABC):
    """Abstract renderer supporting multiple visual styles."""

    @abstractmethod
    def begin_frame(self, width: int, height: int) -> None:
        """Start a new frame."""
        pass

    @abstractmethod
    def draw_cell_background(
        self,
        vertices: list[tuple[float, float]],
        color: tuple[int, int, int]
    ) -> None:
        """Draw cell polygon background."""
        pass

    @abstractmethod
    def draw_object(
        self,
        center: tuple[float, float],
        obj: WorldObj,
        size: float
    ) -> None:
        """Draw an object at given position."""
        pass

    @abstractmethod
    def draw_agent(
        self,
        center: tuple[float, float],
        facing: float,  # Angle in radians
        size: float
    ) -> None:
        """Draw the agent."""
        pass

    @abstractmethod
    def end_frame(self) -> np.ndarray:
        """Finish frame and return RGB array."""
        pass


class MinimalRenderer(Renderer):
    """Clean vector-based rendering for VLM evaluation."""
    pass


class SpriteRenderer(Renderer):
    """Textured sprite-based rendering for visual complexity testing."""
    pass
```

### 6.2 Visual Difficulty Axis

Rendering complexity can be configured to test VLM robustness:

```python
@dataclass
class RenderConfig:
    """Configuration for visual complexity."""
    style: str = "minimal"           # "minimal", "sprite", "noisy"

    # Minimal style options
    cell_outline: bool = True
    object_labels: bool = False

    # Complexity additions
    background_noise: float = 0.0    # 0-1 noise level
    color_jitter: float = 0.0        # 0-1 color variation
    rotation_jitter: float = 0.0     # Random rotation (radians)

    # Sprite style options
    sprite_set: str = "default"
    antialiasing: bool = True
```

---

## 7. Success Criteria and Scoring

### 7.1 Goal Predicates

```python
class GoalPredicate(ABC):
    """Abstract goal predicate."""

    @abstractmethod
    def check(self, state: WorldState) -> bool:
        """Check if goal is satisfied."""
        pass

    @abstractmethod
    def get_progress(self, state: WorldState) -> float:
        """Get progress toward goal (0-1) for auxiliary metrics."""
        pass


class ObjectInZone(GoalPredicate):
    """Goal: object center in zone for N consecutive steps."""

    def __init__(self, object_id: str, zone_id: str, consecutive_steps: int = 1):
        self.object_id = object_id
        self.zone_id = zone_id
        self.consecutive_steps = consecutive_steps
        self._steps_in_zone = 0

    def check(self, state: WorldState) -> bool:
        obj = state.objects[self.object_id]
        zone = state.objects[self.zone_id]

        if obj.cell_id in zone.covered_cells:
            self._steps_in_zone += 1
        else:
            self._steps_in_zone = 0

        return self._steps_in_zone >= self.consecutive_steps

    def get_progress(self, state: WorldState) -> float:
        obj = state.objects[self.object_id]
        zone = state.objects[self.zone_id]

        # Distance-based progress
        obj_pos = state.tiling.cell_to_canonical(obj.cell_id)
        zone_pos = state.tiling.cell_to_canonical(zone.cell_id)

        max_dist = 1.414  # Diagonal of unit square
        current_dist = ((obj_pos[0] - zone_pos[0])**2 + (obj_pos[1] - zone_pos[1])**2)**0.5

        return 1.0 - (current_dist / max_dist)
```

### 7.2 Multi-Metric Scoring

```python
@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    # Binary
    success: bool

    # Auxiliary
    steps_taken: int
    optimal_steps: int | None      # If computed
    efficiency: float | None       # steps_taken / optimal_steps
    invalid_actions: int
    goal_progress: float           # 0-1 progress at episode end
    time_in_zone: int              # Steps object spent in goal zone


def compute_episode_metrics(
    episode_log: list[dict],
    goal: GoalPredicate,
    optimal_solution: list[Action] | None = None
) -> EpisodeMetrics:
    """Compute all metrics from episode log."""
    # ... implementation ...
```

---

## 8. Natural Language Domain Integration

### 8.1 NL Wrapper Architecture

```python
class NLGridWorldWrapper:
    """
    Wrapper that accepts natural language commands and executes on GridWorld.

    Implements the same observation/action interface but with string actions.
    """

    def __init__(self, env: MultiGridEnv, parser: CommandParser):
        self.env = env
        self.parser = parser

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset underlying environment."""
        return self.env.reset(**kwargs)

    def step(self, nl_command: str) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Parse NL command and execute on GridWorld.

        Args:
            nl_command: Natural language command like "move north" or "push the red cube"

        Returns:
            Standard gymnasium step outputs
        """
        action, parse_info = self.parser.parse(nl_command, self.env.state)

        if action is None:
            # Unparseable command
            obs = self.env._get_obs()
            info = {"parse_error": True, "raw_command": nl_command}
            return obs, -0.1, False, False, info

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["parsed_action"] = action.name
        info["raw_command"] = nl_command

        return obs, reward, terminated, truncated, info
```

### 8.2 Command Parser

```python
import re

class CommandParser:
    """
    Parse natural language commands to discrete actions.

    Uses strict grammar with regex for MVP. Can be extended with
    semantic parsing for more flexibility.
    """

    # Grammar patterns
    PATTERNS = {
        # Movement
        r"move\s+(north|south|east|west|forward|backward)": "_parse_move",
        r"go\s+(north|south|east|west|forward|backward)": "_parse_move",
        r"turn\s+(left|right)": "_parse_turn",
        r"rotate\s+(left|right|clockwise|counter-?clockwise)": "_parse_turn",

        # Object interaction
        r"pick\s*up(\s+the)?(\s+\w+)?(\s+\w+)?": "_parse_pickup",
        r"grab(\s+the)?(\s+\w+)?(\s+\w+)?": "_parse_pickup",
        r"drop(\s+the)?(\s+\w+)?": "_parse_drop",
        r"push(\s+the)?(\s+\w+)?(\s+\w+)?": "_parse_push",

        # Wait
        r"wait|stay|stop": "_parse_wait",
    }

    def parse(self, command: str, state: WorldState) -> tuple[Action | None, dict]:
        """
        Parse command string to Action.

        Returns:
            (Action, info_dict) or (None, error_dict)
        """
        command = command.lower().strip()

        for pattern, handler_name in self.PATTERNS.items():
            match = re.match(pattern, command)
            if match:
                handler = getattr(self, handler_name)
                return handler(match, state)

        return None, {"error": "unrecognized_command", "command": command}

    def _parse_move(self, match: re.Match, state: WorldState) -> tuple[Action, dict]:
        direction = match.group(1)
        if direction in ("forward",):
            return Action.FORWARD, {"direction": "forward"}
        elif direction in ("backward",):
            return Action.BACKWARD, {"direction": "backward"}
        else:
            # Map cardinal to facing + forward
            # This requires turning first - return sequence or just forward
            return Action.FORWARD, {"direction": direction}

    # ... other handlers ...
```

---

## 9. Cross-Domain Verification

### 9.1 State Correspondence Protocol

To verify cross-domain equivalence, states are mapped to a canonical form:

```python
@dataclass
class CanonicalState:
    """Domain-agnostic state representation for cross-domain comparison."""
    agent_position: tuple[float, float]  # Normalized [0,1]
    agent_facing: float                   # Angle in radians
    object_positions: dict[str, tuple[float, float]]  # obj_id -> position
    goal_achieved: bool


def to_canonical(domain_state: Any, domain_type: str) -> CanonicalState:
    """Convert domain-specific state to canonical form."""
    if domain_type == "gridworld":
        return _gridworld_to_canonical(domain_state)
    elif domain_type == "physics":
        return _physics_to_canonical(domain_state)
    # ... etc
```

### 9.2 Equivalence Checking

```python
def check_state_equivalence(
    state_a: CanonicalState,
    state_b: CanonicalState,
    position_tolerance: float = 0.1
) -> tuple[bool, dict]:
    """
    Check if two canonical states are equivalent.

    Returns:
        (is_equivalent, details_dict)
    """
    details = {}

    # Check agent position
    agent_dist = _euclidean_distance(state_a.agent_position, state_b.agent_position)
    details["agent_position_diff"] = agent_dist
    agent_match = agent_dist <= position_tolerance

    # Check object positions
    obj_diffs = {}
    for obj_id in state_a.object_positions:
        if obj_id in state_b.object_positions:
            dist = _euclidean_distance(
                state_a.object_positions[obj_id],
                state_b.object_positions[obj_id]
            )
            obj_diffs[obj_id] = dist
    details["object_position_diffs"] = obj_diffs
    objects_match = all(d <= position_tolerance for d in obj_diffs.values())

    # Check goal
    details["goal_match"] = state_a.goal_achieved == state_b.goal_achieved

    is_equivalent = agent_match and objects_match and details["goal_match"]

    return is_equivalent, details
```

---

## 10. Output Formats

### 10.1 Episode Log (JSON)

```json
{
  "task_id": "move_red_cube_001",
  "tiling": "hex",
  "seed": 42,
  "model_id": "gpt-4o",

  "trajectory": [
    {
      "step": 0,
      "observation": "base64_encoded_image_or_path",
      "state": {
        "agent": {"cell_id": "hex_0_0", "facing": 0},
        "objects": {"cube_red": {"cell_id": "hex_2_3"}}
      }
    },
    {
      "step": 1,
      "action": "FORWARD",
      "action_raw": "move forward",
      "observation": "...",
      "state": {...},
      "reward": 0.0,
      "info": {"invalid_action": false}
    }
  ],

  "metrics": {
    "success": true,
    "steps_taken": 15,
    "optimal_steps": 12,
    "efficiency": 0.8,
    "invalid_actions": 2,
    "goal_progress": 1.0
  }
}
```

---

## 11. Soft Performance Guidelines

- **Target grid sizes**: Up to 50x50 cells (2500 cells) without noticeable latency
- **Step latency**: < 10ms for action execution (excluding rendering)
- **Rendering**: 30+ FPS for human visualization, batch mode for evaluation
- **Memory**: < 100MB per environment instance

---

## 12. Risk Notes

### 12.1 Contamination Concerns

- **Hex grids**: Present in strategy games (Civilization, Settlers of Catan adaptations) - some contamination risk
- **Triangle grids**: Less common but present in some puzzle games
- **Mitigation**: Use exotic Archimedean tilings (3-4-6-4) and visual style variation

### 12.2 Coordinate Discretization

- Normalized [0,1] to cell mapping may cause edge effects
- Different tilings have different cell densities at same resolution
- **Mitigation**: Document mapping algorithms, test boundary conditions

---

## 13. References

- [Red Blob Games: Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/) - Comprehensive hex coordinate math
- [Euclidean tilings by convex regular polygons](https://en.wikipedia.org/wiki/Euclidean_tilings_by_convex_regular_polygons) - Archimedean tiling reference
- [MiniGrid](https://github.com/Farama-Foundation/Minigrid) - Reference for Gymnasium patterns (not extensible for our needs)
- [Griddly](https://github.com/Bam4d/Griddly) - Alternative grid engine (square-only)

---

## Appendices

See companion documents:
- [Appendix A: Square Tiling](appendix_square.md)
- [Appendix B: Hexagonal Tiling](appendix_hex.md)
- [Appendix C: Triangular Tiling](appendix_triangle.md)
- [Appendix D: Exotic Tilings](appendix_exotic.md)
- [Appendix E: Test Cases and Walkthroughs](test_cases.md)
