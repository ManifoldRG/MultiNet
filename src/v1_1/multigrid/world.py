# multigrid/world.py

"""
World State and Action Execution for MultiGrid

Handles:
- World state management (agent, objects, goals)
- Action execution with full mechanism support
- Object interactions (keys/doors, switches/gates, hazards, teleporters)
"""

from typing import Optional, TYPE_CHECKING
from .agent import AgentState, Action
from .objects.base import WorldObj, ObjectRegistry
from .base import Tiling
from .goals import Goal, create_goal_from_spec
from .visibility import compute_visible_cells

if TYPE_CHECKING:
    from .goals import Goal


class WorldState:
    """Complete world state."""

    def __init__(self, tiling: Tiling):
        self.tiling = tiling
        self.agent = AgentState(cell_id="", facing=0)
        self.objects: dict[str, WorldObj] = {}  # object_id -> WorldObj
        self.goal: Optional[Goal] = None  # Goal predicate
        self.rules: dict = {}  # Game rules (key_consumption, etc.)
        self.hazard_hit: bool = False  # Track if agent hit a hazard

        # Partial observability state
        self.observability_mode: str = "full"  # "full", "view_cone", "fog_of_war"
        self.view_radius: int = 3
        self.visible_cells: set[str] = set()
        self.explored_cells: set[str] = set()

    @classmethod
    def from_task_spec(cls, task_spec: dict, tiling: Tiling, seed: int = 0) -> "WorldState":
        """Create world state from task specification."""
        # Generate tiling graph
        grid_size = task_spec.get("tiling", {}).get("grid_size", {"width": 10, "height": 10})
        tiling.generate_graph(grid_size["width"], grid_size["height"], seed)

        state = cls(tiling)

        # Store rules
        state.rules = task_spec.get("rules", {})

        # Initialize agent
        scene = task_spec.get("scene", {})
        agent_spec = scene.get("agent", {"position": {"x": 0.1, "y": 0.1}})
        agent_pos = agent_spec.get("position", {"x": 0.1, "y": 0.1})
        agent_cell = tiling.canonical_to_cell(agent_pos["x"], agent_pos["y"])
        state.agent = AgentState(
            cell_id=agent_cell,
            facing=agent_spec.get("facing", 0)
        )

        # Initialize objects with type-specific parameters
        for obj_spec in scene.get("objects", []):
            obj = state._create_object_from_spec(obj_spec, tiling)
            if obj:
                state.objects[obj.id] = obj

        # Initialize goal from task spec
        goal_spec = task_spec.get("goal", {})
        if goal_spec:
            state.goal = create_goal_from_spec(goal_spec, tiling)

        # Link switches to gates
        state._link_switches_and_gates()

        # Compute zone covered_cells
        _compute_zone_covered_cells(state, tiling)

        return state

    def _create_object_from_spec(self, obj_spec: dict, tiling: Tiling) -> Optional[WorldObj]:
        """Create an object from specification with type-specific parameters."""
        obj_type = obj_spec.get("type", "movable")
        obj_id = obj_spec["id"]
        color = obj_spec.get("color", "grey")

        # Build kwargs based on object type
        kwargs = {"id": obj_id, "color": color}

        if obj_type == "door":
            kwargs["is_locked"] = obj_spec.get("is_locked", True)

        elif obj_type == "switch":
            kwargs["switch_type"] = obj_spec.get("switch_type", "toggle")
            kwargs["controls"] = obj_spec.get("controls", [])
            kwargs["initial_state"] = obj_spec.get("initial_state", False)

        elif obj_type == "gate":
            kwargs["is_open"] = obj_spec.get("is_open", False)
            kwargs["controlled_by"] = obj_spec.get("controlled_by", [])
            kwargs["require_all"] = obj_spec.get("require_all", False)

        elif obj_type == "hazard":
            kwargs["hazard_type"] = obj_spec.get("hazard_type", "lava")
            kwargs["damage"] = obj_spec.get("damage", 1.0)

        elif obj_type == "teleporter":
            kwargs["linked_to"] = obj_spec.get("linked_to")
            kwargs["cooldown"] = obj_spec.get("cooldown", 1)

        elif obj_type == "zone":
            kwargs["radius_hops"] = obj_spec.get("radius_hops", 1)

        try:
            obj = ObjectRegistry.create(obj_type, **kwargs)
            obj_pos = obj_spec.get("position", {"x": 0.5, "y": 0.5})
            obj.cell_id = tiling.canonical_to_cell(obj_pos["x"], obj_pos["y"])
            return obj
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not create object {obj_id}: {e}")
            return None

    def _link_switches_and_gates(self) -> None:
        """Link switches to their controlled gates."""
        # Build gate lookup
        gates = {obj.id: obj for obj in self.objects.values()
                 if obj.obj_type == "gate"}

        # Link switches to gates
        for obj in self.objects.values():
            if obj.obj_type == "switch":
                for gate_id in obj.controls:
                    if gate_id in gates:
                        gate = gates[gate_id]
                        if obj.id not in gate.controlled_by:
                            gate.controlled_by.append(obj.id)

    def update_visibility(self) -> None:
        """Recompute visible cells based on observability mode."""
        if self.observability_mode == "full":
            self.visible_cells = set(self.tiling.cells.keys())
            self.explored_cells = set(self.tiling.cells.keys())
        else:
            facing = self.agent.facing if self.observability_mode == "view_cone" else None
            self.visible_cells = compute_visible_cells(
                self.agent.cell_id,
                self.tiling,
                self,
                self.view_radius,
                facing=facing,
            )
            self.explored_cells |= self.visible_cells

    def can_move_to(self, cell_id: str) -> bool:
        """Check if agent can move to cell."""
        for obj in self.objects.values():
            if obj.cell_id == cell_id and not obj.can_overlap():
                return False
        return True

    def get_object_at(self, cell_id: str) -> Optional[WorldObj]:
        """Get first non-overlappable object at cell."""
        for obj in self.objects.values():
            if obj.cell_id == cell_id and not obj.can_overlap():
                return obj
        return None

    def get_all_objects_at(self, cell_id: str) -> list[WorldObj]:
        """Get all objects at cell (including overlappable)."""
        return [obj for obj in self.objects.values() if obj.cell_id == cell_id]

    def get_objects_by_type(self, obj_type: str) -> list[WorldObj]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects.values() if obj.obj_type == obj_type]

    def update_gate_states(self) -> None:
        """Update all gate states based on their controlling switches."""
        switches = {obj.id: obj for obj in self.objects.values()
                    if obj.obj_type == "switch"}

        for obj in self.objects.values():
            if obj.obj_type == "gate":
                if not obj.controlled_by:
                    continue

                # Check controlling switches
                active_switches = [
                    switches[sw_id].is_active
                    for sw_id in obj.controlled_by
                    if sw_id in switches
                ]

                if not active_switches:
                    continue

                if obj.require_all:
                    obj.set_open(all(active_switches))
                else:
                    obj.set_open(any(active_switches))

    def check_hazard_collision(self) -> bool:
        """Check if agent is on a hazard."""
        for obj in self.get_all_objects_at(self.agent.cell_id):
            if obj.obj_type == "hazard":
                self.hazard_hit = True
                return True
        return False

    def check_teleporter(self) -> Optional[str]:
        """Check if agent is on a teleporter and should be transported."""
        for obj in self.get_all_objects_at(self.agent.cell_id):
            if obj.obj_type == "teleporter" and obj.can_teleport():
                dest_id = obj.linked_to
                # Find destination teleporter
                if dest_id in self.objects:
                    dest = self.objects[dest_id]
                    if dest.cell_id:
                        obj.use()
                        return dest.cell_id
        return None

    def tick_teleporters(self) -> None:
        """Reduce cooldown on all teleporters."""
        for obj in self.objects.values():
            if obj.obj_type == "teleporter":
                obj.tick()

    def check_goal(self) -> bool:
        """Check if goal is achieved."""
        if self.goal is None:
            return False
        return self.goal.check(self)


def execute_action(
    state: WorldState,
    action: Action,
    tiling: Tiling
) -> tuple[WorldState, bool, dict]:
    """
    Execute action and return (new_state, done, info).

    Handles all mechanism interactions:
    - Keys unlock doors of matching color
    - Switches control gates
    - Hazards terminate the episode
    - Teleporters transport the agent

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

    elif action == Action.BACKWARD:
        facing_dir = agent.get_facing_direction(tiling)
        # Get opposite direction
        facing_idx = tiling.directions.index(facing_dir)
        opposite_idx = (facing_idx + len(tiling.directions) // 2) % len(tiling.directions)
        opposite_dir = tiling.directions[opposite_idx]
        next_cell = tiling.get_neighbor(agent.cell_id, opposite_dir)
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

    elif action == Action.PICKUP:
        if agent.holding is not None:
            info["invalid_action"] = True
        else:
            # Check if there's an object in the agent's cell first
            obj = state.get_object_at(agent.cell_id)

            # If not in agent's cell, check the cell in facing direction
            if not obj:
                facing_dir = agent.get_facing_direction(tiling)
                target_cell = tiling.get_neighbor(agent.cell_id, facing_dir)
                if target_cell:
                    obj = state.get_object_at(target_cell)

            if obj and obj.can_pickup():
                agent.holding = obj
                obj.cell_id = None  # Remove from grid
                state.objects.pop(obj.id, None)  # Remove from objects dict
                info["action_effect"] = "picked_up"
                info["picked_up_type"] = obj.obj_type
            else:
                info["invalid_action"] = True

    elif action == Action.DROP:
        if agent.holding is None:
            info["invalid_action"] = True
        else:
            # Check if current cell is free for dropping
            if state.can_move_to(agent.cell_id):
                # Drop object in current cell
                dropped_obj = agent.holding
                dropped_obj.cell_id = agent.cell_id
                state.objects[dropped_obj.id] = dropped_obj  # Add back to objects dict
                agent.holding = None
                info["action_effect"] = "dropped"
            else:
                # Cannot drop here - cell is occupied
                info["invalid_action"] = True

    elif action == Action.PUSH:
        facing_dir = agent.get_facing_direction(tiling)
        target_cell = tiling.get_neighbor(agent.cell_id, facing_dir)
        if target_cell:
            obj = state.get_object_at(target_cell)
            if obj and obj.can_push():
                push_dest = tiling.get_neighbor(target_cell, facing_dir)
                # Validate push destination
                if push_dest is not None and state.can_move_to(push_dest):
                    obj.cell_id = push_dest
                    info["action_effect"] = "pushed"
                    info["pushed_to"] = push_dest
                else:
                    info["invalid_action"] = True
                    info["reason"] = "push_destination_blocked"
            else:
                info["invalid_action"] = True
                info["reason"] = "nothing_to_push" if not obj else "object_not_pushable"
        else:
            info["invalid_action"] = True
            info["reason"] = "no_target_cell"

    elif action == Action.TOGGLE:
        # Toggle interacts with doors (unlock) and switches (activate)
        facing_dir = agent.get_facing_direction(tiling)
        target_cell = tiling.get_neighbor(agent.cell_id, facing_dir)

        toggled = False

        if target_cell:
            # Check for door
            for obj in state.get_all_objects_at(target_cell):
                if obj.obj_type == "door":
                    if obj.is_locked:
                        # Try to unlock with held key
                        if agent.holding and agent.holding.obj_type == "key":
                            if agent.holding.color == obj.color:
                                obj.unlock()
                                info["action_effect"] = "unlocked_door"
                                info["door_id"] = obj.id
                                toggled = True

                                # Consume key if rules say so
                                if state.rules.get("key_consumption", True):
                                    agent.holding.used = True
                                    agent.holding = None
                                break
                    else:
                        # Toggle open/closed
                        obj.toggle()
                        info["action_effect"] = "toggled_door"
                        info["door_open"] = obj.is_open
                        toggled = True
                        break

                elif obj.obj_type == "switch":
                    if obj.activate():
                        info["action_effect"] = "activated_switch"
                        info["switch_id"] = obj.id
                        info["switch_active"] = obj.is_active
                        toggled = True
                        # Update gate states
                        state.update_gate_states()
                        break

        # Also check current cell for switches (step-on activation)
        if not toggled:
            for obj in state.get_all_objects_at(agent.cell_id):
                if obj.obj_type == "switch":
                    if obj.activate():
                        info["action_effect"] = "activated_switch"
                        info["switch_id"] = obj.id
                        info["switch_active"] = obj.is_active
                        toggled = True
                        state.update_gate_states()
                        break

        if not toggled:
            info["invalid_action"] = True
            info["reason"] = "nothing_to_toggle"

    elif action == Action.WAIT:
        info["action_effect"] = "waited"

    # Post-action processing

    # Check for hold-type switches (deactivate if agent left)
    _update_hold_switches(state)

    # Update gate states
    state.update_gate_states()

    # Tick teleporter cooldowns
    state.tick_teleporters()

    # Check for teleporter transport
    teleport_dest = state.check_teleporter()
    if teleport_dest:
        agent.cell_id = teleport_dest
        info["teleported_to"] = teleport_dest

    # Check for hazard collision
    if state.check_hazard_collision():
        info["hazard_hit"] = True
        return state, True, info  # Episode terminates on hazard

    # Check goal
    done = state.check_goal()

    return state, done, info


def _bfs_zone(tiling: Tiling, center_cell_id: str, radius: int) -> set[str]:
    """
    BFS from center cell up to radius hops. Returns set of cell IDs within radius.

    No blocking — zones expand freely through the tiling graph.
    """
    covered = {center_cell_id}
    if radius <= 0:
        return covered

    frontier = [(center_cell_id, 0)]
    while frontier:
        next_frontier = []
        for cell_id, hops in frontier:
            if hops >= radius:
                continue
            cell = tiling.cells.get(cell_id)
            if cell is None:
                continue
            for neighbor_id in cell.neighbors.values():
                if neighbor_id not in covered:
                    covered.add(neighbor_id)
                    next_frontier.append((neighbor_id, hops + 1))
        frontier = next_frontier

    return covered


def _compute_zone_covered_cells(state: WorldState, tiling: Tiling) -> None:
    """Compute covered_cells for every zone object in the world."""
    for obj in state.objects.values():
        if obj.obj_type == "zone" and obj.cell_id:
            obj.covered_cells = _bfs_zone(tiling, obj.cell_id, obj.radius_hops)


def _update_hold_switches(state: WorldState) -> None:
    """Update hold-type switches based on agent position."""
    for obj in state.objects.values():
        if obj.obj_type == "switch" and obj.switch_type == "hold":
            if obj.cell_id == state.agent.cell_id:
                # Agent is on switch - activate
                if not obj.is_active:
                    obj.activate()
            else:
                # Agent left switch - deactivate
                obj.deactivate()
