# objects/builtin.py

"""
Built-in Object Types for MultiGrid

Provides all standard object types for gridworld puzzles:
- Movable: Pickable/pushable objects (boxes, balls)
- Wall: Impassable barriers
- Zone: Target areas (overlappable)
- Key: Colored keys for unlocking doors
- Door: Barriers that require matching key to unlock
- Switch: Controls gates (toggle/hold/one-shot modes)
- Gate: Barriers controlled by switches
- Hazard: Dangerous cells that terminate episode
- Teleporter: Linked pairs that transport agent
"""

from typing import Optional, Literal
from .base import WorldObj, ObjectRegistry


@ObjectRegistry.register("movable")
class MovableObj(WorldObj):
    """Movable object (can be picked up or pushed)."""

    @property
    def obj_type(self) -> str:
        return "movable"

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return True

    def can_push(self) -> bool:
        return True


@ObjectRegistry.register("wall")
class Wall(WorldObj):
    """Wall object (blocks movement)."""

    @property
    def obj_type(self) -> str:
        return "wall"

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False


@ObjectRegistry.register("zone")
class Zone(WorldObj):
    """Target zone - agent and objects can occupy."""

    def __init__(self, id: str, color: str, radius_hops: int = 1):
        super().__init__(id, color)
        self.radius_hops = radius_hops
        self.covered_cells: set[str] = set()  # Computed from tiling

    @property
    def obj_type(self) -> str:
        return "zone"

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False


@ObjectRegistry.register("key")
class Key(WorldObj):
    """
    Key object for unlocking doors.

    Keys can be picked up and used to unlock doors of matching color.
    Depending on rules.key_consumption, keys may be consumed on use.
    """

    def __init__(self, id: str, color: str):
        super().__init__(id, color)
        self.used: bool = False  # Track if key has been used

    @property
    def obj_type(self) -> str:
        return "key"

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return True

    def can_push(self) -> bool:
        return False


@ObjectRegistry.register("door")
class Door(WorldObj):
    """
    Door object that blocks movement until unlocked.

    Doors require a key of matching color to unlock. Once unlocked,
    the door becomes passable (can_overlap returns True).

    Attributes:
        is_locked: Whether the door is currently locked
        is_open: Whether the door is open (unlocked and toggled open)
    """

    def __init__(self, id: str, color: str, is_locked: bool = True):
        super().__init__(id, color)
        self.is_locked = is_locked
        self.is_open = not is_locked  # Unlocked doors start open

    @property
    def obj_type(self) -> str:
        return "door"

    def can_overlap(self) -> bool:
        # Can pass through if unlocked and open
        return self.is_open

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False

    def unlock(self) -> bool:
        """Unlock the door. Returns True if successfully unlocked."""
        if self.is_locked:
            self.is_locked = False
            self.is_open = True
            return True
        return False

    def toggle(self) -> None:
        """Toggle door open/closed (only works if unlocked)."""
        if not self.is_locked:
            self.is_open = not self.is_open


@ObjectRegistry.register("switch")
class Switch(WorldObj):
    """
    Switch that controls one or more gates.

    Switch types:
    - toggle: Each activation flips the state
    - hold: Active only while agent is on the switch
    - one_shot: Can only be activated once

    Attributes:
        switch_type: Type of switch behavior
        is_active: Current switch state
        controls: List of gate IDs this switch controls
        used: Whether one_shot switch has been used
    """

    def __init__(
        self,
        id: str,
        color: str,
        switch_type: Literal["toggle", "hold", "one_shot"] = "toggle",
        controls: Optional[list[str]] = None,
        initial_state: bool = False
    ):
        super().__init__(id, color)
        self.switch_type = switch_type
        self.is_active = initial_state
        self.controls = controls or []
        self.used = False  # For one_shot switches

    @property
    def obj_type(self) -> str:
        return "switch"

    def can_overlap(self) -> bool:
        # Agent can stand on switches
        return True

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False

    def activate(self) -> bool:
        """
        Activate the switch.

        Returns True if state changed.
        """
        if self.switch_type == "one_shot":
            if self.used:
                return False
            self.used = True
            self.is_active = True
            return True
        elif self.switch_type == "toggle":
            self.is_active = not self.is_active
            return True
        elif self.switch_type == "hold":
            if not self.is_active:
                self.is_active = True
                return True
            return False
        return False

    def deactivate(self) -> bool:
        """
        Deactivate the switch (for hold type when agent leaves).

        Returns True if state changed.
        """
        if self.switch_type == "hold" and self.is_active:
            self.is_active = False
            return True
        return False


@ObjectRegistry.register("gate")
class Gate(WorldObj):
    """
    Gate that opens/closes based on switch state.

    Gates are controlled by switches. When the controlling switch(es)
    are active, the gate opens (becomes passable).

    Attributes:
        is_open: Whether the gate is currently open
        controlled_by: List of switch IDs that control this gate
        require_all: If True, all switches must be active; if False, any one
    """

    def __init__(
        self,
        id: str,
        color: str,
        is_open: bool = False,
        controlled_by: Optional[list[str]] = None,
        require_all: bool = False
    ):
        super().__init__(id, color)
        self.is_open = is_open
        self.controlled_by = controlled_by or []
        self.require_all = require_all

    @property
    def obj_type(self) -> str:
        return "gate"

    def can_overlap(self) -> bool:
        return self.is_open

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False

    def set_open(self, is_open: bool) -> None:
        """Set gate open/closed state."""
        self.is_open = is_open


@ObjectRegistry.register("hazard")
class Hazard(WorldObj):
    """
    Hazardous cell that terminates the episode.

    When the agent steps on a hazard, the episode ends with failure.
    Common examples: lava, spikes, pits.

    Attributes:
        hazard_type: Type of hazard (for rendering)
        damage: Damage dealt (for future health system)
    """

    def __init__(
        self,
        id: str,
        color: str = "red",
        hazard_type: str = "lava",
        damage: float = 1.0
    ):
        super().__init__(id, color)
        self.hazard_type = hazard_type
        self.damage = damage

    @property
    def obj_type(self) -> str:
        return "hazard"

    def can_overlap(self) -> bool:
        # Agent can step on hazards (but will be damaged/killed)
        return True

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False


@ObjectRegistry.register("teleporter")
class Teleporter(WorldObj):
    """
    Teleporter that transports agent to linked destination.

    Teleporters come in pairs. When agent steps on one, they are
    transported to the linked teleporter.

    Attributes:
        linked_to: ID of the destination teleporter
        cooldown: Steps before teleporter can be used again
        current_cooldown: Current cooldown counter
    """

    def __init__(
        self,
        id: str,
        color: str = "purple",
        linked_to: Optional[str] = None,
        cooldown: int = 1
    ):
        super().__init__(id, color)
        self.linked_to = linked_to
        self.cooldown = cooldown
        self.current_cooldown = 0

    @property
    def obj_type(self) -> str:
        return "teleporter"

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return False

    def can_push(self) -> bool:
        return False

    def can_teleport(self) -> bool:
        """Check if teleporter is ready to use."""
        return self.current_cooldown == 0 and self.linked_to is not None

    def use(self) -> None:
        """Use the teleporter, starting cooldown."""
        self.current_cooldown = self.cooldown

    def tick(self) -> None:
        """Reduce cooldown by one step."""
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
