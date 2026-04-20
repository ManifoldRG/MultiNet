# objects/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhysicsProperties:
    """Physics properties for objects (stubbed for future implementation)."""
    mass: float = 1.0
    friction: float = 0.5
    restitution: float = 0.0  # Bounciness


class WorldObj(ABC):
    """Base class for all objects in the world."""

    def __init__(self, id: str, color: str):
        self.id = id
        self.color = color
        self.cell_id: Optional[str] = None  # Current location

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

    def get_physics(self) -> PhysicsProperties:
        """Get physics properties. Override in subclasses for custom behavior."""
        return PhysicsProperties()


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
