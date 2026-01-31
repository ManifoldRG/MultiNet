# objects/__init__.py

from .base import WorldObj, ObjectRegistry, PhysicsProperties
from .builtin import MovableObj, Wall, Zone

__all__ = ['WorldObj', 'ObjectRegistry', 'PhysicsProperties', 'MovableObj', 'Wall', 'Zone']
