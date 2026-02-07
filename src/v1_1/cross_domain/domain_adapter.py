"""
Domain Adapter Abstract Base Class

Defines the interface for mapping canonical task specifications
to domain-specific environments and back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .canonical_task_spec import CanonicalTaskSpec


@dataclass
class GUIAction:
    """
    Action type for Domain 4 (GUI manipulation) -- forward-looking.

    Designed now to ensure the cross-domain interface supports
    mouse/keyboard GUI interactions from the start.
    """
    action_type: str  # "mouse_click", "mouse_drag", "key_press"
    x: float = 0.0
    y: float = 0.0
    drag_to_x: float = 0.0
    drag_to_y: float = 0.0
    key: str = ""     # For key_press actions


class DomainAdapter(ABC):
    """
    Abstract base class for domain adapters.

    Maps canonical task specs to domain-specific environments
    and provides a Gymnasium-like interface for evaluation.

    Implementations:
    - GridWorldDomainAdapter: MiniGrid/MultiGrid gridworlds
    - PhysicsDomainAdapter (future): Pymunk 2D physics
    - NLDomainAdapter (future): Natural language commands
    - GUIDomainAdapter (future): Pygame GUI manipulation
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Unique domain identifier."""
        ...

    @property
    @abstractmethod
    def action_type(self) -> str:
        """Action type: 'discrete', 'continuous', 'text', 'gui'."""
        ...

    @abstractmethod
    def from_canonical(self, spec: CanonicalTaskSpec) -> Any:
        """
        Convert canonical task spec to domain-specific environment.

        Args:
            spec: Domain-agnostic task specification

        Returns:
            Domain-specific environment or configuration
        """
        ...

    @abstractmethod
    def to_canonical(self, domain_spec: Any) -> CanonicalTaskSpec:
        """
        Convert domain-specific spec to canonical task spec.

        Args:
            domain_spec: Domain-specific task specification

        Returns:
            Canonical task specification
        """
        ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Reset the environment. Returns (observation, info)."""
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action. Returns (obs, reward, terminated, truncated, info)."""
        ...

    @abstractmethod
    def check_success(self) -> bool:
        """Check if the task goal has been achieved."""
        ...

    def render(self) -> Optional[np.ndarray]:
        """Render current state as RGB array."""
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass
