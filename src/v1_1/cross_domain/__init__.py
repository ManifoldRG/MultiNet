"""
Cross-Domain Interface for MultiNet v1.1

Provides canonical task specification and domain adapter abstractions
for evaluating models across different action domains (GridWorld, Physics, NL, GUI).
"""

from .canonical_task_spec import CanonicalTaskSpec, CanonicalGoal, CanonicalObject
from .domain_adapter import DomainAdapter, GUIAction

__all__ = [
    "CanonicalTaskSpec",
    "CanonicalGoal",
    "CanonicalObject",
    "DomainAdapter",
    "GUIAction",
]
