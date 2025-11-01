"""
Magma Model Adapters

This package contains adapters for the Magma model.
"""

from .magma_bfcl_adapter import MagmaBFCLAdapter
from .magma_mcq_adapter import MagmaMCQAdapter
from .magma_overcooked_adapter import MagmaOvercookedAdapter
from .magma_vqa_adapter import MagmaVQAAdapter

__all__ = ['MagmaBFCLAdapter', 'MagmaMCQAdapter', 'MagmaOvercookedAdapter', 'MagmaVQAAdapter']

