"""
Natural Language Domain (Domain 3) for MultiNet v1.1

Provides NL action parsing, NL environment wrapper, and NL model interface
for evaluating models that produce natural language action commands.
"""

from .nl_action_parser import NLActionParser
from .nl_env import NLGridWorldEnv

__all__ = ["NLActionParser", "NLGridWorldEnv"]
