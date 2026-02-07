"""
Pre-configured MiniGrid Environments by Tier

Provides convenient access to environments organized by difficulty tier.
"""

from .tier_envs import (
    get_tier1_envs,
    get_tier2_envs,
    get_tier3_envs,
    get_tier4_envs,
    get_tier5_envs,
    get_all_envs,
    get_env_by_id,
    list_available_envs,
)

__all__ = [
    "get_tier1_envs",
    "get_tier2_envs",
    "get_tier3_envs",
    "get_tier4_envs",
    "get_tier5_envs",
    "get_all_envs",
    "get_env_by_id",
    "list_available_envs",
]
