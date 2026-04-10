"""
Import-time bootstrap helpers for the gridworld package.

This module keeps third-party environment plugin discovery from pulling in
optional stacks such as `shimmy` / `mujoco_py` during ordinary MiniGrid use.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata


def disable_gymnasium_env_plugins() -> None:
    """Prevent Gymnasium from auto-loading external environment plugins."""
    original = importlib_metadata.entry_points
    if getattr(original, "_multinet_filtered", False):
        return

    def filtered_entry_points(*args, **kwargs):
        group = kwargs.get("group")

        if group is None and args:
            group = args[0]

        if group == "gymnasium.envs":
            return ()

        return original(*args, **kwargs)

    setattr(filtered_entry_points, "_multinet_filtered", True)
    importlib_metadata.entry_points = filtered_entry_points
