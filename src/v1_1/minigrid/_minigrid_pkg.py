"""
Helper module to import the gymnasium minigrid package without naming conflicts.

The local minigrid directory shadows the installed minigrid package. This module
provides access to the installed package by loading it directly from disk.

Usage:
    from ._minigrid_pkg import mg_Grid, mg_MiniGridEnv, mg_Key, mg_Door, ...
"""

import sys
import os
import importlib.util

def _load_pkg_module(module_name, pkg_path):
    """Load a module directly from a package path."""
    module_path = os.path.join(pkg_path, module_name.replace(".", "/") + ".py")
    if not os.path.exists(module_path):
        # Try __init__.py for packages
        module_path = os.path.join(pkg_path, module_name.replace(".", "/"), "__init__.py")

    if not os.path.exists(module_path):
        raise ImportError(f"Cannot find module {module_name} at {module_path}")

    spec = importlib.util.spec_from_file_location(f"_gym_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)

    # Handle subpackage imports by setting up parent packages
    parts = module_name.split(".")
    if len(parts) > 1:
        parent_name = ".".join(parts[:-1])
        if f"_gym_{parent_name}" not in sys.modules:
            _load_pkg_module(parent_name, pkg_path)

    spec.loader.exec_module(module)
    return module

# Find the installed minigrid package
_venv_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_site_packages_candidates = [
    os.path.join(_venv_path, ".venv", "lib", "python3.10", "site-packages"),
    "/home/sean/mosaic/lib/python3.10/site-packages",
]

_minigrid_pkg_path = None
for _candidate in _site_packages_candidates:
    _test_path = os.path.join(_candidate, "minigrid")
    if os.path.exists(_test_path) and os.path.isdir(_test_path):
        _minigrid_pkg_path = _candidate
        break

if _minigrid_pkg_path is None:
    raise ImportError(
        "Could not find installed minigrid package. "
        "Please install it with: pip install minigrid"
    )

# Add the site-packages to path temporarily for this import
_old_path = sys.path.copy()
sys.path.insert(0, _minigrid_pkg_path)

# Remove any local minigrid modules from sys.modules
_local_minigrid_mods = [k for k in sys.modules.keys()
                        if k == "minigrid" or k.startswith("minigrid.")]
_saved_mods = {k: sys.modules.pop(k) for k in _local_minigrid_mods}

try:
    # Import the gymnasium minigrid package
    import minigrid as _gym_minigrid
    from minigrid.core.grid import Grid as mg_Grid
    from minigrid.core.mission import MissionSpace as mg_MissionSpace
    from minigrid.core.world_object import (
        WorldObj as mg_WorldObj,
        Key as mg_Key,
        Door as mg_Door,
        Goal as mg_Goal,
        Wall as mg_Wall,
        Lava as mg_Lava,
        Box as mg_Box,
        Ball as mg_Ball,
        COLOR_TO_IDX as mg_COLOR_TO_IDX,
    )
    from minigrid.minigrid_env import MiniGridEnv as mg_MiniGridEnv
finally:
    # Restore sys.path
    sys.path = _old_path

    # Remove gymnasium minigrid from sys.modules so local one can be imported
    gym_mods = [k for k in sys.modules.keys()
                if k == "minigrid" or k.startswith("minigrid.")]
    for mod in gym_mods:
        if mod in sys.modules:
            del sys.modules[mod]

    # Restore local minigrid modules
    sys.modules.update(_saved_mods)
