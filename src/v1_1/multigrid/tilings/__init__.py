# tilings/__init__.py

from .square import SquareTiling
from .hex import HexTiling
from .triangle import TriangleTiling
from .archimedean_3464 import Archimedean3464Tiling
from .archimedean_488 import Archimedean488Tiling

__all__ = [
    'SquareTiling',
    'HexTiling',
    'TriangleTiling',
    'Archimedean3464Tiling',
    'Archimedean488Tiling',
]
