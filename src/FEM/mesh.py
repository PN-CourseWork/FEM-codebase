import numpy as np
from .datastructures import Mesh


def line_mesh(L: float, n_elem: int) -> Mesh:
    """Create 1D mesh on [0, L]."""
    VX = np.linspace(0, L, n_elem + 1)
    EToV = np.column_stack([np.arange(n_elem), np.arange(1, n_elem + 1)])
    return Mesh(VX=VX, EToV=EToV)
