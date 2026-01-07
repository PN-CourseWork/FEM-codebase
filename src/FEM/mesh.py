from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    """
    Finite element mesh.

    Attributes
    ----------
    VX : ndarray (n_nodes,)
        Vertex coordinates
    EToV : ndarray (n_elem, nodes_per_elem)
        Element-to-vertex connectivity
    """

    VX: np.ndarray
    EToV: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.VX)

    @property
    def n_elem(self) -> int:
        return len(self.EToV)


def line_mesh(L: float, n_elem: int) -> Mesh:
    """Create 1D mesh on [0, L]."""
    VX = np.linspace(0, L, n_elem + 1)
    EToV = np.column_stack([np.arange(n_elem), np.arange(1, n_elem + 1)])
    return Mesh(VX, EToV)
