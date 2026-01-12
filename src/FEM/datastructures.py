from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh:
    """Finite element mesh for 1D P1 elements."""

    VX: np.ndarray
    EToV: np.ndarray

    nonodes: int = None
    noelms: int = None

    def __post_init__(self):
        self.nonodes = len(self.VX)
        self.noelms = len(self.EToV)

    def sorted(self):
        """Return (new_mesh, permutation) with nodes sorted by coordinate."""
        perm = np.argsort(self.VX.ravel())

        inverse = np.empty(self.nonodes, dtype=np.int64)
        inverse[perm] = np.arange(self.nonodes)

        new_VX = self.VX[perm]
        new_cells = inverse[self.EToV]

        return Mesh(
            VX=new_VX,
            EToV=new_cells,
        ), perm



@dataclass
class Mesh2d:
    """Finite element mesh for 2D."""

    VX: np.ndarray
    VY: np.ndarray
    EToV: np.ndarray # Element to vertex mapping. Ordered counterclockwise 

    nonodes: int = None
    noelms1: int = None
    noelms2: int = None

    # Geometric information
    vec_t: np.ndarray   # tangential vector
    vec_n: np.ndarray   # normal vector

    def __post_init__(self):
        # Sort
        self.nonodes = len(self.VX)
        self.noelms1 = len(self.EToV)
        self.noelms2 = len(self.EToV)

