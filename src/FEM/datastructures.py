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
