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

    # Domain 
    x0: float 
    y0: float 
    L1: float
    L2: float
    noelms1: int
    noelms2: int
    noelms: int = None 
    nonodes: int = None
    nonodes1: int = None 
    nonodes2: int = None 

    # Everything ordered in columns major with nodes 
    VX: np.ndarray = None # x-coordinates 
    VY: np.ndarray = None # y-coordinates 
    

    EToV: np.ndarray = None # Element to vertex mapping. Ordered counterclockwise, 3 dims 


    # Geometric information

    def __post_init__(self):
        # Compute number of nodes
        self.nonodes1 = self.noelms1 + 1
        self.nonodes2 = self.noelms2 + 1
        self.nonodes = self.nonodes1 * self.nonodes2
        self.noelms = self.noelms1 * self.noelms2 * 2  # 2 triangles per quad

        # Create coordinate arrays (L1, L2 are lengths, not endpoints)
        temp_x = np.linspace(self.x0, self.x0 + self.L1, self.nonodes1)
        temp_y = np.linspace(self.y0, self.y0 + self.L2, self.nonodes2)

        # meshgrid creates 2D arrays
        XX, YY = np.meshgrid(temp_x, temp_y)

        # Flatten in column-major order (F = Fortran order)
        self.VX = XX.flatten(order="F")
        self.VY = np.flip(YY.flatten(order="F"))

        # Build EToV mapping for triangular mesh
        self.EToV = np.zeros((self.noelms, 3), dtype=np.int64)

        elem_idx = 0
        for col in range(self.noelms1):
            for row in range(self.noelms2):
                # Node indices (0-based internally)
                UL = row + col * self.nonodes2          # Upper-left
                LL = row + 1 + col * self.nonodes2      # Lower-left
                UR = row + (col + 1) * self.nonodes2    # Upper-right
                LR = row + 1 + (col + 1) * self.nonodes2  # Lower-right

                # Triangle 1: UL -> LR -> UR (counterclockwise) - 1-based indexing
                self.EToV[elem_idx] = [UL + 1, LR + 1, UR + 1]
                elem_idx += 1

                # Triangle 2: LL -> LR -> UL (counterclockwise) - 1-based indexing
                self.EToV[elem_idx] = [LL + 1, LR + 1, UL + 1]
                elem_idx += 1



