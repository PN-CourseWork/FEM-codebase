from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3

BOUNDARY_TOL = 1e-10

N_VERTICES = 3

EDGE_VERTICES = np.array([[0, 1], [1, 2], [2, 0]])


@dataclass
class Mesh2d:
    """2D triangular mesh for P1 finite elements."""

    x0: float
    y0: float
    L1: float
    L2: float
    noelms1: int
    noelms2: int
    diagonal: str = "nw_se"  # "nw_se" or "sw_ne" - direction of triangle split

    # Computed mesh properties
    noelms: int = field(init=False)
    nonodes: int = field(init=False)
    nonodes1: int = field(init=False)
    nonodes2: int = field(init=False)

    # Mesh arrays
    VX: NDArray[np.float64] = field(init=False)
    VY: NDArray[np.float64] = field(init=False)
    EToV: NDArray[np.int64] = field(init=False)

    # Basis function data
    abc: NDArray[np.float64] = field(init=False)
    delta: NDArray[np.float64] = field(init=False)

    # Boundary data
    boundary_edges: NDArray[np.int64] = field(init=False)
    boundary_sides: NDArray[np.int64] = field(init=False)

    # Internal vertex index arrays (0-based indices into VX, VY)
    _v1: NDArray[np.int64] = field(init=False, repr=False)
    _v2: NDArray[np.int64] = field(init=False, repr=False)
    _v3: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_mesh_properties()
        self._generate_mesh()
        self._compute_vertex_indices()
        self._compute_boundary_edges()
        self._compute_basis()

    def _compute_mesh_properties(self) -> None:
        self.nonodes1 = self.noelms1 + 1
        self.nonodes2 = self.noelms2 + 1
        self.nonodes = self.nonodes1 * self.nonodes2
        self.noelms = self.noelms1 * self.noelms2 * 2

    def _generate_mesh(self) -> None:
        temp_x = np.linspace(self.x0, self.x0 + self.L1, self.nonodes1)
        temp_y = np.linspace(self.y0 + self.L2, self.y0, self.nonodes2)

        XX, YY = np.meshgrid(temp_x, temp_y)
        self.VX = XX.flatten(order="F")
        self.VY = YY.flatten(order="F")

        # Column-major element ordering with upper-first (MATLAB convention)
        col, row = np.meshgrid(np.arange(self.noelms1), np.arange(self.noelms2))
        col, row = col.flatten(order="F"), row.flatten(order="F")

        UL = row + col * self.nonodes2
        LL = UL + 1
        UR = UL + self.nonodes2
        LR = UR + 1

        # Pre-allocate and assign directly 
        self.EToV = np.empty((self.noelms, 3), dtype=np.int64)

        if self.diagonal == "nw_se":
            # Diagonal from upper-left to lower-right
            # Upper triangles first (even indices): [UL, LR, UR]
            self.EToV[0::2, 0] = UL
            self.EToV[0::2, 1] = LR
            self.EToV[0::2, 2] = UR
            # Lower triangles (odd indices): [LL, LR, UL]
            self.EToV[1::2, 0] = LL
            self.EToV[1::2, 1] = LR
            self.EToV[1::2, 2] = UL
        elif self.diagonal == "sw_ne":
            # Diagonal from lower-left to upper-right
            # Left triangles first (even indices): [UL, LL, UR]
            self.EToV[0::2, 0] = UL
            self.EToV[0::2, 1] = LL
            self.EToV[0::2, 2] = UR
            # Right triangles (odd indices): [LL, LR, UR]
            self.EToV[1::2, 0] = LL
            self.EToV[1::2, 1] = LR
            self.EToV[1::2, 2] = UR
        else:
            raise ValueError(f"diagonal must be 'nw_se' or 'sw_ne', got '{self.diagonal}'")

    def _compute_vertex_indices(self) -> None:
        """Cache vertex indices for each element (already 0-based)."""
        self._v1 = self.EToV[:, 0]
        self._v2 = self.EToV[:, 1]
        self._v3 = self.EToV[:, 2]

    def _compute_boundary_edges(self) -> None:
        elems_per_col = 2 * self.noelms2

        if self.diagonal == "nw_se":
            # Upper triangle: [UL, LR, UR] - edge 2=right, edge 3=top
            # Lower triangle: [LL, LR, UL] - edge 1=bottom, edge 3=left

            # Left: 
            left_elems = np.arange(2, elems_per_col + 1, 2)  
            left_edge = 3
            # Right: 
            right_start = (self.noelms1 - 1) * elems_per_col
            right_elems = right_start + np.arange(1, elems_per_col, 2) + 1  
            right_edge = 2
            # Bottom: 
            bottom_elems = np.arange(1, self.noelms1 + 1) * elems_per_col  
            bottom_edge = 1
            # Top: 
            top_elems = np.arange(self.noelms1) * elems_per_col + 1  
            top_edge = 3
        else:  # sw_ne
            # Left triangle: [UL, LL, UR] - edge 1=left, edge 3=top
            # Right triangle: [LL, LR, UR] - edge 1=bottom, edge 2=right

            # Left: 
            left_elems = np.arange(1, elems_per_col, 2)  
            left_edge = 1
            # Right: 
            right_start = (self.noelms1 - 1) * elems_per_col
            right_elems = right_start + np.arange(2, elems_per_col + 1, 2)  
            right_edge = 2
            # Bottom: 
            bottom_elems = np.arange(1, self.noelms1 + 1) * elems_per_col  
            bottom_edge = 1
            # Top: 
            top_elems = np.arange(self.noelms1) * elems_per_col + 1  
            top_edge = 3

        # Pre-allocate boundary arrays
        n_boundary = 2 * (self.noelms1 + self.noelms2)
        self.boundary_edges = np.empty((n_boundary, 2), dtype=np.int64)
        self.boundary_sides = np.empty(n_boundary, dtype=np.int64)

        # Fill boundary edges and sides by section
        i = 0
        # Left side
        n = self.noelms2
        self.boundary_edges[i:i+n, 0] = left_elems
        self.boundary_edges[i:i+n, 1] = left_edge
        self.boundary_sides[i:i+n] = LEFT
        i += n
        # Right side
        self.boundary_edges[i:i+n, 0] = right_elems
        self.boundary_edges[i:i+n, 1] = right_edge
        self.boundary_sides[i:i+n] = RIGHT
        i += n
        # Bottom side
        n = self.noelms1
        self.boundary_edges[i:i+n, 0] = bottom_elems
        self.boundary_edges[i:i+n, 1] = bottom_edge
        self.boundary_sides[i:i+n] = BOTTOM
        i += n
        # Top side
        self.boundary_edges[i:i+n, 0] = top_elems
        self.boundary_edges[i:i+n, 1] = top_edge
        self.boundary_sides[i:i+n] = TOP

    def _compute_basis(self) -> None:
        """Compute delta and basis function coefficients for each element."""
        x1, y1, x2, y2, x3, y3 = self.vertex_coords

        self.delta = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Pre-allocate abc array
        # Shape: (noelms, 3 basis functions, 3 coefficients [a, b, c])
        self.abc = np.empty((self.noelms, 3, 3), dtype=np.float64)
        # Basis function 1: a1, b1, c1
        self.abc[:, 0, 0] = x2 * y3 - x3 * y2
        self.abc[:, 0, 1] = y2 - y3
        self.abc[:, 0, 2] = x3 - x2
        # Basis function 2: a2, b2, c2
        self.abc[:, 1, 0] = x3 * y1 - x1 * y3
        self.abc[:, 1, 1] = y3 - y1
        self.abc[:, 1, 2] = x1 - x3
        # Basis function 3: a3, b3, c3
        self.abc[:, 2, 0] = x1 * y2 - x2 * y1
        self.abc[:, 2, 1] = y1 - y2
        self.abc[:, 2, 2] = x2 - x1

    @property
    def vertex_coords(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Return (x1, y1, x2, y2, x3, y3) coordinates for all elements."""
        return (
            self.VX[self._v1],
            self.VY[self._v1],
            self.VX[self._v2],
            self.VY[self._v2],
            self.VX[self._v3],
            self.VY[self._v3],
        )


def outernormal(
    n: int,
    k: int,
    VX: NDArray[np.float64],
    VY: NDArray[np.float64],
    EToV: NDArray[np.int64],
) -> tuple[float, float]:
    """Compute unit outer normal for element n (0-indexed), edge k (1-indexed)."""
    vertices = EToV[n]
    va, vb = vertices[EDGE_VERTICES[k - 1]]
    xa, ya = VX[va], VY[va]
    xb, yb = VX[vb], VY[vb]
    dx, dy = xb - xa, yb - ya
    length = np.sqrt(dx**2 + dy**2)
    return float(dy / length), float(-dx / length)
