from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import meshio

# Boundary side constants (Generic)
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3

# Boundary side constants (Cylinder Benchmark)
INLET = 2
OUTLET = 3
WALLS = 4
CYLINDER = 5

# Tolerance for boundary node detection (floating-point comparison)
BOUNDARY_TOL = 1e-10

# Element configuration (P1 triangles)
N_LOCAL_NODES = 3
N_VERTICES = 3

# Edge k (1,2,3) connects these vertex positions in EToV
EDGE_VERTICES = np.array([[0, 1], [1, 2], [2, 0]])


@dataclass
class Mesh2d:
    """2D triangular mesh for P1 finite elements."""

    # Domain parameters (user-provided)
    x0: float
    y0: float
    L1: float
    L2: float
    noelms1: int
    noelms2: int

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

    # Internal vertex index arrays
    _v1: NDArray[np.int64] = field(init=False, repr=False)
    _v2: NDArray[np.int64] = field(init=False, repr=False)
    _v3: NDArray[np.int64] = field(init=False, repr=False)

    # CSR assembly pattern (pre-computed for direct CSR construction)
    _csr_indptr: NDArray[np.int64] = field(init=False, repr=False)
    _csr_indices: NDArray[np.int64] = field(init=False, repr=False)
    _csr_data_map: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_mesh_properties()
        self._generate_mesh()
        self._compute_assembly_indices()
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

        col, row = np.meshgrid(np.arange(self.noelms1), np.arange(self.noelms2))
        col, row = col.flatten(order="F"), row.flatten(order="F")

        UL = row + col * self.nonodes2
        LL = UL + 1
        UR = UL + self.nonodes2
        LR = UR + 1

        # Pre-allocate and assign directly (faster than column_stack)
        self.EToV = np.empty((self.noelms, 3), dtype=np.int64)
        # Upper triangles: [UL, LR, UR]
        self.EToV[0::2, 0] = UL + 1
        self.EToV[0::2, 1] = LR + 1
        self.EToV[0::2, 2] = UR + 1
        # Lower triangles: [LL, LR, UL]
        self.EToV[1::2, 0] = LL + 1
        self.EToV[1::2, 1] = LR + 1
        self.EToV[1::2, 2] = UL + 1

    def _compute_assembly_indices(self) -> None:
        """Compute vertex indices and CSR sparsity pattern for direct assembly. """
        self._v1 = self.EToV[:, 0] - 1
        self._v2 = self.EToV[:, 1] - 1
        self._v3 = self.EToV[:, 2] - 1

        # Build (row, col) pairs for all element matrix entries
        # For P1: 9 entries per element (3x3 local matrix)
        nodes = np.empty((self.noelms, N_LOCAL_NODES), dtype=np.int64)
        nodes[:, 0] = self._v1
        nodes[:, 1] = self._v2
        nodes[:, 2] = self._v3

        n = N_LOCAL_NODES
        rows = np.repeat(nodes, n, axis=1).ravel()
        cols = np.tile(nodes, n).ravel()
        n_entries = len(rows)

        # Sort by (row, col) to group duplicates and build CSR structure
        sort_order = np.lexsort((cols, rows))
        sorted_rows = rows[sort_order]
        sorted_cols = cols[sort_order]

        # Find boundaries between unique (row, col) pairs
        row_diff = np.diff(sorted_rows, prepend=-1)
        col_diff = np.diff(sorted_cols, prepend=-1)
        is_new_pair = (row_diff != 0) | (col_diff != 0)

        # Build CSR structure from unique pairs
        unique_rows = sorted_rows[is_new_pair]
        unique_cols = sorted_cols[is_new_pair]

        # indptr: cumulative count of entries per row
        self._csr_indptr = np.zeros(self.nonodes + 1, dtype=np.int64)
        np.add.at(self._csr_indptr, unique_rows + 1, 1)
        np.cumsum(self._csr_indptr, out=self._csr_indptr)

        self._csr_indices = unique_cols

        # Map each original entry to its position in CSR data array
        pair_indices = np.cumsum(is_new_pair) - 1

        # Invert sort to get mapping for original (unsorted) order
        self._csr_data_map = np.empty(n_entries, dtype=np.int64)
        self._csr_data_map[sort_order] = pair_indices

    def _compute_boundary_edges(self) -> None:
        elems_per_col = 2 * self.noelms2
        left_elems = np.arange(2, 2 * self.noelms2 + 1, 2)
        right_start = (self.noelms1 - 1) * elems_per_col + 1
        right_elems = np.arange(right_start, right_start + elems_per_col, 2)
        bottom_elems = np.arange(1, self.noelms1 + 1) * elems_per_col
        top_elems = 1 + np.arange(self.noelms1) * elems_per_col

        # Pre-allocate boundary arrays 
        n_boundary = 2 * (self.noelms1 + self.noelms2)
        self.boundary_edges = np.empty((n_boundary, 2), dtype=np.int64)
        self.boundary_sides = np.empty(n_boundary, dtype=np.int64)

        # Fill boundary edges and sides by section
        i = 0
        # Left side
        n = self.noelms2
        self.boundary_edges[i:i+n, 0] = left_elems
        self.boundary_edges[i:i+n, 1] = 3
        self.boundary_sides[i:i+n] = LEFT
        i += n
        # Right side
        self.boundary_edges[i:i+n, 0] = right_elems
        self.boundary_edges[i:i+n, 1] = 2
        self.boundary_sides[i:i+n] = RIGHT
        i += n
        # Bottom side
        n = self.noelms1
        self.boundary_edges[i:i+n, 0] = bottom_elems
        self.boundary_edges[i:i+n, 1] = 1
        self.boundary_sides[i:i+n] = BOTTOM
        i += n
        # Top side
        self.boundary_edges[i:i+n, 0] = top_elems
        self.boundary_edges[i:i+n, 1] = 3
        self.boundary_sides[i:i+n] = TOP

    def _compute_basis(self) -> None:
        """Compute delta and basis function coefficients for each element."""
        x1, y1, x2, y2, x3, y3 = self.vertex_coords

        self.delta = 0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        # Pre-allocate abc array (faster than stack + column_stack)
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

    @classmethod
    def from_meshio(
        cls,
        mesh: meshio.Mesh | str | Path,
        tol: float = BOUNDARY_TOL,
    ) -> Mesh2d:
        """
        Create Mesh2d from a meshio mesh or mesh file.

        Parameters
        ----------
        mesh : meshio.Mesh or str or Path
            Either a meshio Mesh object or path to a mesh file.
        tol : float
            Tolerance for boundary node detection.

        Returns
        -------
        Mesh2d
            The mesh object with all computed properties.
        """
        import meshio as mio

        if isinstance(mesh, (str, Path)):
            mesh = mio.read(mesh)

        # Extract points (2D only)
        points = mesh.points[:, :2]
        VX = points[:, 0].astype(np.float64)
        VY = points[:, 1].astype(np.float64)

        # Extract triangles (find triangle cells)
        EToV = None
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                EToV = cell_block.data.astype(np.int64) + 1  # 1-based indexing
                break

        if EToV is None:
            raise ValueError("No triangle cells found in mesh")

        # Compute bounding box
        x0, y0 = float(VX.min()), float(VY.min())
        x1, y1 = float(VX.max()), float(VY.max())
        L1, L2 = x1 - x0, y1 - y0

        # Estimate noelms1, noelms2 from mesh size
        avg_elem_area = np.abs(L1 * L2) / len(EToV)
        avg_elem_size = np.sqrt(2 * avg_elem_area)  # approximate edge length
        noelms1 = max(1, int(round(L1 / avg_elem_size)))
        noelms2 = max(1, int(round(L2 / avg_elem_size)))

        # Create instance without calling __post_init__
        instance = object.__new__(cls)
        instance.x0 = x0
        instance.y0 = y0
        instance.L1 = L1
        instance.L2 = L2
        instance.noelms1 = noelms1
        instance.noelms2 = noelms2
        instance.VX = VX
        instance.VY = VY
        instance.EToV = EToV
        instance.noelms = len(EToV)
        instance.nonodes = len(VX)
        instance.nonodes1 = noelms1 + 1
        instance.nonodes2 = noelms2 + 1

        # Compute derived properties
        instance._compute_assembly_indices()
        
        # Check for physical tags (line elements)
        has_tags = False
        line_cells = None
        line_tags = None
        
        if "line" in mesh.cells_dict:
            line_cells = mesh.cells_dict["line"]
            # Check for tags in cell_data
            # meshio typically stores them under "gmsh:physical"
            if "gmsh:physical" in mesh.cell_data_dict:
                 if "line" in mesh.cell_data_dict["gmsh:physical"]:
                     line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
                     has_tags = True
        
        if has_tags and line_cells is not None and line_tags is not None:
            instance._compute_boundary_edges_from_tags(line_cells, line_tags)
        else:
            instance._compute_boundary_edges_unstructured(tol)
            
        instance._compute_basis()

        return instance

    def _compute_boundary_edges_from_tags(
        self, 
        line_cells: NDArray[np.int64], 
        line_tags: NDArray[np.int64]
    ) -> None:
        """
        Compute boundary edges using physical tags from mesh file.
        
        Parameters
        ----------
        line_cells : (N, 2) array of node indices (0-based)
        line_tags : (N,) array of physical tags
        """
        # Create a dictionary mapping sorted edge nodes to tag
        # (min(n1, n2), max(n1, n2)) -> tag
        edge_to_tag = {}
        for (n1, n2), tag in zip(line_cells, line_tags):
            edge = tuple(sorted((n1, n2)))
            edge_to_tag[edge] = tag
            
        boundary_edges_list = []
        boundary_sides_list = []
        
        # Iterate over all elements and find edges that are in the map
        # This is O(noelms) which is fine
        for elem_idx in range(self.noelms):
            elem = elem_idx + 1 # 1-based
            vertices = self.EToV[elem_idx] - 1 # 0-based node indices
            
            for k in range(3): # edges 0, 1, 2
                va, vb = vertices[EDGE_VERTICES[k]]
                edge = tuple(sorted((va, vb)))
                
                if edge in edge_to_tag:
                    boundary_edges_list.append([elem, k + 1]) # 1-based edge
                    boundary_sides_list.append(edge_to_tag[edge])
                    
        self.boundary_edges = np.array(boundary_edges_list, dtype=np.int64)
        self.boundary_sides = np.array(boundary_sides_list, dtype=np.int64)

    def _compute_boundary_edges_unstructured(self, tol: float = BOUNDARY_TOL) -> None:
        """Compute boundary edges for unstructured mesh by finding edges on domain boundary."""
        # Find edges that lie on the bounding box
        x_min, x_max = self.x0, self.x0 + self.L1
        y_min, y_max = self.y0, self.y0 + self.L2

        boundary_edges_list = []
        boundary_sides_list = []

        # Edge k connects vertices EDGE_VERTICES[k-1]
        for elem_idx in range(self.noelms):
            elem = elem_idx + 1  # 1-based element number
            vertices = self.EToV[elem_idx]  # 1-based vertex indices

            for k in range(3):  # edges 1, 2, 3 (0-indexed as 0, 1, 2)
                va, vb = vertices[EDGE_VERTICES[k]]
                xa, ya = self.VX[va - 1], self.VY[va - 1]
                xb, yb = self.VX[vb - 1], self.VY[vb - 1]

                # Check if edge is on a boundary
                side = None
                if abs(xa - x_min) < tol and abs(xb - x_min) < tol:
                    side = LEFT
                elif abs(xa - x_max) < tol and abs(xb - x_max) < tol:
                    side = RIGHT
                elif abs(ya - y_min) < tol and abs(yb - y_min) < tol:
                    side = BOTTOM
                elif abs(ya - y_max) < tol and abs(yb - y_max) < tol:
                    side = TOP

                if side is not None:
                    boundary_edges_list.append([elem, k + 1])  # 1-based edge number
                    boundary_sides_list.append(side)

        self.boundary_edges = np.array(boundary_edges_list, dtype=np.int64)
        self.boundary_sides = np.array(boundary_sides_list, dtype=np.int64)

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
    """Compute unit outer normal for element n, edge k."""
    vertices = EToV[n - 1]
    va, vb = vertices[EDGE_VERTICES[k - 1]]
    xa, ya = VX[va - 1], VY[va - 1]
    xb, yb = VX[vb - 1], VY[vb - 1]
    dx, dy = xb - xa, yb - ya
    length = np.sqrt(dx**2 + dy**2)
    return float(dy / length), float(-dx / length)
