from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from .datastructures import BOUNDARY_TOL, Mesh2d


def get_boundary_nodes(mesh: Mesh2d) -> NDArray[np.int64]:
    """Get all boundary node indices (1-based) from mesh boundary edges."""
    if len(mesh.boundary_edges) == 0:
        return np.array([], dtype=np.int64)

    # boundary_edges is (N, 2) [elem_idx (1-based), edge_local_idx (1-3)]
    # EDGE_VERTICES maps edge_idx (0-2) to vertex indices (0-2)
    # EDGE_VERTICES = [[0, 1], [1, 2], [2, 0]]
    
    # Vectorized lookup
    elems = mesh.boundary_edges[:, 0] - 1  # 0-based element indices
    edges = mesh.boundary_edges[:, 1] - 1  # 0-based edge indices
    
    # Get all 3 vertices for the boundary elements
    # Shape (N, 3)
    elem_nodes = mesh.EToV[elems]
    
    # We need to pick the 2 vertices corresponding to the edge
    # Construct indices arrays
    from .datastructures import EDGE_VERTICES
    
    # Map edge index to vertex indices
    v1_local = EDGE_VERTICES[edges, 0]
    v2_local = EDGE_VERTICES[edges, 1]
    
    # Extract global node indices (1-based from EToV)
    # Use advanced indexing: elem_nodes[row_indices, col_indices]
    n1 = elem_nodes[np.arange(len(elems)), v1_local]
    n2 = elem_nodes[np.arange(len(elems)), v2_local]
    
    # Combine and find unique
    all_bnodes = np.concatenate([n1, n2])
    return np.unique(all_bnodes)


def get_boundary_edges(
    mesh: Mesh2d,
    side: int | None = None,
) -> NDArray[np.int64]:
    """Get boundary edges, optionally filtered by side (LEFT/RIGHT/BOTTOM/TOP)."""
    if side is None:
        return mesh.boundary_edges
    return mesh.boundary_edges[mesh.boundary_sides == side]


def dirbc_2d(
    bnodes: NDArray[np.int64],
    f: NDArray[np.float64],
    A: spmatrix,
    b: NDArray[np.float64],
) -> tuple[spmatrix, NDArray[np.float64]]:
    """Impose Dirichlet BCs by modifying matrix A and vector b. (Algorithm 6)"""
    bnodes_0 = bnodes - 1
    n = A.shape[0]
    A_csr = A.tocsr()

    # A[:, bnodes] @ f == A @ f_full where f_full is zero except at bnodes
    f_full = np.zeros(n)
    f_full[bnodes_0] = f
    b -= A_csr @ f_full
    b[bnodes_0] = f

    # Zero boundary rows/cols and set diagonal to 1
    scale = np.ones(n)
    scale[bnodes_0] = 0

    row_scale = np.repeat(scale, np.diff(A_csr.indptr))
    col_scale = scale[A_csr.indices]

    A_new = A_csr.copy()
    A_new.data *= row_scale * col_scale
    A_new.setdiag(A_new.diagonal() + (scale == 0).astype(float))

    return A_new, b


def _get_edge_coords(
    beds: NDArray[np.int64],
    mesh: Mesh2d,
) -> tuple[
    NDArray[np.int64] | None,
    NDArray[np.int64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    """Get edge endpoint coordinates for boundary edges. """
    if len(beds) == 0:
        return None, None, None, None, None, None

    # Map local edge number (1,2,3) to next vertex in CCW order
    next_local = np.array([0, 2, 3, 1])
    n, r = beds[:, 0], beds[:, 1]
    s = next_local[r]

    i = mesh.EToV[n - 1, r - 1] - 1
    j = mesh.EToV[n - 1, s - 1] - 1

    return i, j, mesh.VX[i], mesh.VY[i], mesh.VX[j], mesh.VY[j]


def neubc_2d(
    beds: NDArray[np.int64],
    q: NDArray[np.float64],
    mesh: Mesh2d,
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Impose Neumann BCs by updating the load vector. (Algorithm 8 in lecture nodes"""
    i, j, xi, yi, xj, yj = _get_edge_coords(beds, mesh)
    if i is None:
        return b

    edge_lengths = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2) # 2.41
    q_contrib = q * edge_lengths / 2 

    np.add.at(b, i, -q_contrib)
    np.add.at(b, j, -q_contrib)
    return b


def get_edge_midpoints(
    beds: NDArray[np.int64],
    mesh: Mesh2d,
) -> NDArray[np.float64]:
    """Get midpoint coordinates for each boundary edge."""
    i, j, xi, yi, xj, yj = _get_edge_coords(beds, mesh)
    if i is None:
        return np.empty((0, 2))
    # Pre-allocate result (faster than column_stack)
    midpoints = np.empty((len(beds), 2), dtype=np.float64)
    midpoints[:, 0] = (xi + xj) / 2
    midpoints[:, 1] = (yi + yj) / 2
    return midpoints
