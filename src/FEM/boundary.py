from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from .datastructures import BOUNDARY_TOL, Mesh2d


def get_boundary_nodes(mesh: Mesh2d) -> NDArray[np.int64]:
    """Get all boundary node indices (0-based) for a rectangular mesh."""
    on_boundary = (
        (np.abs(mesh.VX - mesh.x0) < BOUNDARY_TOL)
        | (np.abs(mesh.VX - (mesh.x0 + mesh.L1)) < BOUNDARY_TOL)
        | (np.abs(mesh.VY - mesh.y0) < BOUNDARY_TOL)
        | (np.abs(mesh.VY - (mesh.y0 + mesh.L2)) < BOUNDARY_TOL)
    )
    return np.where(on_boundary)[0]


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
    """Impose Dirichlet BCs by modifying matrix A and vector b. (Algorithm 6)

    Args:
        bnodes: Boundary node indices (0-based)
    """
    n = A.shape[0]
    A_csr = A.tocsr()

    # A[:, bnodes] @ f == A @ f_full where f_full is zero except at bnodes
    f_full = np.zeros(n)
    f_full[bnodes] = f
    b -= A_csr @ f_full
    b[bnodes] = f

    # Zero boundary rows/cols and set diagonal to 1
    scale = np.ones(n)
    scale[bnodes] = 0

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

    # EToV is 0-indexed; n-1 because boundary_edges uses 1-indexed element numbers
    i = mesh.EToV[n - 1, r - 1]
    j = mesh.EToV[n - 1, s - 1]

    return i, j, mesh.VX[i], mesh.VY[i], mesh.VX[j], mesh.VY[j]


def neubc_2d(
    beds: NDArray[np.int64],
    q_i: NDArray[np.float64],
    q_j: NDArray[np.float64],
    mesh: Mesh2d,
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Impose Neumann BCs by updating the load vector. (Algorithm 8 in lecture notes)

    Uses exact integration for linear basis functions:
        integral(q * psi_i) = (L/6) * [2*q_i + q_j]
        integral(q * psi_j) = (L/6) * [q_i + 2*q_j]

    Args:
        beds: Boundary edges array (element number, local edge number)
        q_i: Neumann BC values at first node of each edge
        q_j: Neumann BC values at second node of each edge
        mesh: The mesh object
        b: Load vector to update

    Returns:
        Updated load vector
    """
    i, j, xi, yi, xj, yj = _get_edge_coords(beds, mesh)
    if i is None:
        return b

    edge_lengths = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
    L_over_6 = edge_lengths / 6

    # Exact integration for linear elements: int(q * psi_i) = (L/6)*(2*q_i + q_j)
    contrib_i = L_over_6 * (2 * q_i + q_j)
    contrib_j = L_over_6 * (q_i + 2 * q_j)

    np.add.at(b, i, -contrib_i)
    np.add.at(b, j, -contrib_j)
    return b


def get_edge_endpoints(
    beds: NDArray[np.int64],
    mesh: Mesh2d,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get endpoint coordinates for each boundary edge.

    Returns:
        Tuple of (endpoints_i, endpoints_j) where each is shape (n_edges, 2)
        containing (x, y) coordinates for the first and second node of each edge.
    """
    i, j, xi, yi, xj, yj = _get_edge_coords(beds, mesh)
    if i is None:
        empty = np.empty((0, 2))
        return empty, empty

    endpoints_i = np.empty((len(beds), 2), dtype=np.float64)
    endpoints_i[:, 0] = xi
    endpoints_i[:, 1] = yi

    endpoints_j = np.empty((len(beds), 2), dtype=np.float64)
    endpoints_j[:, 0] = xj
    endpoints_j[:, 1] = yj

    return endpoints_i, endpoints_j


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
