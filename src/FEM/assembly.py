from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .datastructures import Mesh2d


def _p1_basis_gradients(
    x1: NDArray[np.float64],
    y1: NDArray[np.float64],
    x2: NDArray[np.float64],
    y2: NDArray[np.float64],
    x3: NDArray[np.float64],
    y3: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute P1 basis function gradients (b_i, c_i coefficients).
    """
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    return b1, b2, b3, c1, c2, c3


def _p1_local_stiffness(
    b1: NDArray[np.float64],
    b2: NDArray[np.float64],
    b3: NDArray[np.float64],
    c1: NDArray[np.float64],
    c2: NDArray[np.float64],
    c3: NDArray[np.float64],
    inv_4delta: NDArray[np.float64],
    lam1: float,
    lam2: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute P1 local stiffness matrix entries"""
    K11 = (lam1 * b1 * b1 + lam2 * c1 * c1) * inv_4delta
    K12 = (lam1 * b1 * b2 + lam2 * c1 * c2) * inv_4delta
    K13 = (lam1 * b1 * b3 + lam2 * c1 * c3) * inv_4delta
    K22 = (lam1 * b2 * b2 + lam2 * c2 * c2) * inv_4delta
    K23 = (lam1 * b2 * b3 + lam2 * c2 * c3) * inv_4delta
    K33 = (lam1 * b3 * b3 + lam2 * c3 * c3) * inv_4delta
    return K11, K12, K13, K22, K23, K33


def _p1_local_load(
    qt_v1: NDArray[np.float64],
    qt_v2: NDArray[np.float64],
    qt_v3: NDArray[np.float64],
    delta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute P1 load vector contributions per element.

    Uses centroid evaluation (as defined in 2.3.a in exercises).
    """
    q_avg = (qt_v1 + qt_v2 + qt_v3) / 3
    return q_avg * delta / 3


def assembly_2d(
    mesh: Mesh2d,
    lam1: float,
    lam2: float,
    qt: NDArray[np.float64],
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Assemble global stiffness matrix A and load vector b using COO format."""
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    x1, y1, x2, y2, x3, y3 = mesh.vertex_coords
    noelms = mesh.noelms

    # Reuse precomputed delta from mesh
    delta = mesh.delta
    inv_4delta = 1.0 / (4.0 * delta)

    # Compute basis gradients (P1-specific)
    b1, b2, b3, c1, c2, c3 = _p1_basis_gradients(x1, y1, x2, y2, x3, y3)

    # Compute local stiffness entries (P1-specific)
    K11, K12, K13, K22, K23, K33 = _p1_local_stiffness(
        b1, b2, b3, c1, c2, c3, inv_4delta, lam1, lam2
    )

    # Preallocate COO arrays (9 entries per element for 3x3 local matrix)
    nnz = 9 * noelms
    row_indices = np.empty(nnz, dtype=np.int64)
    col_indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)

    # Fill row indices: [v1, v1, v1, v2, v2, v2, v3, v3, v3] pattern
    row_indices[0::9] = v1
    row_indices[1::9] = v1
    row_indices[2::9] = v1
    row_indices[3::9] = v2
    row_indices[4::9] = v2
    row_indices[5::9] = v2
    row_indices[6::9] = v3
    row_indices[7::9] = v3
    row_indices[8::9] = v3

    # Fill col indices: [v1, v2, v3, v1, v2, v3, v1, v2, v3] pattern
    col_indices[0::9] = v1
    col_indices[1::9] = v2
    col_indices[2::9] = v3
    col_indices[3::9] = v1
    col_indices[4::9] = v2
    col_indices[5::9] = v3
    col_indices[6::9] = v1
    col_indices[7::9] = v2
    col_indices[8::9] = v3

    # Fill data: local stiffness entries (symmetric matrix)
    data[0::9] = K11
    data[1::9] = K12
    data[2::9] = K13
    data[3::9] = K12  # K21 = K12
    data[4::9] = K22
    data[5::9] = K23
    data[6::9] = K13  # K31 = K13
    data[7::9] = K23  # K32 = K23
    data[8::9] = K33

    # Create COO matrix and convert to CSR (handles duplicate entries automatically)
    A = csr_matrix(coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(mesh.nonodes, mesh.nonodes),
    ))

    # Compute local load contributions (P1-specific)
    contrib = _p1_local_load(qt[v1], qt[v2], qt[v3], delta)

    # Assemble global load vector using bincount with preallocated arrays
    all_nodes = np.empty(3 * noelms, dtype=np.int64)
    all_nodes[0::3] = v1
    all_nodes[1::3] = v2
    all_nodes[2::3] = v3

    all_contrib = np.empty(3 * noelms, dtype=np.float64)
    all_contrib[0::3] = contrib
    all_contrib[1::3] = contrib
    all_contrib[2::3] = contrib

    b = np.bincount(all_nodes, weights=all_contrib, minlength=mesh.nonodes)

    return A, b.astype(np.float64)
