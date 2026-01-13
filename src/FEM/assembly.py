from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .datastructures import Mesh2d, N_LOCAL_NODES


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


def _pack_p1_stiffness_entries(
    K11: NDArray[np.float64],
    K12: NDArray[np.float64],
    K13: NDArray[np.float64],
    K22: NDArray[np.float64],
    K23: NDArray[np.float64],
    K33: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pack P1 local stiffness entries into flat array for assembly.

    Layout: [K11, K12, K13, K21, K22, K23, K31, K32, K33] per element.
    This ordering matches the (row, col) pairs from mesh assembly indices.

    """
    noelms = len(K11)
    n = N_LOCAL_NODES
    n2 = n * n  # 9 for P1
    data = np.empty(noelms * n2)
    data[0::n2] = K11
    data[1::n2] = K12
    data[2::n2] = K13
    data[3::n2] = K12  # Symmetric
    data[4::n2] = K22
    data[5::n2] = K23
    data[6::n2] = K13  # Symmetric
    data[7::n2] = K23  # Symmetric
    data[8::n2] = K33
    return data


def assembly_2d(
    mesh: Mesh2d,
    lam1: float,
    lam2: float,
    qt: NDArray[np.float64],
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Assemble global stiffness matrix A and load vector b. """
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    x1, y1, x2, y2, x3, y3 = mesh.vertex_coords

    # Reuse precomputed delta from mesh
    delta = mesh.delta
    inv_4delta = 1.0 / (4.0 * delta)

    # Compute basis gradients (P1-specific)
    b1, b2, b3, c1, c2, c3 = _p1_basis_gradients(x1, y1, x2, y2, x3, y3)

    # Compute local stiffness entries (P1-specific)
    K11, K12, K13, K22, K23, K33 = _p1_local_stiffness(
        b1, b2, b3, c1, c2, c3, inv_4delta, lam1, lam2
    )

    # Pack element entries into flat array
    element_data = _pack_p1_stiffness_entries(K11, K12, K13, K22, K23, K33)

    # Accumulate into CSR data array using pre-computed mapping
    # This avoids COO-to-CSR conversion (no sum_duplicates, no sorting)
    nnz = len(mesh._csr_indices)
    csr_data = np.zeros(nnz, dtype=np.float64)
    np.add.at(csr_data, mesh._csr_data_map, element_data)

    # Construct CSR matrix directly from pre-computed structure
    A = csr_matrix(
        (csr_data, mesh._csr_indices, mesh._csr_indptr),
        shape=(mesh.nonodes, mesh.nonodes),
    )

    # Compute local load contributions (P1-specific)
    contrib = _p1_local_load(qt[v1], qt[v2], qt[v3], delta)

    # Assemble global load vector using bincount
    all_nodes = np.concatenate([v1, v2, v3])
    all_contrib = np.concatenate([contrib, contrib, contrib])
    b = np.bincount(all_nodes, weights=all_contrib, minlength=mesh.nonodes)

    return A, b.astype(np.float64)
