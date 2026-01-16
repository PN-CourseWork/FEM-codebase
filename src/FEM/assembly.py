from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .datastructures import Mesh2d


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
    """Compute P1 local stiffness matrix entries."""
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute P1 load vector contributions per element.
    """
    # Same contribution for all three nodes 
    contrib = delta / 9 * (qt_v1 + qt_v2 + qt_v3)
    return contrib, contrib, contrib


def assembly_2d(
    mesh: Mesh2d,
    qt: NDArray[np.float64],
    lam1: float = 1.0,
    lam2: float = 1.0,
) -> tuple[csr_matrix, NDArray[np.float64]]:
    """Assemble global stiffness matrix A and load vector b.
    """
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    noelms = mesh.noelms

    # abc[:, i, 1] = b_i, abc[:, i, 2] = c_i
    b1, b2, b3 = mesh.abc[:, 0, 1], mesh.abc[:, 1, 1], mesh.abc[:, 2, 1]
    c1, c2, c3 = mesh.abc[:, 0, 2], mesh.abc[:, 1, 2], mesh.abc[:, 2, 2]
    inv_4delta = 1.0 / (4.0 * mesh.delta)

    # Compute local stiffness entries
    K11, K12, K13, K22, K23, K33 = _p1_local_stiffness(
        b1, b2, b3, c1, c2, c3, inv_4delta, lam1, lam2
    )

    # Build COO arrays
    nnz = 9 * noelms
    row_indices = np.empty(nnz, dtype=np.int64)
    col_indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)

    row_indices[0::9] = v1
    row_indices[1::9] = v1
    row_indices[2::9] = v1
    row_indices[3::9] = v2
    row_indices[4::9] = v2
    row_indices[5::9] = v2
    row_indices[6::9] = v3
    row_indices[7::9] = v3
    row_indices[8::9] = v3

    col_indices[0::9] = v1
    col_indices[1::9] = v2
    col_indices[2::9] = v3
    col_indices[3::9] = v1
    col_indices[4::9] = v2
    col_indices[5::9] = v3
    col_indices[6::9] = v1
    col_indices[7::9] = v2
    col_indices[8::9] = v3

    data[0::9] = K11
    data[1::9] = K12
    data[2::9] = K13
    data[3::9] = K12
    data[4::9] = K22
    data[5::9] = K23
    data[6::9] = K13
    data[7::9] = K23
    data[8::9] = K33

    # Create CSR matrix directly from COO-style input
    A = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(mesh.nonodes, mesh.nonodes),
    )

    # Compute local load contributions 
    contrib_v1, contrib_v2, contrib_v3 = _p1_local_load(qt[v1], qt[v2], qt[v3], mesh.delta)

    # Assemble global load vector 
    all_nodes = np.empty(3 * noelms, dtype=np.int64)
    all_nodes[0::3] = v1
    all_nodes[1::3] = v2
    all_nodes[2::3] = v3

    all_contrib = np.empty(3 * noelms, dtype=np.float64)
    all_contrib[0::3] = contrib_v1
    all_contrib[1::3] = contrib_v2
    all_contrib[2::3] = contrib_v3

    b = np.bincount(all_nodes, weights=all_contrib, minlength=mesh.nonodes)

    return A, b.astype(np.float64)
