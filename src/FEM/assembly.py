import numpy as np
from numba import njit
from scipy.sparse import csr_matrix


@njit
def _build_csr_1d(Ke_all, n_nodes):
    """Build CSR arrays directly for 1D tridiagonal FEM matrix."""
    n_elem = len(Ke_all)

    # Tridiagonal: 2 entries in first/last row, 3 in middle rows
    nnz = 2 + 3 * (n_nodes - 2) + 2 if n_nodes > 2 else (2 * n_nodes - 1)

    data = np.zeros(nnz)
    indices = np.empty(nnz, dtype=np.int64)
    indptr = np.empty(n_nodes + 1, dtype=np.int64)

    # Build structure
    idx = 0
    for i in range(n_nodes):
        indptr[i] = idx
        if i > 0:
            indices[idx] = i - 1
            idx += 1
        indices[idx] = i
        idx += 1
        if i < n_nodes - 1:
            indices[idx] = i + 1
            idx += 1
    indptr[n_nodes] = idx

    # Accumulate element contributions
    for e in range(n_elem):
        n1, n2 = e, e + 1

        # Row n1: entries at (n1, n1) and (n1, n2)
        row_start = indptr[n1]
        if n1 == 0:
            data[row_start] += Ke_all[e, 0, 0]      # (n1, n1)
            data[row_start + 1] += Ke_all[e, 0, 1]  # (n1, n2)
        else:
            data[row_start + 1] += Ke_all[e, 0, 0]  # (n1, n1) - middle entry
            data[row_start + 2] += Ke_all[e, 0, 1]  # (n1, n2) - right entry

        # Row n2: entries at (n2, n1) and (n2, n2)
        row_start = indptr[n2]
        data[row_start] += Ke_all[e, 1, 0]          # (n2, n1) - left entry
        if n2 == n_nodes - 1:
            data[row_start + 1] += Ke_all[e, 1, 1]  # (n2, n2)
        else:
            data[row_start + 1] += Ke_all[e, 1, 1]  # (n2, n2) - middle entry

    return data, indices, indptr


def assemble_matrix_1d(Ke_all, n_nodes):
    """Assemble 1D FEM matrix directly to CSR (optimal for tridiagonal)."""
    data, indices, indptr = _build_csr_1d(Ke_all, n_nodes)
    return csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes))


@njit
def assemble_vector(elements, fe_all, n_nodes):
    """Assemble element load vectors into global vector."""
    b = np.zeros(n_nodes)
    n_elem, npe = fe_all.shape

    for e in range(n_elem):
        for i in range(npe):
            b[elements[e, i]] += fe_all[e, i]

    return b
