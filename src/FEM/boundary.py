import numpy as np
from scipy.sparse import csr_matrix


def apply_dirichlet(A: csr_matrix, b: np.ndarray, nodes, values) -> None:
    """
    Apply Dirichlet BC at specified nodes. Modifies A and b in-place.
    """
    nodes = np.atleast_1d(nodes)
    values = np.atleast_1d(values)

    if len(values) == 1 and len(nodes) > 1:
        values = np.full(len(nodes), values[0])

    for node, val in zip(nodes, values):
        start = A.indptr[node]
        end = A.indptr[node + 1]

        A.data[start:end] = 0.0

        found = False
        for k in range(start, end):
            if A.indices[k] == node:
                A.data[k] = 1.0
                found = True
                break

        if not found:
            A[node, node] = 1.0

        b[node] = val
