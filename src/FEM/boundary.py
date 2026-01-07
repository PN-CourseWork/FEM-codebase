import numpy as np


def apply_dirichlet(A, b: np.ndarray, node: int, value: float) -> None:
    """
    Apply Dirichlet BC at a node. Modifies A and b in-place.

    Note: Only zeros the row, not the column. The column values remain
    but don't affect the solution since the BC row enforces u[node] = value.

    Parameters
    ----------
    A : csr_matrix
        Global stiffness matrix (modified in-place)
    b : ndarray
        Global load vector (modified in-place)
    node : int
        Node index
    value : float
        Prescribed value u[node] = value
    """
    # Zero row (efficient in CSR - just modify data array)
    A.data[A.indptr[node]:A.indptr[node + 1]] = 0
    A[node, node] = 1.0
    b[node] = value
