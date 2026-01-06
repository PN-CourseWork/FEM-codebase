import numpy as np


def apply_dirichlet_bc(
    A: np.ndarray, b: np.ndarray, node: int, value: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Dirichlet boundary condition at a single node.

    Modifies the system in-place using the elimination method:
    - Sets A[node, node] = 1, zeros other entries in row/column
    - Adjusts RHS for neighboring nodes
    - Sets b[node] = value

    Parameters
    ----------
    A : ndarray (M, M)
        Global stiffness matrix (modified in-place)
    b : ndarray (M,)
        Global load vector (modified in-place)
    node : int
        Node index where BC is applied
    value : float
        Prescribed value u(node) = value

    Returns
    -------
    A, b : tuple
        Modified matrix and vector (same objects, returned for convenience)
    """
    M = len(b)

    # Adjust RHS for neighbors
    for i in range(M):
        if i != node:
            b[i] -= A[i, node] * value

    # Zero row and column
    A[node, :] = 0
    A[:, node] = 0
    A[node, node] = 1
    b[node] = value

    return A, b


def apply_dirichlet_bc_symmetric(
    A: np.ndarray, b: np.ndarray, node: int, value: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Dirichlet BC preserving symmetry (for symmetric matrices).

    Same as apply_dirichlet_bc but only modifies upper triangle,
    suitable for use with Cholesky solver.

    Parameters
    ----------
    A : ndarray (M, M)
        Global stiffness matrix (modified in-place)
    b : ndarray (M,)
        Global load vector (modified in-place)
    node : int
        Node index where BC is applied
    value : float
        Prescribed value u(node) = value

    Returns
    -------
    A, b : tuple
        Modified matrix and vector
    """
    M = len(b)

    # Adjust RHS for neighbors (using symmetry)
    for i in range(M):
        if i != node:
            b[i] -= A[min(i, node), max(i, node)] * value

    # Zero row and column
    A[node, :] = 0
    A[:, node] = 0
    A[node, node] = 1
    b[node] = value

    return A, b
