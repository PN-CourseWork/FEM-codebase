import numpy as np


def solve_symmetric(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve linear system Ax = b for symmetric positive definite A using Cholesky.

    Parameters
    ----------
    A : ndarray (M, M)
        Symmetric positive definite matrix
    b : ndarray (M,)
        Right-hand side vector

    Returns
    -------
    x : ndarray (M,)
        Solution vector
    """
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x


def solve_general(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve linear system Ax = b for general (non-symmetric) A.

    Parameters
    ----------
    A : ndarray (M, M)
        Coefficient matrix
    b : ndarray (M,)
        Right-hand side vector

    Returns
    -------
    x : ndarray (M,)
        Solution vector
    """
    return np.linalg.solve(A, b)
