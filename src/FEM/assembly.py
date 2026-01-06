import numpy as np
from typing import Callable


def assemble_1d(
    M: int,
    element_matrix_fn: Callable[[float], np.ndarray],
    h: float,
    element_load_fn: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble global stiffness matrix and load vector for 1D FEM.

    Parameters
    ----------
    M : int
        Number of nodes
    element_matrix_fn : callable
        Function that takes element size h and returns 2x2 element matrix
    h : float
        Element size (uniform mesh)
    element_load_fn : callable, optional
        Function that takes element size h and returns length-2 element load vector

    Returns
    -------
    A : ndarray (M, M)
        Global stiffness matrix
    b : ndarray (M,)
        Global load vector
    """
    A = np.zeros((M, M))
    b = np.zeros(M)

    Ke = element_matrix_fn(h)
    fe = element_load_fn(h) if element_load_fn else np.zeros(2)

    for i in range(M - 1):
        A[i, i] += Ke[0, 0]
        A[i, i + 1] += Ke[0, 1]
        A[i + 1, i] += Ke[1, 0]
        A[i + 1, i + 1] += Ke[1, 1]

        b[i] += fe[0]
        b[i + 1] += fe[1]

    return A, b


def assemble_1d_variable(
    x: np.ndarray,
    element_matrix_fn: Callable[[float], np.ndarray],
    element_load_fn: Callable[[float], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble global system for 1D FEM with variable element sizes.

    Parameters
    ----------
    x : ndarray
        Node coordinates
    element_matrix_fn : callable
        Function that takes element size h and returns 2x2 element matrix
    element_load_fn : callable, optional
        Function that takes element size h and returns length-2 element load vector

    Returns
    -------
    A : ndarray (M, M)
        Global stiffness matrix
    b : ndarray (M,)
        Global load vector
    """
    M = len(x)
    A = np.zeros((M, M))
    b = np.zeros(M)

    for i in range(M - 1):
        h = x[i + 1] - x[i]
        Ke = element_matrix_fn(h)
        fe = element_load_fn(h) if element_load_fn else np.zeros(2)

        A[i, i] += Ke[0, 0]
        A[i, i + 1] += Ke[0, 1]
        A[i + 1, i] += Ke[1, 0]
        A[i + 1, i + 1] += Ke[1, 1]

        b[i] += fe[0]
        b[i + 1] += fe[1]

    return A, b
