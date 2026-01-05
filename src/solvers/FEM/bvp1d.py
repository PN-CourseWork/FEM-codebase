"""1D Finite Element Method for Boundary Value Problems.

Solves problems of the form: -u'' + c*u = f, with Dirichlet BCs.

For the reaction-diffusion equation u'' - u = 0 (c=-1, f=0),
this corresponds to the weak form derived in Chapter 1 of the lecture notes.
"""

import numpy as np


def element_matrix_reaction(h: float) -> np.ndarray:
    """
    Compute element matrix K for reaction-diffusion: u'' - u = 0.

    From equation (1.27) in the lecture notes:
    K = [ (1/h + h/3)   (-1/h + h/6) ]
        [ (-1/h + h/6)  (1/h + h/3)  ]

    Parameters
    ----------
    h : float
        Element length

    Returns
    -------
    K : ndarray, shape (2, 2)
        Element stiffness matrix
    """
    k11 = 1.0 / h + h / 3.0
    k12 = -1.0 / h + h / 6.0
    return np.array([[k11, k12], [k12, k11]])


def assemble_global_matrix(x: np.ndarray, element_matrix_func=None) -> np.ndarray:
    """
    Assemble global stiffness matrix using Algorithm 1 from lecture notes.

    Parameters
    ----------
    x : ndarray
        Mesh node coordinates (M nodes)
    element_matrix_func : callable, optional
        Function that takes element length h and returns 2x2 element matrix.
        Defaults to element_matrix_reaction.

    Returns
    -------
    A : ndarray, shape (M, M)
        Global stiffness matrix (before BC application)
    """
    if element_matrix_func is None:
        element_matrix_func = element_matrix_reaction

    M = len(x)
    A = np.zeros((M, M))

    for i in range(M - 1):
        hi = x[i + 1] - x[i]
        K = element_matrix_func(hi)

        # Add element contributions to global matrix
        A[i, i] += K[0, 0]
        A[i, i + 1] += K[0, 1]
        A[i + 1, i] += K[1, 0]
        A[i + 1, i + 1] += K[1, 1]

    return A


def apply_dirichlet_bc(
    A: np.ndarray, b: np.ndarray, left_bc: float, right_bc: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Dirichlet boundary conditions using Algorithm 2 from lecture notes.

    Modifies A and b in place to enforce u(0) = left_bc and u(L) = right_bc.

    Parameters
    ----------
    A : ndarray
        Global stiffness matrix (modified in place)
    b : ndarray
        Right-hand side vector (modified in place)
    left_bc : float
        Value at left boundary u(0)
    right_bc : float
        Value at right boundary u(L)

    Returns
    -------
    A, b : tuple
        Modified matrix and vector
    """
    M = len(b)

    # Left BC: u(0) = left_bc
    b[0] = left_bc
    b[1] = b[1] - A[0, 1] * left_bc
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[1, 0] = 0.0

    # Right BC: u(L) = right_bc
    b[M - 1] = right_bc
    b[M - 2] = b[M - 2] - A[M - 2, M - 1] * right_bc
    A[M - 1, M - 1] = 1.0
    A[M - 2, M - 1] = 0.0
    A[M - 1, M - 2] = 0.0

    return A, b


def solve_fem_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the FEM system Au = b using Cholesky factorization.

    Falls back to standard solve if A is not positive definite.

    Parameters
    ----------
    A : ndarray
        System matrix (should be symmetric positive definite)
    b : ndarray
        Right-hand side vector

    Returns
    -------
    u : ndarray
        Solution vector
    """
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        u = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        print("Warning: A not positive definite, using standard solve")
        u = np.linalg.solve(A, b)

    return u


def BVP1D_solver(
    x: np.ndarray,
    left_bc: float,
    right_bc: float,
    element_matrix_func=None,
    rhs_func=None,
) -> np.ndarray:
    """
    Solve 1D BVP with given mesh and boundary conditions.

    Parameters
    ----------
    x : ndarray
        Mesh node coordinates
    left_bc : float
        Value at left boundary u(x[0])
    right_bc : float
        Value at right boundary u(x[-1])
    element_matrix_func : callable, optional
        Element matrix function. Defaults to reaction-diffusion.
    rhs_func : callable, optional
        Right-hand side function f(x). Defaults to zero (homogeneous).

    Returns
    -------
    u : ndarray
        Solution at mesh nodes
    """
    M = len(x)

    # Assemble global matrix
    A = assemble_global_matrix(x, element_matrix_func)

    # Initialize RHS (for f=0, b starts as zeros)
    b = np.zeros(M)

    # TODO: Add load vector assembly for non-zero f

    # Apply boundary conditions
    A, b = apply_dirichlet_bc(A, b, left_bc, right_bc)

    # Solve system
    u = solve_fem_system(A, b)

    return u
