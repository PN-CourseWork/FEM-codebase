"""Spectral building blocks for SEM: LGL nodes, differentiation, and 2D operators."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import sparse


def jacobi_poly(xs: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """Compute Jacobi polynomial P_N^{(alpha,beta)}(x) using recurrence."""
    if N == 0:
        return np.ones_like(xs)
    if N == 1:
        return 0.5 * (alpha - beta + (alpha + beta + 2) * xs)

    jpm2, jpm1 = np.ones_like(xs), 0.5 * (alpha - beta + (alpha + beta + 2) * xs)
    for n in range(2, N + 1):
        am1 = (2 * ((n-1) + alpha) * ((n-1) + beta)) / ((2*(n-1) + alpha + beta + 1) * (2*(n-1) + alpha + beta))
        a0 = (alpha**2 - beta**2) / ((2*(n-1) + alpha + beta + 2) * (2*(n-1) + alpha + beta))
        ap1 = (2 * ((n-1) + 1) * ((n-1) + alpha + beta + 1)) / ((2*(n-1) + alpha + beta + 2) * (2*(n-1) + alpha + beta + 1))
        jpm2, jpm1 = jpm1, ((a0 + xs) * jpm1 - am1 * jpm2) / ap1
    return jpm1


def legendre_gauss_lobatto_nodes(num_nodes: int) -> np.ndarray:
    """Compute LGL nodes on [-1, 1]."""
    degree = num_nodes - 1
    roots = Legendre.basis(degree).deriv().roots()
    return np.sort(np.concatenate(([-1.0], roots, [1.0])))


def legendre_gauss_lobatto_weights(num_nodes: int) -> np.ndarray:
    """Compute LGL quadrature weights (sum to 2)."""
    N = num_nodes - 1
    if N == 0:
        return np.array([2.0])
    nodes = legendre_gauss_lobatto_nodes(num_nodes)
    P_N = jacobi_poly(nodes, 0.0, 0.0, N)
    return 2.0 / (N * (N + 1) * P_N**2)


def _vandermonde(nodes: np.ndarray) -> np.ndarray:
    """Vandermonde matrix for Legendre polynomials."""
    N = len(nodes)
    V = np.zeros((N, N))
    for n in range(N):
        V[:, n] = jacobi_poly(nodes, 0.0, 0.0, n)
    return V


def _vandermonde_x(nodes: np.ndarray) -> np.ndarray:
    """Derivative Vandermonde matrix."""
    N = len(nodes)
    Vx = np.zeros((N, N))
    for n in range(1, N):  # n=0 derivative is 0
        Vx[:, n] = 0.5 * (n + 1) * jacobi_poly(nodes, 1.0, 1.0, n - 1)
    return Vx


def _legendre_diff_matrix(nodes: np.ndarray) -> np.ndarray:
    """Differentiation matrix D such that D @ u ≈ du/dx."""
    V = _vandermonde(nodes)
    Vx = _vandermonde_x(nodes)
    return Vx @ np.linalg.solve(V, np.eye(len(nodes)))


@dataclass
class SpectralElement2D:
    """2D spectral element with pre-computed operators."""
    Nx: int
    Ny: int
    domain: tuple[tuple[float, float], tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0))

    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Dx: np.ndarray = field(init=False, repr=False)
    Dy: np.ndarray = field(init=False, repr=False)
    Dy_T: np.ndarray = field(init=False, repr=False)
    wx: np.ndarray = field(init=False, repr=False)
    wy: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        (x_min, x_max), (y_min, y_max) = self.domain

        # Reference LGL nodes
        xi_x = legendre_gauss_lobatto_nodes(self.Nx + 1)
        xi_y = legendre_gauss_lobatto_nodes(self.Ny + 1)

        # Map to physical domain
        self.x = 0.5 * (x_max - x_min) * (xi_x + 1) + x_min
        self.y = 0.5 * (y_max - y_min) * (xi_y + 1) + y_min
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Scaled differentiation matrices
        scale_x, scale_y = 2.0 / (x_max - x_min), 2.0 / (y_max - y_min)
        self.Dx = scale_x * _legendre_diff_matrix(xi_x)
        self.Dy = scale_y * _legendre_diff_matrix(xi_y)
        self.Dy_T = self.Dy.T

        # Scaled quadrature weights
        w_ref_x = legendre_gauss_lobatto_weights(self.Nx + 1)
        w_ref_y = legendre_gauss_lobatto_weights(self.Ny + 1)
        self.wx = w_ref_x * (x_max - x_min) / 2
        self.wy = w_ref_y * (y_max - y_min) / 2

    @property
    def num_nodes(self) -> int:
        return (self.Nx + 1) * (self.Ny + 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.Nx + 1, self.Ny + 1)


def mass_matrix_2d(wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
    """2D mass matrix diagonal (tensor product of weights)."""
    return np.outer(wx, wy).ravel()


def stiffness_matrix_2d(
    Dx: np.ndarray, Dy: np.ndarray, wx: np.ndarray, wy: np.ndarray
) -> sparse.csr_matrix:
    """2D stiffness matrix for Laplacian: S[i,j] = ∫ ∇φ_i · ∇φ_j dΩ."""
    Wx, Wy = sparse.diags(wx), sparse.diags(wy)
    Sx = sparse.kron(Wy, Dx.T @ Wx.toarray() @ Dx)
    Sy = sparse.kron(Dy.T @ Wy.toarray() @ Dy, Wx)
    return (Sx + Sy).tocsr()
