"""SpectralLevel dataclass and factory function.

A SpectralLevel holds all arrays and operators for one multigrid level.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def build_interpolation_matrix_1d(nodes_inner, nodes_full):
    """Build interpolation matrix from inner grid to full grid.

    Uses Chebyshev polynomial interpolation for spectral accuracy.
    Given values f_inner at nodes_inner, computes f_full = Interp @ f_inner.

    Parameters
    ----------
    nodes_inner : np.ndarray
        Inner grid nodes (excludes boundary points)
    nodes_full : np.ndarray
        Full grid nodes (includes boundary points)

    Returns
    -------
    Interp : np.ndarray
        Interpolation matrix of shape (n_full, n_inner)
    """
    from numpy.polynomial.chebyshev import chebvander

    n_inner = len(nodes_inner)

    # Map physical domain to [-1, 1] for Chebyshev polynomials
    a, b = nodes_full[0], nodes_full[-1]
    xi_inner = 2 * (nodes_inner - a) / (b - a) - 1
    xi_full = 2 * (nodes_full - a) / (b - a) - 1

    # Vandermonde matrices: V[i,k] = T_k(xi[i])
    V_inner = chebvander(xi_inner, n_inner - 1)  # (n_inner, n_inner)
    V_full = chebvander(xi_full, n_inner - 1)  # (n_full, n_inner)

    # Interpolation: f_full = V_full @ coeffs, where coeffs = V_inner^{-1} @ f_inner
    # So: f_full = (V_full @ V_inner^{-1}) @ f_inner = Interp @ f_inner
    Interp = V_full @ np.linalg.solve(V_inner, np.eye(n_inner))

    return Interp


@dataclass
class SpectralLevel:
    """Data structure holding all arrays and operators for one multigrid level.

    Attributes
    ----------
    n : int
        Polynomial order (gives n+1 nodes per dimension)
    level_idx : int
        Level index (0 = coarsest, increasing = finer)
    """

    # Grid info
    n: int  # polynomial order
    level_idx: int

    # 1D node arrays
    x_nodes: np.ndarray
    y_nodes: np.ndarray

    # 2D meshgrid arrays (full grid for velocities)
    X: np.ndarray
    Y: np.ndarray

    # 2D meshgrid arrays (inner grid for pressure)
    X_inner: np.ndarray
    Y_inner: np.ndarray

    # Grid shapes
    shape_full: Tuple[int, int]
    shape_inner: Tuple[int, int]

    # Minimum grid spacing for CFL
    dx_min: float
    dy_min: float

    # 1D differentiation matrices (for O(N³) tensor product operations)
    Dx_1d: np.ndarray  # 1D d/dx matrix (N+1) × (N+1)
    Dy_1d: np.ndarray  # 1D d/dy matrix (N+1) × (N+1)
    Dxx_1d: np.ndarray  # 1D d²/dx² matrix (N+1) × (N+1)
    Dyy_1d: np.ndarray  # 1D d²/dy² matrix (N+1) × (N+1)

    # Interpolation matrices from inner to full grid (for pressure gradient)
    Interp_x: np.ndarray  # 1D interpolation in x direction
    Interp_y: np.ndarray  # 1D interpolation in y direction

    # Solution arrays (flattened)
    u: np.ndarray  # velocity u on full grid
    v: np.ndarray  # velocity v on full grid
    p: np.ndarray  # pressure on inner grid

    # Previous iteration (for convergence)
    u_prev: np.ndarray
    v_prev: np.ndarray

    # RK4 stage buffers
    u_stage: np.ndarray
    v_stage: np.ndarray
    p_stage: np.ndarray

    # Residual arrays
    R_u: np.ndarray
    R_v: np.ndarray
    R_p: np.ndarray

    # Work buffers for derivatives
    du_dx: np.ndarray
    du_dy: np.ndarray
    dv_dx: np.ndarray
    dv_dy: np.ndarray
    lap_u: np.ndarray
    lap_v: np.ndarray
    dp_dx: np.ndarray
    dp_dy: np.ndarray

    @property
    def n_nodes_full(self) -> int:
        return self.shape_full[0] * self.shape_full[1]

    @property
    def n_nodes_inner(self) -> int:
        return self.shape_inner[0] * self.shape_inner[1]


def build_spectral_level(
    n: int,
    level_idx: int,
    basis_x,
    basis_y,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> SpectralLevel:
    """Construct a SpectralLevel with all operators and arrays.

    Parameters
    ----------
    n : int
        Polynomial order (n+1 nodes per dimension)
    level_idx : int
        Level index in hierarchy
    basis_x, basis_y : Basis objects
        Spectral basis (Chebyshev or Legendre Lobatto)
    Lx, Ly : float
        Domain dimensions

    Returns
    -------
    SpectralLevel
        Fully initialized level
    """
    # Grid shapes
    shape_full = (n + 1, n + 1)
    shape_inner = (n - 1, n - 1)
    n_full = shape_full[0] * shape_full[1]
    n_inner = shape_inner[0] * shape_inner[1]

    # 1D nodes
    x_nodes = basis_x.nodes(n + 1)
    y_nodes = basis_y.nodes(n + 1)

    # 2D meshgrids
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    x_inner = x_nodes[1:-1]
    y_inner = y_nodes[1:-1]
    X_inner, Y_inner = np.meshgrid(x_inner, y_inner, indexing="ij")

    # Grid spacing
    dx_min = np.min(np.diff(x_nodes))
    dy_min = np.min(np.diff(y_nodes))

    # Build 1D differentiation matrices (stored for O(N³) tensor product operations)
    Dx_1d = basis_x.diff_matrix(x_nodes)
    Dy_1d = basis_y.diff_matrix(y_nodes)
    Dxx_1d = Dx_1d @ Dx_1d
    Dyy_1d = Dy_1d @ Dy_1d

    # Build interpolation matrices from inner to full grid (for pressure gradient)
    Interp_x = build_interpolation_matrix_1d(x_inner, x_nodes)
    Interp_y = build_interpolation_matrix_1d(y_inner, y_nodes)

    # Allocate solution and work arrays
    return SpectralLevel(
        n=n,
        level_idx=level_idx,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        X=X,
        Y=Y,
        X_inner=X_inner,
        Y_inner=Y_inner,
        shape_full=shape_full,
        shape_inner=shape_inner,
        dx_min=dx_min,
        dy_min=dy_min,
        # 1D matrices for tensor product operations
        Dx_1d=Dx_1d,
        Dy_1d=Dy_1d,
        Dxx_1d=Dxx_1d,
        Dyy_1d=Dyy_1d,
        Interp_x=Interp_x,
        Interp_y=Interp_y,
        # Solution arrays
        u=np.zeros(n_full),
        v=np.zeros(n_full),
        p=np.zeros(n_inner),
        u_prev=np.zeros(n_full),
        v_prev=np.zeros(n_full),
        u_stage=np.zeros(n_full),
        v_stage=np.zeros(n_full),
        p_stage=np.zeros(n_inner),
        R_u=np.zeros(n_full),
        R_v=np.zeros(n_full),
        R_p=np.zeros(n_inner),
        du_dx=np.zeros(n_full),
        du_dy=np.zeros(n_full),
        dv_dx=np.zeros(n_full),
        dv_dy=np.zeros(n_full),
        lap_u=np.zeros(n_full),
        lap_v=np.zeros(n_full),
        dp_dx=np.zeros(n_full),
        dp_dy=np.zeros(n_full),
    )
