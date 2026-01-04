"""Numba JIT-compiled kernels for spectral residual computation.

These kernels are the performance-critical inner loops of the spectral solver.
All functions use Numba's @njit decorator for compiled performance.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def compute_derivatives_and_laplacian(
    u_2d: np.ndarray,
    v_2d: np.ndarray,
    Dx_1d: np.ndarray,
    Dy_1d_T: np.ndarray,
    Dxx_1d: np.ndarray,
    Dyy_1d_T: np.ndarray,
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
    lap_u: np.ndarray,
    lap_v: np.ndarray,
):
    """JIT-compiled computation of all velocity derivatives and Laplacians.

    Computes:
    - du/dx, du/dy, dv/dx, dv/dy (first derivatives)
    - lap_u = d²u/dx² + d²u/dy², lap_v = d²v/dx² + d²v/dy² (Laplacians)

    All outputs are flattened 1D arrays.

    Parameters
    ----------
    u_2d, v_2d : np.ndarray
        2D velocity arrays, shape (N+1, N+1)
    Dx_1d, Dy_1d_T : np.ndarray
        1D differentiation matrices (Dy transposed)
    Dxx_1d, Dyy_1d_T : np.ndarray
        1D second derivative matrices (Dyy transposed)
    du_dx, du_dy, dv_dx, dv_dy : np.ndarray
        Output arrays for first derivatives (flattened)
    lap_u, lap_v : np.ndarray
        Output arrays for Laplacians (flattened)

    Returns
    -------
    du_dx_2d, dv_dy_2d : np.ndarray
        2D derivative arrays needed for divergence computation
    """
    n = u_2d.shape[0]

    # Compute all matrix products
    # du/dx = Dx @ u, du/dy = u @ Dy.T
    du_dx_2d = Dx_1d @ u_2d
    du_dy_2d = u_2d @ Dy_1d_T
    dv_dx_2d = Dx_1d @ v_2d
    dv_dy_2d = v_2d @ Dy_1d_T

    # Laplacians: lap_u = Dxx @ u + u @ Dyy.T
    lap_u_2d = Dxx_1d @ u_2d + u_2d @ Dyy_1d_T
    lap_v_2d = Dxx_1d @ v_2d + v_2d @ Dyy_1d_T

    # Flatten to output arrays
    idx = 0
    for i in range(n):
        for j in range(n):
            du_dx[idx] = du_dx_2d[i, j]
            du_dy[idx] = du_dy_2d[i, j]
            dv_dx[idx] = dv_dx_2d[i, j]
            dv_dy[idx] = dv_dy_2d[i, j]
            lap_u[idx] = lap_u_2d[i, j]
            lap_v[idx] = lap_v_2d[i, j]
            idx += 1

    return du_dx_2d, dv_dy_2d  # Return these for divergence computation


@njit(cache=True, fastmath=True)
def compute_momentum_residuals(
    u: np.ndarray,
    v: np.ndarray,
    du_dx: np.ndarray,
    du_dy: np.ndarray,
    dv_dx: np.ndarray,
    dv_dy: np.ndarray,
    lap_u: np.ndarray,
    lap_v: np.ndarray,
    dp_dx: np.ndarray,
    dp_dy: np.ndarray,
    nu: float,
    R_u: np.ndarray,
    R_v: np.ndarray,
):
    """JIT-compiled momentum residual computation.

    Computes:
        R_u = -conv_u - dp/dx + nu*lap_u
        R_v = -conv_v - dp/dy + nu*lap_v

    where conv_u = u*du/dx + v*du/dy, conv_v = u*dv/dx + v*dv/dy

    Parameters
    ----------
    u, v : np.ndarray
        Flattened velocity arrays
    du_dx, du_dy, dv_dx, dv_dy : np.ndarray
        Flattened first derivative arrays
    lap_u, lap_v : np.ndarray
        Flattened Laplacian arrays
    dp_dx, dp_dy : np.ndarray
        Flattened pressure gradient arrays
    nu : float
        Kinematic viscosity (1/Re)
    R_u, R_v : np.ndarray
        Output residual arrays (modified in place)
    """
    n = u.shape[0]
    for i in range(n):
        conv_u = u[i] * du_dx[i] + v[i] * du_dy[i]
        conv_v = u[i] * dv_dx[i] + v[i] * dv_dy[i]
        R_u[i] = -conv_u - dp_dx[i] + nu * lap_u[i]
        R_v[i] = -conv_v - dp_dy[i] + nu * lap_v[i]


@njit(cache=True, fastmath=True)
def compute_continuity_residual(
    du_dx_2d: np.ndarray,
    dv_dy_2d: np.ndarray,
    beta_squared: float,
    R_p: np.ndarray,
):
    """JIT-compiled continuity residual on inner grid.

    Computes: R_p = -β² * (du/dx + dv/dy) on interior points only

    Parameters
    ----------
    du_dx_2d, dv_dy_2d : np.ndarray
        2D derivative arrays on full grid
    beta_squared : float
        Artificial compressibility parameter
    R_p : np.ndarray
        Output residual array on inner grid (modified in place)
    """
    n = du_dx_2d.shape[0]
    idx = 0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            R_p[idx] = -beta_squared * (du_dx_2d[i, j] + dv_dy_2d[i, j])
            idx += 1


@njit(cache=True, fastmath=True)
def interpolate_and_differentiate_pressure(
    p_inner: np.ndarray,
    Interp_x: np.ndarray,
    Interp_y_T: np.ndarray,
    Dx_1d: np.ndarray,
    Dy_1d_T: np.ndarray,
    dp_dx: np.ndarray,
    dp_dy: np.ndarray,
    shape_inner: tuple,
):
    """JIT-compiled pressure interpolation and gradient computation.

    Combines:
    1. Interpolation from inner to full grid: p_full = Interp_x @ p_inner @ Interp_y.T
    2. Gradient computation: dp/dx = Dx @ p_full, dp/dy = p_full @ Dy.T

    Parameters
    ----------
    p_inner : np.ndarray
        Flattened pressure on inner grid
    Interp_x, Interp_y_T : np.ndarray
        Interpolation matrices (Interp_y transposed)
    Dx_1d, Dy_1d_T : np.ndarray
        1D differentiation matrices (Dy transposed)
    dp_dx, dp_dy : np.ndarray
        Output gradient arrays on full grid (modified in place)
    shape_inner : tuple
        Shape of inner grid (N-1, N-1)
    """
    # Reshape to 2D
    ni, nj = shape_inner
    p_inner_2d = p_inner.reshape((ni, nj))

    # Interpolate: p_full = Interp_x @ p_inner @ Interp_y.T
    p_full_2d = Interp_x @ p_inner_2d @ Interp_y_T

    # Compute gradients
    dp_dx_2d = Dx_1d @ p_full_2d
    dp_dy_2d = p_full_2d @ Dy_1d_T

    # Flatten to output
    n = dp_dx_2d.shape[0]
    idx = 0
    for i in range(n):
        for j in range(n):
            dp_dx[idx] = dp_dx_2d[i, j]
            dp_dy[idx] = dp_dy_2d[i, j]
            idx += 1
