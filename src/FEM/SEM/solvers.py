"""Spectral Element Method solvers."""

from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from .mesh import SEMMesh2D
from .assembly import (
    assemble_global_stiffness,
    assemble_load_vector,
    apply_dirichlet_bc_func,
)


def solve_poisson_sem(
    mesh: SEMMesh2D,
    f_func: Callable[[NDArray, NDArray], NDArray],
    bc_func: Callable[[NDArray, NDArray], NDArray],
    bc_sides: list[str] | None = None,
) -> NDArray[np.float64]:
    """Solve -∇²u = f with Dirichlet BCs."""
    A = assemble_global_stiffness(mesh)
    b = assemble_load_vector(mesh, f_func)
    A, b = apply_dirichlet_bc_func(A, b, mesh, bc_func, bc_sides)
    return spsolve(A, b)


def l2_error_sem(
    mesh: SEMMesh2D,
    u_numerical: NDArray[np.float64],
    u_exact: Callable[[NDArray, NDArray], NDArray],
) -> float:
    """Compute L2 error ||u_h - u||_2."""
    ref = mesh.ref_elem
    w_ref = np.outer(ref.wx, ref.wy).ravel()
    error_sq = 0.0

    for e in range(mesh.noelms):
        glb = mesh.loc2glb[e]
        diff = u_numerical[glb] - u_exact(mesh.VX[glb], mesh.VY[glb])
        error_sq += mesh.jacobians[e] * np.sum(w_ref * diff**2)

    return np.sqrt(error_sq)


def linf_error_sem(
    mesh: SEMMesh2D,
    u_numerical: NDArray[np.float64],
    u_exact: Callable[[NDArray, NDArray], NDArray],
) -> float:
    """Compute L∞ error max|u_h - u|."""
    return np.max(np.abs(u_numerical - u_exact(mesh.VX, mesh.VY)))


def h1_seminorm_error_sem(
    mesh: SEMMesh2D,
    u_numerical: NDArray[np.float64],
    grad_u_exact: Callable[[NDArray, NDArray], tuple[NDArray, NDArray]],
) -> float:
    """Compute H1 seminorm error |u_h - u|_1 = ||∇(u_h - u)||_2."""
    ref = mesh.ref_elem
    w_ref = np.outer(ref.wx, ref.wy).ravel()
    nloc_1d = mesh.polynomial_order + 1
    error_sq = 0.0

    for e in range(mesh.noelms):
        glb = mesh.loc2glb[e]
        u_e = u_numerical[glb].reshape((nloc_1d, nloc_1d))

        # Numerical gradient (scaled for physical coordinates)
        J = mesh.jacobians[e]
        h = np.sqrt(J)
        du_dx_num = (ref.Dx @ u_e).ravel() / h
        du_dy_num = (u_e @ ref.Dy.T).ravel() / h

        # Exact gradient
        du_dx_ex, du_dy_ex = grad_u_exact(mesh.VX[glb], mesh.VY[glb])

        # Integrate gradient error
        diff_x, diff_y = du_dx_num - du_dx_ex, du_dy_num - du_dy_ex
        error_sq += J * np.sum(w_ref * (diff_x**2 + diff_y**2))

    return np.sqrt(error_sq)
