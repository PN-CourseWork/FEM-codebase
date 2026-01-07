import numpy as np
from numba import njit

from .mesh import Mesh


@njit
def _interpolate_1d(VX, EToV, u_nodal, x):
    """Numba-accelerated 1D interpolation."""
    n_pts = len(x)
    n_elem = len(EToV)
    u = np.zeros(n_pts)

    for i in range(n_pts):
        xi = x[i]
        for e in range(n_elem):
            n1, n2 = EToV[e]
            x1, x2 = VX[n1], VX[n2]
            if x1 <= xi <= x2:
                t = (xi - x1) / (x2 - x1)
                u[i] = (1 - t) * u_nodal[n1] + t * u_nodal[n2]
                break

    return u


def interpolate(mesh: Mesh, u_nodal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate piecewise linear interpolant at points x."""
    return _interpolate_1d(mesh.VX, mesh.EToV, u_nodal, np.asarray(x).ravel())


def project(mesh: Mesh, func) -> np.ndarray:
    """Project a function onto the mesh (nodal interpolation)."""
    return func(mesh.VX)


def l2_norm_element(mesh: Mesh, u_nodal: np.ndarray, e: int, n_quad: int = 3) -> float:
    """Compute L2 norm of interpolant over element e."""
    n1, n2 = mesh.EToV[e]
    x1, x2 = mesh.VX[n1], mesh.VX[n2]
    u1, u2 = u_nodal[n1], u_nodal[n2]
    h = x2 - x1

    if n_quad == 2:
        pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        wts = np.array([1.0, 1.0])
    else:
        pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        wts = np.array([5/9, 8/9, 5/9])

    integral = 0.0
    for pt, wt in zip(pts, wts):
        t = 0.5 * (pt + 1)
        u_val = (1 - t) * u1 + t * u2
        integral += u_val**2 * wt

    return np.sqrt(integral * h / 2)


def l2_norm(mesh: Mesh, u_nodal: np.ndarray) -> float:
    """Compute global L2 norm of interpolant."""
    total = sum(l2_norm_element(mesh, u_nodal, e)**2 for e in range(mesh.n_elem))
    return np.sqrt(total)


def l2_error_element(mesh: Mesh, u_nodal: np.ndarray, exact, e: int, n_quad: int = 5) -> float:
    """Compute L2 error of interpolant vs exact solution over element e."""
    n1, n2 = mesh.EToV[e]
    x1, x2 = mesh.VX[n1], mesh.VX[n2]
    u1, u2 = u_nodal[n1], u_nodal[n2]
    h = x2 - x1

    pts, wts = np.polynomial.legendre.leggauss(n_quad)

    integral = 0.0
    for pt, wt in zip(pts, wts):
        x_phys = x1 + 0.5 * (pt + 1) * h
        t = 0.5 * (pt + 1)
        u_interp = (1 - t) * u1 + t * u2
        integral += (u_interp - exact(x_phys))**2 * wt

    return np.sqrt(integral * h / 2)
