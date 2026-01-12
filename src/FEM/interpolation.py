import numpy as np
from numba import njit

from .datastructures import Mesh


@njit
def _interpolate_1d(x_coords, cells, u_nodal, x):
    """Numba-accelerated 1D interpolation."""
    n_pts = len(x)
    n_elem = len(cells)
    u = np.zeros(n_pts)

    for i in range(n_pts):
        xi = x[i]
        for e in range(n_elem):
            n1, n2 = cells[e]
            x1, x2 = x_coords[n1], x_coords[n2]
            if x1 <= xi <= x2:
                t = (xi - x1) / (x2 - x1)
                u[i] = (1 - t) * u_nodal[n1] + t * u_nodal[n2]
                break

    return u


def interpolate(mesh: Mesh, u_nodal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate piecewise linear interpolant at points x."""
    x_coords = mesh.VX.ravel()
    return _interpolate_1d(x_coords, mesh.EToV, u_nodal, np.asarray(x).ravel())


def _element_edges_1d(mesh: Mesh):
    """Extract left/right coordinates and lengths for each 1D element."""
    x_left = mesh.VX[mesh.EToV[:, 0]].ravel()
    x_right = mesh.VX[mesh.EToV[:, 1]].ravel()
    h = np.abs(x_right - x_left)
    return x_left, x_right, h


def discrete_l2_error(mesh: Mesh, u_nodal: np.ndarray, u_exact) -> float:
    """
    Compute discrete L2 error using exact analytical integration.

    """
    x = mesh.VX
    u_ex = np.asarray(u_exact(x))
    e = u_nodal - u_ex  # Error at nodes

    # Get element connectivity (nodes may not be consecutive after AMR!)
    i_nodes = mesh.EToV[:, 0]
    j_nodes = mesh.EToV[:, 1]

    # Element sizes from connectivity
    h = mesh.VX[j_nodes] - mesh.VX[i_nodes]
    e_left = e[i_nodes]
    e_right = e[j_nodes]

    # Exact analytical integration: (h/3)[e_i² + e_i·e_{i+1} + e_{i+1}²]
    element_errors_sq = (h / 3.0) * (e_left**2 + e_left * e_right + e_right**2)

    return np.sqrt(np.sum(element_errors_sq))


def linf_error(mesh: Mesh, u_nodal: np.ndarray, u_exact, n_sample: int = 10) -> float:
    """
    Compute global L-infinity error ||u_h - u_exact||_inf.
    """
    x_left, x_right, _ = _element_edges_1d(mesh)

    u_left = u_nodal[mesh.EToV[:, 0]]
    u_right = u_nodal[mesh.EToV[:, 1]]

    if u_left.ndim == 1:
        u_left = u_left[:, None]
        u_right = u_right[:, None]

    pts = np.linspace(-1, 1, n_sample)
    max_err = 0.0

    for pt in pts:
        x_phys = x_left + 0.5 * (pt + 1) * (x_right - x_left)
        u_ex = np.asarray(u_exact(x_phys)).reshape(-1)

        t = 0.5 * (pt + 1)
        u_h = ((1 - t) * u_left + t * u_right).reshape(-1)

        current_max = np.max(np.abs(u_h - u_ex))
        if current_max > max_err:
            max_err = current_max
    return max_err
