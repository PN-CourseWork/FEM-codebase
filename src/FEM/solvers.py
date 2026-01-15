from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import scipy.sparse.linalg as spla

from .assembly import assembly_2d
from .boundary import dirbc_2d, get_boundary_edges, get_edge_midpoints, neubc_2d
from .datastructures import BOTTOM, BOUNDARY_TOL, LEFT, Mesh2d


def _apply_neumann(
    mesh: Mesh2d,
    b: NDArray[np.float64],
    side: int,
    q_func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Apply Neumann BC on a boundary side."""
    beds = get_boundary_edges(mesh, side)
    midpts = get_edge_midpoints(beds, mesh)
    q = q_func(midpts[:, 0], midpts[:, 1])
    return neubc_2d(beds, q, mesh, b)


def solve_mixed_bc_2d(
    mesh: Mesh2d,
    q_tilde_func: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    q_left: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    q_bottom: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    f_dirichlet: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    lam1: float = 1,
    lam2: float = 1,
) -> NDArray[np.float64]:
    """Solve 2D BVP with mixed BCs: Neumann on left/bottom, Dirichlet on right/top."""
    qt = q_tilde_func(mesh.VX, mesh.VY)
    A, b = assembly_2d(mesh, qt, lam1, lam2)

    b = _apply_neumann(mesh, b, LEFT, q_left)
    b = _apply_neumann(mesh, b, BOTTOM, q_bottom)

    right_nodes = np.where(np.abs(mesh.VX - (mesh.x0 + mesh.L1)) < BOUNDARY_TOL)[0] + 1
    top_nodes = np.where(np.abs(mesh.VY - (mesh.y0 + mesh.L2)) < BOUNDARY_TOL)[0] + 1
    gamma2_nodes = np.unique(np.concatenate([right_nodes, top_nodes]))
    f = f_dirichlet(mesh.VX[gamma2_nodes - 1], mesh.VY[gamma2_nodes - 1])

    A, b = dirbc_2d(gamma2_nodes, f, A, b)
    return spla.spsolve(A, b)


def _q_zero(x: NDArray[np.float64], _y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Zero Neumann boundary condition."""
    return np.zeros_like(x)


def solve_dirichlet_bc_2d(
    mesh: Mesh2d,
    q_tilde_func: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    f_dirichlet: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
    lam1: float = 1,
    lam2: float = 1,
) -> NDArray[np.float64]:
    """Solve 2D BVP with Dirichlet BCs on all boundaries."""
    from .boundary import get_boundary_nodes

    qt = q_tilde_func(mesh.VX, mesh.VY)
    A, b = assembly_2d(mesh, qt, lam1, lam2)

    bnodes = get_boundary_nodes(mesh)
    f = f_dirichlet(mesh.VX[bnodes - 1], mesh.VY[bnodes - 1])

    A, b = dirbc_2d(bnodes, f, A, b)
    return spla.spsolve(A, b)


def Driver28b(
    x0: float,
    y0: float,
    L1: float,
    L2: float,
    noelms1: int,
    noelms2: int,
    lam1: float,
    lam2: float,
    fun: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    qt: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]
]:
    """28b: Quarter domain with mixed BCs (Neumann left/bottom, Dirichlet right/top)."""
    mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=noelms1, noelms2=noelms2)
    U = solve_mixed_bc_2d(mesh, qt, _q_zero, _q_zero, fun, lam1=lam1, lam2=lam2)
    return mesh.VX, mesh.VY, mesh.EToV - 1, U


def Driver28c(
    x0: float,
    y0: float,
    L1: float,
    L2: float,
    noelms1: int,
    noelms2: int,
    lam1: float,
    lam2: float,
    fun: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    qt: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]
]:
    """28c: Full domain with pure Dirichlet BCs on all boundaries."""
    mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=noelms1, noelms2=noelms2)
    U = solve_dirichlet_bc_2d(mesh, qt, fun, lam1=lam1, lam2=lam2)
    return mesh.VX, mesh.VY, mesh.EToV - 1, U
