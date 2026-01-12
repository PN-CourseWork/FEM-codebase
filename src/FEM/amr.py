import numpy as np
from typing import Tuple, Callable
from numba import njit

from .datastructures import Mesh


@njit
def _refine_1d_sorted(VX, EToV, marked_mask):
    """
    Optimized refinement for 1D meshes that maintains sorted order.
    """
    n_elem = len(EToV)
    n_marked = np.sum(marked_mask)

    n_new_elem = n_elem + n_marked
    n_new_nodes = len(VX) + n_marked

    new_VX = np.empty(n_new_nodes, dtype=VX.dtype)
    new_EToV = np.empty((n_new_elem, 2), dtype=EToV.dtype)
    parent_map = np.empty(n_new_elem, dtype=np.int64)

    vx_idx = 0
    elem_idx = 0

    for e in range(n_elem):
        # Add left node
        new_VX[vx_idx] = VX[e]
        vx_idx += 1

        if marked_mask[e]:
            # Refine element e
            # Create midpoint
            mid = 0.5 * (VX[e] + VX[e + 1])
            new_VX[vx_idx] = mid
            vx_idx += 1

            # Create 2 new elements
            # Elem 1: [current_node_idx-1, current_node_idx] (Left -> Mid)
            new_EToV[elem_idx, 0] = vx_idx - 2
            new_EToV[elem_idx, 1] = vx_idx - 1
            parent_map[elem_idx] = e
            elem_idx += 1

            # Elem 2: [current_node_idx, current_node_idx+1] (Mid -> Right)
            new_EToV[elem_idx, 0] = vx_idx - 1
            new_EToV[elem_idx, 1] = vx_idx
            parent_map[elem_idx] = e
            elem_idx += 1
        else:
            # Copy element e
            # It connects [vx_idx-1, vx_idx]
            new_EToV[elem_idx, 0] = vx_idx - 1
            new_EToV[elem_idx, 1] = vx_idx
            parent_map[elem_idx] = e
            elem_idx += 1

    # Add the very last node
    new_VX[vx_idx] = VX[-1]

    return new_VX, new_EToV, parent_map


def refine(mesh: Mesh, marked: np.ndarray) -> Tuple[Mesh, np.ndarray]:
    """Refine marked elements by splitting them at the midpoint (1D only)."""
    marked = np.asarray(marked)

    # Create boolean mask for efficient Numba lookup
    marked_mask = np.zeros(mesh.noelms, dtype=bool)
    marked_mask[marked] = True

    new_VX, new_cells, parent_map = _refine_1d_sorted(
        mesh.VX, mesh.EToV, marked_mask
    )
    return Mesh(VX=new_VX, EToV=new_cells), parent_map


@njit
def _estimate_error_l2_core(
    x_L, h_fine, u_fine_L, u_fine_R, xc_L, hc, uc_L, uc_R, parent_map, n_coarse, n_fine
):
    """
    Compute L2 error using exact analytical integration.

    """
    error_indicators = np.zeros(n_coarse)
    fine_elem_errors = np.zeros(n_fine)

    for i in range(n_fine):
        # Fine element data
        xl = x_L[i]
        hf = h_fine[i]
        ufl = u_fine_L[i]
        ufr = u_fine_R[i]

        # Coarse parent data
        xcl = xc_L[i]
        ucl = uc_L[i]
        ucr = uc_R[i]

        # Compute error at left and right nodes of fine element
        t_L = (xl - xcl) / hc[i]
        u_coarse_L = (1.0 - t_L) * ucl + t_L * ucr

        # Right endpoint: x = xl + hf
        x_R = xl + hf
        t_R = (x_R - xcl) / hc[i]
        u_coarse_R = (1.0 - t_R) * ucl + t_R * ucr

        # Error at nodes
        e_L = ufl - u_coarse_L
        e_R = ufr - u_coarse_R

        elem_err_sq = (hf / 3.0) * (e_L**2 + e_L * e_R + e_R**2)

        fine_elem_errors[i] = elem_err_sq

    # Accumulate into coarse elements
    for i in range(n_fine):
        error_indicators[parent_map[i]] += fine_elem_errors[i]

    return np.sqrt(error_indicators)


@njit
def _estimate_error_l2_extract(
    VX_fine: np.ndarray,
    EToV_fine: np.ndarray,
    u_fine: np.ndarray,
    VX_coarse: np.ndarray,
    EToV_coarse: np.ndarray,
    u_coarse: np.ndarray,
    parent_map: np.ndarray,
) -> np.ndarray:
    """
    JIT-compiled A posteriori Error Estimator (Two-Grid L2 Norm).
    """
    n_fine = EToV_fine.shape[0]
    n_coarse = EToV_coarse.shape[0]

    error_indicators = np.zeros(n_coarse)
    fine_elem_errors = np.zeros(n_fine)

    for i in range(n_fine):
        # Fine element data
        node_L = EToV_fine[i, 0]
        node_R = EToV_fine[i, 1]
        xl = VX_fine[node_L]
        xr = VX_fine[node_R]
        hf = xr - xl
        ufl = u_fine[node_L]
        ufr = u_fine[node_R]

        # Coarse parent data
        parent = parent_map[i]
        parent_node_L = EToV_coarse[parent, 0]
        parent_node_R = EToV_coarse[parent, 1]
        xcl = VX_coarse[parent_node_L]
        xcr = VX_coarse[parent_node_R]
        hc = xcr - xcl
        ucl = u_coarse[parent_node_L]
        ucr = u_coarse[parent_node_R]

        # Compute error at left and right nodes of fine element
        t_L = (xl - xcl) / hc
        u_coarse_L = (1.0 - t_L) * ucl + t_L * ucr

        t_R = (xr - xcl) / hc
        u_coarse_R = (1.0 - t_R) * ucl + t_R * ucr

        # Error at nodes
        e_L = ufl - u_coarse_L
        e_R = ufr - u_coarse_R

        elem_err_sq = (hf / 3.0) * (e_L**2 + e_L * e_R + e_R**2)
        fine_elem_errors[i] = elem_err_sq

    # Accumulate into coarse elements
    for i in range(n_fine):
        error_indicators[parent_map[i]] += fine_elem_errors[i]

    return np.sqrt(error_indicators)


def estimate_error_l2(
    mesh_fine: Mesh,
    u_fine: np.ndarray,
    mesh_coarse: Mesh,
    u_coarse: np.ndarray,
    parent_map: np.ndarray,
) -> np.ndarray:
    """
    A posteriori Error Estimator (Two-Grid L2 Norm).
    """
    return _estimate_error_l2_extract(
        mesh_fine.VX,
        mesh_fine.EToV,
        u_fine,
        mesh_coarse.VX,
        mesh_coarse.EToV,
        u_coarse,
        parent_map,
    )


def refinement_error(mesh: Mesh, func) -> np.ndarray:
    """
    Compute refinement error estimates for all elements.
    """
    cells = mesh.EToV
    VX = mesh.VX
    x_left = VX[cells[:, 0]][:, None]
    x_right = VX[cells[:, 1]][:, None]
    x_mid = 0.5 * (x_left + x_right)

    h = np.linalg.norm(x_right - x_left, axis=1, keepdims=True)

    u_left = func(x_left)
    u_right = func(x_right)
    u_mid = func(x_mid)

    # Difference between exact solution and linear interpolant at the midpoint
    delta = np.abs(u_mid - 0.5 * (u_left + u_right))

    # Estimate L2 error on the element
    return delta * np.sqrt(h / 3.0)


def mark_elements(
    errors: np.ndarray, alpha: float = 0.5, tol: float = None
) -> np.ndarray:
    """Return indices of elements to refine."""
    threshold = alpha * (tol if tol is not None else np.max(errors))
    return np.where(errors > threshold)[0]


def run_amr(
    mesh: Mesh,
    solve_fn: Callable[[Mesh], np.ndarray],
    estimator_fn: Callable[[Mesh, np.ndarray], np.ndarray],
    marker_fn: Callable[[np.ndarray], np.ndarray],
    metric_fn: Callable[[np.ndarray], float] = lambda e: np.linalg.norm(e),
    stop_fn: Callable[
        [Mesh, np.ndarray, np.ndarray, float, dict, list[dict]], bool
    ] = None,
    record_fn: Callable[[Mesh, np.ndarray, np.ndarray, float], dict] = None,
    tol: float = None,
    max_dof: int = None,
    max_iter: int = None,
) -> Tuple[Mesh, np.ndarray, list[dict]]:
    stats: list[dict] = []
    iteration = 0

    while True:
        u = solve_fn(mesh)
        est_errors = estimator_fn(mesh, u)
        metric = metric_fn(est_errors)

        extra = record_fn(mesh, u, est_errors, metric) if record_fn else {}
        entry = {"iter": iteration, "dof": mesh.nonodes, "error_est": metric}
        entry.update(extra)
        stats.append(entry)

        reached_tol = tol is not None and metric < tol
        reached_dof = max_dof is not None and mesh.nonodes >= max_dof
        reached_iter = max_iter is not None and len(stats) >= max_iter
        custom_stop = (
            stop_fn(mesh, u, est_errors, metric, entry, stats) if stop_fn else False
        )
        if reached_tol or reached_dof or reached_iter or custom_stop:
            break

        marked = marker_fn(est_errors)
        if len(marked) == 0:
            break

        mesh, _ = refine(mesh, marked)
        iteration += 1

    return mesh, u, stats
