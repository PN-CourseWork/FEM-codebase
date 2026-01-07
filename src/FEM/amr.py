import numpy as np
from numba import njit

from .mesh import Mesh


@njit
def _compute_refinement_errors(VX, EToV, u_left, u_right, u_mid):
    """Compute Δerr_i = |u(x_mid) - 0.5*(u(x1) + u(x2))| * sqrt(h/3)"""
    n_elem = len(EToV)
    errors = np.empty(n_elem)

    for e in range(n_elem):
        n1, n2 = EToV[e]
        h = VX[n2] - VX[n1]
        delta = u_mid[e] - 0.5 * (u_left[e] + u_right[e])
        errors[e] = np.abs(delta) * np.sqrt(h / 3.0)

    return errors


def refinement_error(mesh: Mesh, func) -> np.ndarray:
    """Compute refinement error estimates for all elements."""
    VX, EToV = mesh.VX, mesh.EToV

    x_left = VX[EToV[:, 0]]
    x_right = VX[EToV[:, 1]]
    x_mid = 0.5 * (x_left + x_right)

    u_left = func(x_left)
    u_right = func(x_right)
    u_mid = func(x_mid)

    return _compute_refinement_errors(VX, EToV, u_left, u_right, u_mid)


@njit
def _build_refined_mesh(VX, EToV, marked_mask):
    """Build refined mesh arrays."""
    n_elem = len(EToV)
    n_nodes = len(VX)
    n_marked = np.sum(marked_mask)

    n_new_nodes = n_nodes + n_marked
    n_new_elem = n_elem + n_marked

    new_VX = np.empty(n_new_nodes)
    new_EToV = np.empty((n_new_elem, 2), dtype=np.int64)

    # Copy original nodes
    for i in range(n_nodes):
        new_VX[i] = VX[i]

    # Add midpoints and build elements
    next_node = n_nodes
    next_elem = 0

    for e in range(n_elem):
        n1, n2 = EToV[e, 0], EToV[e, 1]

        if marked_mask[e]:
            x_mid = 0.5 * (VX[n1] + VX[n2])
            mid_idx = next_node
            new_VX[mid_idx] = x_mid
            next_node += 1

            new_EToV[next_elem, 0] = n1
            new_EToV[next_elem, 1] = mid_idx
            next_elem += 1

            new_EToV[next_elem, 0] = mid_idx
            new_EToV[next_elem, 1] = n2
            next_elem += 1
        else:
            new_EToV[next_elem, 0] = n1
            new_EToV[next_elem, 1] = n2
            next_elem += 1

    return new_VX, new_EToV


@njit
def _reorder_mesh_1d(VX, EToV):
    """Reorder nodes by coordinate and update connectivity."""
    n_nodes = len(VX)
    n_elem = len(EToV)

    order = np.argsort(VX)

    inverse = np.empty(n_nodes, dtype=np.int64)
    for i in range(n_nodes):
        inverse[order[i]] = i

    new_VX = np.empty(n_nodes)
    for i in range(n_nodes):
        new_VX[i] = VX[order[i]]

    new_EToV = np.empty((n_elem, 2), dtype=np.int64)
    for e in range(n_elem):
        new_EToV[e, 0] = inverse[EToV[e, 0]]
        new_EToV[e, 1] = inverse[EToV[e, 1]]

    return new_VX, new_EToV


def mark_elements(errors: np.ndarray, alpha: float = 0.5, tol: float = None) -> np.ndarray:
    """
    Mark elements for refinement.

    Criteria:
        tol=None:  err > α * max(err)  (relative to max)
        tol=value: err > α * tol       (absolute threshold)
    """
    if tol is None:
        threshold = alpha * np.max(errors)
    else:
        threshold = alpha * tol
    return np.where(errors > threshold)[0]


def refine(mesh: Mesh, marked) -> Mesh:
    """Refine marked elements by splitting at midpoint."""
    marked = np.asarray(marked)
    if len(marked) == 0:
        return mesh

    marked_mask = np.zeros(mesh.n_elem, dtype=np.bool_)
    marked_mask[marked] = True

    new_VX, new_EToV = _build_refined_mesh(mesh.VX, mesh.EToV, marked_mask)
    new_VX, new_EToV = _reorder_mesh_1d(new_VX, new_EToV)

    return Mesh(new_VX, new_EToV)
