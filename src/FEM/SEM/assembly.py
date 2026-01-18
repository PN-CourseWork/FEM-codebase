"""Global assembly for Spectral Element Method."""

from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .mesh import SEMMesh2D
from .spectral import stiffness_matrix_2d, mass_matrix_2d


def assemble_global_stiffness(mesh: SEMMesh2D) -> sparse.csr_matrix:
    """Assemble global stiffness matrix for Laplacian operator.

    For axis-aligned square elements, gradient scaling cancels with Jacobian,
    so K_e = K_ref for all elements.
    """
    noelms, nloc = mesh.noelms, mesh.nloc
    ref = mesh.ref_elem
    K_ref = stiffness_matrix_2d(ref.Dx, ref.Dy, ref.wx, ref.wy).toarray()

    # Broadcast to all elements
    K_all = np.broadcast_to(K_ref[np.newaxis, :, :], (noelms, nloc, nloc)).copy()

    # Build global index arrays
    glb = mesh.loc2glb
    rows = np.broadcast_to(glb[:, :, np.newaxis], (noelms, nloc, nloc))
    cols = np.broadcast_to(glb[:, np.newaxis, :], (noelms, nloc, nloc))

    return sparse.csr_matrix(
        (K_all.ravel(), (rows.ravel(), cols.ravel())),
        shape=(mesh.nonodes, mesh.nonodes),
    )


def assemble_global_mass(mesh: SEMMesh2D) -> sparse.csr_matrix:
    """Assemble global (lumped) mass matrix. Returns diagonal matrix."""
    ref = mesh.ref_elem
    M_ref_diag = mass_matrix_2d(ref.wx, ref.wy)
    mass_diag = np.zeros(mesh.nonodes, dtype=np.float64)

    for e in range(mesh.noelms):
        np.add.at(mass_diag, mesh.loc2glb[e], mesh.jacobians[e] * M_ref_diag)

    return sparse.diags(mass_diag, format="csr")


def assemble_load_vector(
    mesh: SEMMesh2D,
    f_func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Assemble load vector b_i = ∫_Ω f φ_i dΩ."""
    b = np.zeros(mesh.nonodes, dtype=np.float64)
    ref = mesh.ref_elem
    w_ref = mass_matrix_2d(ref.wx, ref.wy)

    for e in range(mesh.noelms):
        glb = mesh.loc2glb[e]
        f_e = f_func(mesh.VX[glb], mesh.VY[glb])
        np.add.at(b, glb, mesh.jacobians[e] * w_ref * f_e)

    return b


def apply_dirichlet_bc(
    A: sparse.csr_matrix,
    b: NDArray[np.float64],
    bc_nodes: NDArray[np.int64],
    bc_values: NDArray[np.float64],
) -> tuple[sparse.csr_matrix, NDArray[np.float64]]:
    """Apply Dirichlet BC by zeroing rows/cols and setting diagonal to 1."""
    n = A.shape[0]
    A_csr = A.tocsr()
    b_mod = b.copy()

    # Modify RHS
    f_full = np.zeros(n, dtype=np.float64)
    f_full[bc_nodes] = bc_values
    b_mod -= A_csr @ f_full
    b_mod[bc_nodes] = bc_values

    # Scale rows and columns to zero for BC nodes
    scale = np.ones(n, dtype=np.float64)
    scale[bc_nodes] = 0.0
    row_scale = np.repeat(scale, np.diff(A_csr.indptr))
    col_scale = scale[A_csr.indices]

    A_mod = A_csr.copy()
    A_mod.data *= row_scale * col_scale

    # Set diagonal to 1
    diag = A_mod.diagonal()
    diag[bc_nodes] = 1.0
    A_mod.setdiag(diag)

    return A_mod, b_mod


def apply_dirichlet_bc_func(
    A: sparse.csr_matrix,
    b: NDArray[np.float64],
    mesh: SEMMesh2D,
    bc_func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    sides: list[str] | None = None,
) -> tuple[sparse.csr_matrix, NDArray[np.float64]]:
    """Apply Dirichlet BC using a boundary function g(x, y)."""
    if sides is None:
        bc_nodes = mesh.get_boundary_nodes("all")
    else:
        bc_nodes = np.unique(np.concatenate([mesh.get_boundary_nodes(s) for s in sides]))

    bc_values = bc_func(mesh.VX[bc_nodes], mesh.VY[bc_nodes])
    return apply_dirichlet_bc(A, b, bc_nodes, bc_values)
