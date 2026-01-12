import numpy as np
import scipy.sparse.linalg as spla

from .assembly import (
    assemble_diffusion,
    assemble_diffusion_mass,
    assemble_advection,
    assemble_load,
)
from .boundary import apply_dirichlet
from .datastructures import Mesh
from .mesh import line_mesh


def solve_advection_diffusion_1d(mesh: Mesh, eps: float, psi: float, f_func):
    """
    Solve -eps*u'' + psi*u' = f on 1D mesh with homogeneous Dirichlet BCs (ends).
    """
    A = assemble_diffusion(mesh, lam=eps) + assemble_advection(mesh, psi)
    b = assemble_load(mesh, f_func)

    apply_dirichlet(A, b, 0, 0.0)
    apply_dirichlet(A, b, mesh.nonodes - 1, 0.0)

    u = spla.spsolve(A, b)
    return u, A, b


def solve_bvp_1d(L: float, c: float, d: float, n_elem: int):
    """
    Solve u'' - u = 0 on [0, L] with u(0)=c, u(L)=d using linear FEM.
    Follows the Matlab BVP1D signature: length and boundary values in, solution out.
    """
    mesh = line_mesh(L, n_elem)
    u, A, b = solve_reaction_diffusion_1d(
        mesh, lam=1.0, reaction=1.0, dirichlet_bc=([0, mesh.nonodes - 1], [c, d])
    )
    return mesh, u, A, b


def solve_reaction_diffusion_1d(mesh: Mesh, lam: float, reaction: float, dirichlet_bc):
    """
    Solve -lam*u'' + reaction*u = 0 on a 1D mesh with Dirichlet BCs.
    dirichlet_bc: (nodes, values)
    """
    A = assemble_diffusion_mass(mesh, lam=lam, coeff=reaction)
    b = np.zeros(mesh.nonodes)

    nodes, vals = dirichlet_bc
    apply_dirichlet(A, b, nodes, vals)

    u = spla.spsolve(A, b)
    return u, A, b


def solve_reaction_diffusion_1d_amr_hierarchical(
    L: float,
    c: float,
    d: float,
    x: np.ndarray,
    f_func,
    tol: float,
    max_dof: int = 2000,
    max_iter: int = None,
):
    from .amr import mark_elements, estimate_error_l2, refine
    from .datastructures import Mesh

    n_elem = len(x) - 1
    EToV = np.array([[i, i + 1] for i in range(n_elem)])
    mesh_coarse = Mesh(VX=x, EToV=EToV)

    def solve_bvp(m):
        """Solve -u'' + u = f with Dirichlet BCs."""
        A = assemble_diffusion_mass(m, lam=1.0, coeff=1.0)
        b = assemble_load(m, lambda x: -f_func(x))
        apply_dirichlet(A, b, np.array([0, m.nonodes - 1]), np.array([c, d]))
        return spla.spsolve(A, b)

    # First iteration: solve on coarse, then refine ALL and solve on fine
    u_coarse = solve_bvp(mesh_coarse)
    marked_all = np.arange(mesh_coarse.noelms)
    mesh_fine, parent_map = refine(mesh_coarse, marked_all)
    u_fine = solve_bvp(mesh_fine)

    stats = []
    iteration = 0

    while True:
        # Estimate errors on coarse mesh by comparing with fine
        errors = estimate_error_l2(mesh_fine, u_fine, mesh_coarse, u_coarse, parent_map)
        error_metric = np.max(errors)  # Use max for absolute marking consistency

        # Record statistics (report coarse DOF since that's what we're refining)
        """
        stats.append(
            {
                "iteration": iteration,
                "dof": mesh_coarse.nonodes,
                "error_est": error_metric,
            }
        )
        """

        # Check stopping criteria
        reached_tol = tol is not None and error_metric < tol
        reached_dof = max_dof is not None and mesh_coarse.nonodes >= max_dof
        reached_iter = max_iter is not None and iteration >= max_iter

        if reached_tol or reached_dof or reached_iter:
            # Return coarse mesh (it's the one we check convergence on)
            break

        # Mark elements on coarse mesh for refinement (absolute marking)
        marked = mark_elements(errors, alpha=1.0, tol=tol)

        if len(marked) == 0:
            break

        mesh_coarse, parent_map_marked = refine(mesh_coarse, marked)

        u_coarse = solve_bvp(mesh_coarse)

        # Then refine ALL elements of new coarse to get new fine (for error estimation)
        marked_all = np.arange(mesh_coarse.noelms)
        mesh_fine, parent_map = refine(mesh_coarse, marked_all)
        u_fine = solve_bvp(mesh_fine)

        iteration += 1

    return mesh_coarse, u_coarse, stats
