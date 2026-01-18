"""Spectral Element Method for 2D PDEs on quad meshes.

Example
-------
>>> from FEM.SEM import SEMMesh2D, solve_poisson_sem, l2_error_sem
>>>
>>> mesh = SEMMesh2D(filepath="mesh.msh", polynomial_order=4)
>>> u = solve_poisson_sem(mesh, f_func, bc_func)
>>> error = l2_error_sem(mesh, u, u_exact)
"""

from .mesh import SEMMesh2D, load_gmsh_quads, create_unit_square_mesh
from .assembly import (
    assemble_global_stiffness,
    assemble_global_mass,
    assemble_load_vector,
    apply_dirichlet_bc,
    apply_dirichlet_bc_func,
)
from .solvers import solve_poisson_sem, l2_error_sem, linf_error_sem, h1_seminorm_error_sem

__all__ = [
    "SEMMesh2D",
    "load_gmsh_quads",
    "create_unit_square_mesh",
    "assemble_global_stiffness",
    "assemble_global_mass",
    "assemble_load_vector",
    "apply_dirichlet_bc",
    "apply_dirichlet_bc_func",
    "solve_poisson_sem",
    "l2_error_sem",
    "linf_error_sem",
    "h1_seminorm_error_sem",
]
