"""FEM package for 2D finite element methods.

This package implements P1 (linear) triangular finite elements for solving
2D Poisson-type PDEs with Dirichlet and Neumann boundary conditions.

Main components:
- Mesh2d: 2D triangular mesh generation
- assembly_2d: Global stiffness matrix and load vector assembly
- dirbc_2d, neubc_2d: Boundary condition application
- solve_mixed_bc_2d, solve_dirichlet_bc_2d: High-level solvers
"""

from .datastructures import (
    Mesh2d,
    LEFT,
    RIGHT,
    BOTTOM,
    TOP,
    BOUNDARY_TOL,
    EDGE_VERTICES,
    outernormal,
)
from .assembly import assembly_2d
from .boundary import (
    dirbc_2d,
    neubc_2d,
    get_boundary_nodes,
    get_boundary_edges,
    get_edge_midpoints,
)
from .solvers import (
    solve_mixed_bc_2d,
    solve_dirichlet_bc_2d,
    Driver28b,
    Driver28c,
)

__all__ = [
    # Mesh
    "Mesh2d",
    "LEFT",
    "RIGHT",
    "BOTTOM",
    "TOP",
    "BOUNDARY_TOL",
    "EDGE_VERTICES",
    "outernormal",
    # Assembly
    "assembly_2d",
    # Boundary conditions
    "dirbc_2d",
    "neubc_2d",
    "get_boundary_nodes",
    "get_boundary_edges",
    "get_edge_midpoints",
    # Solvers
    "solve_mixed_bc_2d",
    "solve_dirichlet_bc_2d",
    "Driver28b",
    "Driver28c",
]
