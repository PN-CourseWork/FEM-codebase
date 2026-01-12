"""
AMR solver for Exercise 1.7 (Sustainability Benchmarking).

DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
Date: January 2026
"""

from FEM import solve_reaction_diffusion_1d_amr_hierarchical


def DriverAMR17(L, c, d, x, func, tol, maxit):
    """
    AMR solver for 1D BVP: -u'' + u = f on [0,L] (Exercise 1.7).

    Uses hierarchical error estimation for efficiency.

    Parameters:
        L: Domain length
        c, d: Dirichlet BCs at x=0 and x=L
        x: Initial mesh nodes
        func: Source term f(x) = u''(x) - u(x)
        tol: Tolerance for AMR
        maxit: Maximum iterations

    Returns:
        xAMR: Final mesh nodes
        u: Final solution
        iter: Number of iterations
    """
    mesh_final, u_final, stats = solve_reaction_diffusion_1d_amr_hierarchical(
        L=L, c=c, d=d, x=x, f_func=func, tol=tol, max_dof=2000, max_iter=maxit
    )
    return mesh_final.VX, u_final, len(stats)
