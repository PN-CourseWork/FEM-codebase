"""
DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment - Exercise 1.7

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import spsolve

from FEM import (
    discrete_l2_error,
    refine,
    estimate_error_l2,
    assemble_diffusion,
    assemble_mass,
    assemble_load,
    apply_dirichlet,
    setup_style,
    save_figure,
    line_mesh,
)
from FEM.amr import mark_elements
from FEM.datastructures import Mesh

setup_style()


# Problem definition
def u_exact(x):
    return np.exp(-800 * (x - 0.4) ** 2) + 0.25 * np.exp(-40 * (x - 0.8) ** 2)


f_source = lambda x: (
    -1601 * np.exp(-800 * (x - 0.4) ** 2)
    + (-1600 * x + 640.0) ** 2 * np.exp(-800 * (x - 0.4) ** 2)
    - 20.25 * np.exp(-40 * (x - 0.8) ** 2)
    + 0.25 * (-80 * x + 64.0) ** 2 * np.exp(-40 * (x - 0.8) ** 2)
)

C_VAL, D_VAL = u_exact(0.0), u_exact(1.0)


def solve_bvp(mesh):
    """Solve the reaction-diffusion BVP on given mesh."""
    A = assemble_diffusion(mesh) + assemble_mass(mesh)
    b = assemble_load(mesh, lambda x: -f_source(x), n_quad=5)
    apply_dirichlet(A, b, 0, C_VAL)
    apply_dirichlet(A, b, mesh.nonodes - 1, D_VAL)
    return spsolve(A, b)


def run_amr_study(tol):
    """Run AMR study and return final mesh, solution, and stats DataFrame."""
    print(f"Starting AMR Study (tol={tol})...")

    x_init = np.array([0.0, 0.5, 1.0])
    EToV = np.array([[i, i + 1] for i in range(len(x_init) - 1)])
    mesh_c = Mesh(VX=x_init, EToV=EToV)
    u_c = solve_bvp(mesh_c)

    mesh_f, parent_map = refine(mesh_c, np.arange(mesh_c.noelms))
    u_f = solve_bvp(mesh_f)

    # JIT warmup for estimate_error_l2
    _ = estimate_error_l2(mesh_f, u_f, mesh_c, u_c, parent_map)

    stats = []
    cum_time = 0.0
    cum_dof = mesh_c.nonodes + mesh_f.nonodes  # Count initial solve DOF

    for iteration in range(100):
        t0 = time.time()

        errors = estimate_error_l2(mesh_f, u_f, mesh_c, u_c, parent_map)
        error_est = np.max(errors)
        error_true = discrete_l2_error(mesh_c, u_c, u_exact)

        converged = error_est < tol
        marked = [] if converged else mark_elements(errors, alpha=1.0, tol=tol)

        if not converged and len(marked) > 0:
            mesh_c, _ = refine(mesh_c, marked)
            u_c = solve_bvp(mesh_c)
            mesh_f, parent_map = refine(mesh_c, np.arange(mesh_c.noelms))
            u_f = solve_bvp(mesh_f)
            # Accumulate DOF only when we actually solve
            cum_dof += mesh_c.nonodes + mesh_f.nonodes

        cum_time += time.time() - t0

        stats.append(
            {
                "iteration": iteration,
                "dof": cum_dof,  # Cumulative DOF across all iterations
                "error_est": error_est,
                "error_true": error_true,
                "cumulative_time": cum_time,
            }
        )

        if converged or len(marked) == 0:
            break

    return mesh_c, u_c, pd.DataFrame(stats)


def solve_uniform_fem(n_elem):
    """Solve with uniform mesh, return (dof, error, time)."""
    t0 = time.time()
    mesh = line_mesh(1.0, n_elem)
    u = solve_bvp(mesh)
    error_l2 = discrete_l2_error(mesh, u, u_exact)
    return mesh.nonodes, error_l2, time.time() - t0


# Run studies
TOL = 1e-4
mesh_final, u_final, df_amr = run_amr_study(TOL)
print(f"\nAMR: {df_amr['dof'].iloc[-1]} DOF, error={df_amr['error_true'].iloc[-1]:.2e}")

# JIT warmup for uniform FEM
print("Warming up JIT for uniform FEM...")
_ = solve_uniform_fem(10)

n_elem_list = np.logspace(1, 4, 25, dtype=int)
# Limit to max 1e4 DOF (DOF = n_elem + 1 in 1D)
n_elem_list = n_elem_list[n_elem_list + 1 <= 10000]
uniform_results = [solve_uniform_fem(n) for n in n_elem_list]
uniform_dofs, uniform_errors, uniform_times = zip(*uniform_results)

# Plot 1: Solution
fig, ax = plt.subplots(figsize=(8, 5))
x_fine = np.linspace(0, 1, 1000)
ax.plot(x_fine, u_exact(x_fine), "k-", label="Exact", alpha=0.6)
ax.scatter(
    mesh_final.VX, u_final, s=6, color="r", label=f"FEM ({mesh_final.noelms} elem)"
)
ax.set_title(f"AMR Solution (tol={TOL})")
ax.legend()
save_figure(fig, "figures/A1/ex_7/ex_7_solution.pdf")

# Plot 2: Mesh distribution
fig, ax = plt.subplots(figsize=(8, 5))
h = np.diff(mesh_final.VX)
x_mid = 0.5 * (mesh_final.VX[:-1] + mesh_final.VX[1:])
ax.bar(x_mid, h, width=h, edgecolor="k", align="center")
ax.set_yscale("log")
ax.set_xlabel("x")
ax.set_ylabel("Element Size h")
ax.set_title("Final Mesh Element Distribution")
save_figure(fig, "figures/A1/ex_7/ex_7_mesh.pdf")

# Plot 3: Convergence
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(df_amr["dof"], df_amr["error_true"], "o-", label="AMR")
ax.loglog(uniform_dofs, uniform_errors, "s--", color="gray", alpha=0.8, label="Uniform")
dof_ref = np.array([uniform_dofs[0], uniform_dofs[-1]])
ax.loglog(
    dof_ref,
    uniform_errors[0] * (dof_ref / dof_ref[0]) ** -2,
    "k:",
    label=r"$O(N^{-2})$",
)
ax.set_xlabel("DOF")
ax.set_ylabel("L2 Error")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
save_figure(fig, "figures/A1/ex_7/ex_7_convergence.pdf")

# Plot 4: CPU time vs error
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(df_amr["error_true"], df_amr["cumulative_time"], "o-", label="AMR")
ax.loglog(
    uniform_errors, uniform_times, "s--", color="gray", alpha=0.8, label="Uniform"
)
ax.set_xlabel("L2 Error")
ax.set_ylabel("CPU Time (s)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.invert_xaxis()
save_figure(fig, "figures/A1/ex_7/ex_7_cpu_time.pdf")

# Plot 5: CPU time vs DOF
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(df_amr["dof"], df_amr["cumulative_time"], "o-", label="AMR")
ax.loglog(uniform_dofs, uniform_times, "s--", color="gray", alpha=0.8, label="Uniform")
ax.set_xlabel("DOF")
ax.set_ylabel("CPU Time (s)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
save_figure(fig, "figures/A1/ex_7/ex_7_cpu_time_vs_dof.pdf")

# Plot 6: Error distribution
fig, ax = plt.subplots(figsize=(8, 4))
mesh_ref, parent_map = refine(mesh_final, np.arange(mesh_final.noelms))
u_ref = solve_bvp(mesh_ref)
est_errors = estimate_error_l2(mesh_ref, u_ref, mesh_final, u_final, parent_map)
x_mid = 0.5 * (mesh_final.VX[:-1] + mesh_final.VX[1:])
ax.plot(x_mid, est_errors, "b.-")
ax.set_xlabel("x")
ax.set_ylabel("Estimated Error")
ax.set_yscale("log")
save_figure(fig, "figures/A1/ex_7/ex_7_error_distribution.pdf")
