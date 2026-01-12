"""
DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment - Exercise 1.6

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from FEM import (
    refinement_error,
    mark_elements,
    interpolate,
    run_amr,
    setup_style,
    save_figure,
)

setup_style()


def u(x):
    return np.exp(-800 * (x - 0.4) ** 2) + 0.25 * np.exp(-40 * (x - 0.8) ** 2)


def run_config(alpha, use_relative, tol, x_init):
    marker = (
        (lambda err: mark_elements(err, alpha=alpha))
        if use_relative
        else (lambda err: mark_elements(err, alpha=alpha, tol=tol))
    )

    def record_l2(mesh, u, est, metric):
        return {"l2_est": np.linalg.norm(est)}

    # Create mesh from initial nodes (same as ex_7 and benchmark)
    from FEM.datastructures import Mesh

    n_elem = len(x_init) - 1
    EToV = np.column_stack([np.arange(n_elem), np.arange(1, n_elem + 1)])
    mesh0 = Mesh(VX=x_init, EToV=EToV)

    mesh_final, u_final, stats = run_amr(
        mesh0,
        lambda m: u(m.VX),
        lambda m, _u: refinement_error(m, u),
        marker,
        metric_fn=np.max,
        record_fn=record_l2,
        tol=tol,
    )
    df = pd.DataFrame(stats)
    df["criterion"] = f"{'Relative' if use_relative else 'Absolute'} (alpha={alpha})"
    return mesh_final, u_final, df


# Run AMR with different strategies
x_init = np.array(
    [0.0, 0.5, 1.0]
)  # Same initial mesh as benchmark and ex_7 (2 elements)
tol = 1e-4
alpha_rel = 0.95
alpha_abs = 1.00

mesh_rel, _, hist_rel = run_config(alpha_rel, True, tol, x_init)
mesh_abs, _, hist_abs = run_config(alpha_abs, False, tol, x_init)

hist_rel["iteration"] = range(len(hist_rel))
hist_abs["iteration"] = range(len(hist_abs))

# Assertions
final_err_rel = hist_rel["error_est"].iloc[-1]
assert final_err_rel < tol, (
    f"Relative strategy failed to reach tol: {final_err_rel} >= {tol}"
)

final_err_abs = hist_abs["error_est"].iloc[-1]
assert final_err_abs < tol, (
    f"Absolute strategy failed to reach tol: {final_err_abs} >= {tol}"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x_fine = np.linspace(0, 1, 1000)
u_exact = u(x_fine)

for ax, mesh, title in zip(axes, [mesh_rel, mesh_abs], ["Relative", "Absolute"]):
    x_nodes = mesh.VX
    u_nodal = u(x_nodes)
    u_interp = interpolate(mesh, u_nodal, x_fine)

    ax.plot(x_fine, u_exact, "k-", label="Exact")
    ax.plot(x_fine, u_interp, "--", label="Interpolant")
    ax.plot(x_nodes, u_nodal, "o", label="Nodes")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title(f"{title}: {mesh.noelms} elements")
    ax.legend(loc="upper right")

save_figure(fig, "figures/A1/ex_6/ex_6_solution.pdf")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, mesh, title in zip(axes, [mesh_rel, mesh_abs], ["Relative", "Absolute"]):
    x_nodes = mesh.VX
    h = np.diff(x_nodes)
    x_mid = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    ax.bar(x_mid, h, width=h * 0.9, edgecolor="black")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"Element size $h$")
    ax.set_title(f"{title}: {mesh.noelms} elements")
    ax.set_yscale("log")

save_figure(fig, "figures/A1/ex_6/ex_6_element_size.pdf")

df_perf = pd.concat([hist_rel, hist_abs], ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for crit, grp in df_perf.groupby("criterion"):
    axes[0].plot(grp["iteration"], grp["error_est"], marker="o", label=crit)
    axes[1].plot(grp["dof"], grp["error_est"], marker="o", label=crit)
    axes[2].plot(grp["dof"], grp["l2_est"], marker="o", label=crit)

axes[0].axhline(tol, color="red", linestyle="--", label=f"tol={tol}")
axes[0].set_yscale("log")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel(r"$\max_i \Delta\mathrm{err}_i$")
axes[0].set_title("Convergence (Max Error)")
axes[0].legend()

axes[1].axhline(tol, color="red", linestyle="--", label=f"tol={tol}")
axes[1].set_yscale("log")
axes[1].set_xlabel("Degrees of Freedom")
axes[1].set_ylabel(r"$\max_i \Delta\mathrm{err}_i$")
axes[1].set_title("Efficiency (Max Error)")
axes[1].legend()

axes[2].set_yscale("log")
axes[2].set_xlabel("Degrees of Freedom")
axes[2].set_ylabel(r"Global L2 Error Estimate")
axes[2].set_title("L2 Error Convergence")
axes[2].legend()

save_figure(fig, "figures/A1/ex_6/ex_6_performance.pdf")

# Plot Error Distribution
fig, ax = plt.subplots(figsize=(8, 5))

# Recompute element-wise errors for final meshes
err_rel = refinement_error(mesh_rel, u)
err_abs = refinement_error(mesh_abs, u)

x_mid_rel = 0.5 * (mesh_rel.VX[:-1] + mesh_rel.VX[1:])
x_mid_abs = 0.5 * (mesh_abs.VX[:-1] + mesh_abs.VX[1:])

ax.plot(x_mid_rel, err_rel, "o", label="Relative", alpha=0.7)
ax.plot(x_mid_abs, err_abs, "x", label="Absolute", alpha=0.7)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"Element Error Estimate")
ax.set_title("Error Distribution on Final Mesh")
ax.legend()
ax.set_yscale("log")

save_figure(fig, "figures/A1/ex_6/ex_6_error_distribution.pdf")
