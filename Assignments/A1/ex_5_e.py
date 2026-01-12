"""
DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment - Exercise 1.5(e)

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from FEM import (
    line_mesh,
    solve_advection_diffusion_1d,
    discrete_l2_error,
    setup_style,
    save_figure,
)

setup_style()


def analytical_solution(x, psi, eps):
    # Numerically stable form derived in ex_5_stability.py:
    # u(x) = (1/psi) * [x(1 - e^{-psi/eps}) + e^{-psi/eps} - e^{psi(x-1)/eps}] / (1 - e^{-psi/eps})

    exp_neg_K = np.exp(-psi / eps)
    exp_x_minus_1_K = np.exp(psi * (x - 1) / eps)

    nom = x * (1 - exp_neg_K) + exp_neg_K - exp_x_minus_1_K
    denom = 1 - exp_neg_K

    return (1 / psi) * nom / denom


psi = 1.0
eps_values = [1.0, 0.01, 0.0001]
f = 1.0
n_elem_values = [100, 200, 300, 400]

eps_grid, n_grid = np.meshgrid(eps_values, n_elem_values)
df_conv = pd.DataFrame({"eps": eps_grid.ravel(), "n_elem": n_grid.ravel()})
df_conv["h"] = 1.0 / df_conv["n_elem"]


def compute_l2_error(r):
    mesh = line_mesh(1.0, int(r["n_elem"]))
    u_fem, _, _ = solve_advection_diffusion_1d(
        mesh, r["eps"], psi, lambda x: f * np.ones_like(x)
    )
    return discrete_l2_error(
        mesh, u_fem, lambda x: analytical_solution(x, psi, r["eps"])
    )


df_conv["error"] = df_conv.apply(compute_l2_error, axis=1)

fig, axes = plt.subplots(1, len(eps_values), figsize=(18, 5))
if len(eps_values) == 1:
    axes = [axes]

for ax, eps in zip(axes, eps_values):
    grp = df_conv[df_conv["eps"] == eps]
    ax.loglog(grp["h"], grp["error"], "o-", label="FEM error")
    ref = grp["error"].iloc[0] * (grp["h"] / grp["h"].iloc[0]) ** 2
    ax.loglog(grp["h"], ref, "k--", label=r"$O(h^2)$")
    rate = np.polyfit(np.log(grp["h"]), np.log(grp["error"]), 1)[0]

    if eps == 1.0:
        assert 1.9 < rate < 2.1, (
            f"Convergence rate for eps=1.0 is {rate}, expected ~2.0"
        )

    ax.set_title(rf"$\varepsilon = {eps}$, rate $\approx {rate:.2f}$")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\|u_{\mathrm{FEM}} - u_{\mathrm{exact}}\|_2$")
    ax.legend()
plt.tight_layout()
save_figure(fig, "figures/A1/ex_5/ex_5_e_convergence.pdf")
plt.close(fig)

x_fine = np.linspace(0, 1, 1000)
fig, axes = plt.subplots(1, len(eps_values), figsize=(18, 5))
if len(eps_values) == 1:
    axes = [axes]

for ax, eps in zip(axes, eps_values):
    ax.plot(x_fine, analytical_solution(x_fine, psi, eps), label="Analytical")
    for n in n_elem_values:
        mesh_sol = line_mesh(1.0, n)
        u_sol, _, _ = solve_advection_diffusion_1d(
            mesh_sol, eps, psi, lambda x: f * np.ones_like(x)
        )
        ax.scatter(mesh_sol.VX, u_sol, s=6, label=f"FEM n={n}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title(rf"$\varepsilon = {eps}$")
    ax.legend()
plt.tight_layout()
save_figure(fig, "figures/A1/ex_5/ex_5_e_solutions.pdf")
plt.close(fig)
