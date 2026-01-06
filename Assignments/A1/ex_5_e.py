import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from FEM import (
    uniform_mesh,
    element_advection_diffusion,
    element_load,
    assemble_1d,
    apply_dirichlet_bc,
    solve_general,
)

sns.set_theme(style="darkgrid")


def analytical_solution(x, psi, eps):
    nom = 1 + (np.exp(psi / eps) - 1) * x - np.exp(x * psi / eps)
    denom = np.exp(psi / eps) - 1
    return 1 / psi * nom / denom


def advection_diffusion_FEM(eps, psi, f, M):
    """
    Solve -eps*u'' + psi*u' = f on [0,1] with u(0) = u(1) = 0.
    """
    x, h = uniform_mesh(1.0, M)
    A, b = assemble_1d(
        M,
        lambda h: element_advection_diffusion(h, eps, psi),
        h,
        lambda h: f * element_load(h),
    )
    apply_dirichlet_bc(A, b, 0, 0.0)
    apply_dirichlet_bc(A, b, M - 1, 0.0)
    u = solve_general(A, b)
    return x, u


# Parameters
psi = 1.0
eps_values = [1.0, 0.1, 0.01]
f = 1.0
M_values = [21, 41, 61, 81, 101]

# Build convergence DataFrame directly
eps_grid, M_grid = np.meshgrid(eps_values, M_values)
df_conv = pd.DataFrame({"eps": eps_grid.ravel(), "M": M_grid.ravel()})
df_conv["h"] = 1.0 / (df_conv["M"] - 1)
df_conv["error"] = df_conv.apply(
    lambda r: np.sqrt(r["h"]) * np.linalg.norm(
        advection_diffusion_FEM(r["eps"], psi, f, int(r["M"]))[1] -
        analytical_solution(np.linspace(0, 1, int(r["M"])), psi, r["eps"])
    ), axis=1
)

# Add reference lines for convergence plot
ref_lines = []
for eps in eps_values:
    grp = df_conv[df_conv["eps"] == eps]
    for h in grp["h"]:
        ref_lines.append({"eps": eps, "h": h, "error": grp["error"].iloc[0] * (h / grp["h"].iloc[0]) ** 2, "type": r"$O(h^2)$"})
df_conv["type"] = "FEM error"
df_conv_plot = pd.concat([df_conv[["eps", "h", "error", "type"]], pd.DataFrame(ref_lines)])

# Convergence plot
g = sns.relplot(data=df_conv_plot, x="h", y="error", hue="type", style="type", col="eps", col_order=eps_values, kind="line", dashes=True, markers=True, facet_kws={"sharex": False, "sharey": False})
g.set(xscale="log", yscale="log")
g.set_axis_labels("$h$", r"$\|u_{FEM} - u_{exact}\|_2$")
for ax, eps in zip(g.axes.flat, eps_values):
    grp = df_conv[df_conv["eps"] == eps]
    rate = np.polyfit(np.log(grp["h"]), np.log(grp["error"]), 1)[0]
    ax.set_title(f"$\\varepsilon = {eps}$, rate $\\approx {rate:.2f}$")
g.savefig("figures/ex_5_e_convergence.pdf")
plt.close()

# Build solution DataFrame
x_fine = np.linspace(0, 1, 1000)
df_sol = pd.concat([
    pd.DataFrame({"eps": eps, "x": x_fine, "u": analytical_solution(x_fine, psi, eps), "type": "Analytical"})
    for eps in eps_values
] + [
    pd.DataFrame({"eps": eps, "x": np.linspace(0, 1, M), "u": advection_diffusion_FEM(eps, psi, f, M)[1], "type": f"FEM M={M}"})
    for eps in eps_values for M in M_values
])

# Solution plot
g = sns.relplot(data=df_sol, x="x", y="u", hue="type", style="type", col="eps", col_order=eps_values, kind="line", dashes=True, facet_kws={"sharey": False})
g.set_axis_labels("$x$", "$u(x)$")
for ax, eps in zip(g.axes.flat, eps_values):
    ax.set_title(f"$\\varepsilon = {eps}$")
g.savefig("figures/ex_5_e_solutions.pdf")
plt.close()
