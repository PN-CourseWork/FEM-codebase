import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from FEM import line_mesh, diffusion, advection, load, assemble_matrix_1d, assemble_vector, apply_dirichlet


def analytical_solution(x, psi, eps):
    nom = 1 + (np.exp(psi / eps) - 1) * x - np.exp(x * psi / eps)
    denom = np.exp(psi / eps) - 1
    return 1 / psi * nom / denom


def advection_diffusion_FEM(eps, psi, f, n_elem):
    """
    Solve -eps*u'' + psi*u' = f on [0,1] with u(0) = u(1) = 0.
    """
    mesh = line_mesh(1.0, n_elem)
    h = np.diff(mesh.VX)

    Ke_all = np.array([diffusion(hi, eps) + advection(hi, psi) for hi in h])
    fe_all = np.array([load(hi, f) for hi in h])

    A = assemble_matrix_1d(Ke_all, mesh.n_nodes)
    b = assemble_vector(mesh.EToV, fe_all, mesh.n_nodes)

    apply_dirichlet(A, b, 0, 0.0)
    apply_dirichlet(A, b, mesh.n_nodes - 1, 0.0)

    u = spsolve(A, b)
    return mesh.VX, u


# Parameters
psi = 1.0
eps_values = [1.0, 0.1, 0.01]
f = 1.0
n_elem_values = [20, 40, 60, 80, 100]

# Build convergence DataFrame directly
eps_grid, n_grid = np.meshgrid(eps_values, n_elem_values)
df_conv = pd.DataFrame({"eps": eps_grid.ravel(), "n_elem": n_grid.ravel()})
df_conv["h"] = 1.0 / df_conv["n_elem"]
df_conv["error"] = df_conv.apply(
    lambda r: np.sqrt(r["h"]) * np.linalg.norm(
        advection_diffusion_FEM(r["eps"], psi, f, int(r["n_elem"]))[1] -
        analytical_solution(np.linspace(0, 1, int(r["n_elem"]) + 1), psi, r["eps"])
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
g = sns.relplot(
    data=df_conv_plot, x="h", y="error", hue="type", style="type",
    col="eps", col_order=eps_values, kind="line", dashes=True, markers=True,
    facet_kws={"sharex": False, "sharey": False}, height=4, aspect=1.0
)
g.set(xscale="log", yscale="log")
g.set_axis_labels(r"$h$", r"$\|u_{\mathrm{FEM}} - u_{\mathrm{exact}}\|_2$")
for ax, eps in zip(g.axes.flat, eps_values):
    grp = df_conv[df_conv["eps"] == eps]
    rate = np.polyfit(np.log(grp["h"]), np.log(grp["error"]), 1)[0]
    ax.set_title(rf"$\varepsilon = {eps}$, rate $\approx {rate:.2f}$")
g.figure.savefig("figures/A1/ex_5_e_convergence.pdf")
plt.close()

# Build solution DataFrame
x_fine = np.linspace(0, 1, 1000)
df_sol = pd.concat([
    pd.DataFrame({"eps": eps, "x": x_fine, "u": analytical_solution(x_fine, psi, eps), "type": "Analytical"})
    for eps in eps_values
] + [
    pd.DataFrame({"eps": eps, "x": np.linspace(0, 1, n + 1), "u": advection_diffusion_FEM(eps, psi, f, n)[1], "type": f"FEM $n={n}$"})
    for eps in eps_values for n in n_elem_values
])

# Solution plot
g = sns.relplot(
    data=df_sol, x="x", y="u", hue="type", style="type",
    col="eps", col_order=eps_values, kind="line", dashes=True,
    facet_kws={"sharey": False}, height=4, aspect=1.0
)
g.set_axis_labels(r"$x$", r"$u(x)$")
for ax, eps in zip(g.axes.flat, eps_values):
    ax.set_title(rf"$\varepsilon = {eps}$")
g.figure.savefig("figures/A1/ex_5_e_solutions.pdf")
plt.close()
