"""
DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment - Exercise 1.2

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
"""

import matplotlib.pyplot as plt
import numpy as np

from FEM import solve_bvp_1d, linf_error, setup_style, save_figure

setup_style()

L = 2.0
c = 1.0
d = np.exp(2)

x_uniform = np.array([0.0, 1.0, 2.0])
u_hat_uniform = np.array([1.0, (5 / 16) * (1 + np.exp(2)), np.exp(2)])

x_non_uniform = np.array([0.0, 4 / 3, 2.0])
u_hat_non_uniform = np.array([1.0, 19 / 105 + (10 / 21) * np.exp(2), np.exp(2)])

mesh, u, _, _ = solve_bvp_1d(L, c, d, n_elem=10)
x = mesh.VX
u_exact = np.exp(x)

fig, ax = plt.subplots()

ax.plot(x, u_exact, "-", label=r"Exact $u(x)=e^x$")
ax.plot(x, u, ":o", label=r"FEM ($M=11$)")
ax.plot(x_uniform, u_hat_uniform, "-x", label=r"Uniform (3 nodes)")
ax.plot(x_non_uniform, u_hat_non_uniform, ":^", label=r"Non-uniform (3 nodes)")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)$")
ax.legend()
ax.set_title(r"Comparison of FEM solutions")
save_figure(fig, "figures/A1/ex_2/ex_2_sol.pdf")

n_elem_values = [40, 80, 160, 200]
h_values = np.array([L / n for n in n_elem_values])


def compute_error(n):
    mesh_n, u_n, _, _ = solve_bvp_1d(L, c, d, n)
    return linf_error(mesh_n, u_n, lambda x: np.exp(x))


error_values = np.array([compute_error(n) for n in n_elem_values])

p_est = np.polyfit(np.log(h_values), np.log(error_values), 1)[0]
print(f"Estimated convergence rate: {p_est:.2f}")

assert 1.95 < p_est < 2.05, f"Convergence rate {p_est} not close to 2.0"

fig2, ax2 = plt.subplots()
ax2.loglog(
    h_values, error_values, "o-", label=f"Observed order: $p \\approx {p_est:.2f}$"
)
ax2.loglog(
    h_values,
    error_values[0] * (h_values / h_values[0]) ** 2,
    "--",
    label=r"$O(h^2)$",
)
ax2.set_xlabel(r"$h$")
ax2.set_ylabel(r"$\max |u_{\mathrm{FEM}} - u_{\mathrm{exact}}|$")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)
ax2.set_title(r"Convergence of FEM solution")
save_figure(fig2, "figures/A1/ex_2/ex_2_conv.pdf")
