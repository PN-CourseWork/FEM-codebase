"""
DTU Course 02623 - The Finite Element Method for Partial Differential Equations
Week 1 Assignment - Exercise 1.1

Group: 16
Authors: Philip Korsager Nickel, Aske Funch Schrøder Nielsen
Student ID(s): s214960, s224409
"""

import numpy as np
import matplotlib.pyplot as plt

from FEM.plot_style import setup_style, save_figure

setup_style()


def u_exact(x):
    return np.exp(x)


x_nodes_uniform = np.array([0.0, 1.0, 2.0])

u_hat_uniform = np.array([1.0, (5 / 16) * (1 + np.exp(2)), np.exp(2)])


def _piecewise_linear(x, nodes, values):
    x_arr = np.asarray(x)
    out = np.empty_like(x_arr, dtype=float)
    left = x_arr <= nodes[1]

    h1 = nodes[1] - nodes[0]
    t_left = (nodes[1] - x_arr[left]) / h1
    out[left] = values[0] * t_left + values[1] * (1 - t_left)

    h2 = nodes[2] - nodes[1]
    t_right = (nodes[2] - x_arr[~left]) / h2
    out[~left] = values[1] * t_right + values[2] * (1 - t_right)

    return out


uI_nodes_uniform = u_exact(x_nodes_uniform)

x_nodes_nonuniform = np.array([0.0, 4 / 3, 2.0])

u_hat_nonuniform = np.array([1.0, 19 / 105 + (10 / 21) * np.exp(2), np.exp(2)])

uI_nodes_nonuniform = u_exact(x_nodes_nonuniform)

# Sanity checks
expected_uniform_mid = (5 / 16) * (1 + np.exp(2))


expected_nonuniform_mid = 19 / 105 + (10 / 21) * np.exp(2)

x_plot = np.linspace(0, 2, 500)
u_exact_vals = u_exact(x_plot)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

u_fem_vals_uniform = _piecewise_linear(x_plot, x_nodes_uniform, u_hat_uniform)
u_interp_vals_uniform = _piecewise_linear(x_plot, x_nodes_uniform, uI_nodes_uniform)

ax1.plot(x_plot, u_exact_vals, "-", label=r"Exact $u(x)=e^x$")
(line_fem,) = ax1.plot(x_plot, u_fem_vals_uniform, "--", label=r"FEM $\hat{u}(x)$")
(line_interp,) = ax1.plot(
    x_plot, u_interp_vals_uniform, "-.", label=r"Interpolant $u_I(x)$"
)
ax1.plot(x_nodes_uniform, u_hat_uniform, "o", color=line_fem.get_color())
ax1.plot(x_nodes_uniform, uI_nodes_uniform, "s", color=line_interp.get_color())
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$u(x)$")
ax1.legend()
ax1.set_title(r"(a) Uniform mesh")

u_fem_vals_nonuniform = _piecewise_linear(x_plot, x_nodes_nonuniform, u_hat_nonuniform)
u_interp_vals_nonuniform = _piecewise_linear(
    x_plot, x_nodes_nonuniform, uI_nodes_nonuniform
)

ax2.plot(x_plot, u_exact_vals, "-", label=r"Exact $u(x)=e^x$")
(line_fem,) = ax2.plot(x_plot, u_fem_vals_nonuniform, "--", label=r"FEM $\hat{u}(x)$")
(line_interp,) = ax2.plot(
    x_plot, u_interp_vals_nonuniform, "-.", label=r"Interpolant $u_I(x)$"
)
ax2.plot(x_nodes_nonuniform, u_hat_nonuniform, "o", color=line_fem.get_color())
ax2.plot(x_nodes_nonuniform, uI_nodes_nonuniform, "s", color=line_interp.get_color())
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$u(x)$")
ax2.legend()
ax2.set_title(r"(b) Non-uniform mesh")

save_figure(fig, "figures/A1/ex_1/ex_1_sol.pdf")
