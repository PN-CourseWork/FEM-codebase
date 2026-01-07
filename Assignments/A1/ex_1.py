import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from FEM.plot_style import setup_style, save_figure

setup_style()

def u_exact(x):
    return np.exp(x)

# -------------------------
# UNIFORM CASE
# -------------------------
x_nodes_uniform = np.array([0.0, 1.0, 2.0])

u_hat_uniform = np.array([
    1.0,
    (5/16)*(1 + np.exp(2)),
    np.exp(2)
])

def u_fem_uniform(x):
    if x <= 1.0:
        N0 = (1 - x)
        N1 = x
        return u_hat_uniform[0]*N0 + u_hat_uniform[1]*N1
    else:
        N1 = (2 - x)
        N2 = (x - 1)
        return u_hat_uniform[1]*N1 + u_hat_uniform[2]*N2

uI_nodes_uniform = u_exact(x_nodes_uniform)

def u_interpolant_uniform(x):
    if x <= 1.0:
        N0 = (1 - x)
        N1 = x
        return uI_nodes_uniform[0]*N0 + uI_nodes_uniform[1]*N1
    else:
        N1 = (2 - x)
        N2 = (x - 1)
        return uI_nodes_uniform[1]*N1 + uI_nodes_uniform[2]*N2

# -------------------------
# NON-UNIFORM CASE
# -------------------------
x_nodes_nonuniform = np.array([0.0, 4/3, 2.0])

u_hat_nonuniform = np.array([
    1.0,
    19/101 + (50/101)*np.exp(2),
    np.exp(2)
])

def u_fem_nonuniform(x):
    if x <= x_nodes_nonuniform[1]:
        h1 = x_nodes_nonuniform[1] - x_nodes_nonuniform[0]
        N0 = (x_nodes_nonuniform[1] - x) / h1
        N1 = (x - x_nodes_nonuniform[0]) / h1
        return u_hat_nonuniform[0]*N0 + u_hat_nonuniform[1]*N1
    else:
        h2 = x_nodes_nonuniform[2] - x_nodes_nonuniform[1]
        N1 = (x_nodes_nonuniform[2] - x) / h2
        N2 = (x - x_nodes_nonuniform[1]) / h2
        return u_hat_nonuniform[1]*N1 + u_hat_nonuniform[2]*N2

uI_nodes_nonuniform = u_exact(x_nodes_nonuniform)

def u_interpolant_nonuniform(x):
    if x <= x_nodes_nonuniform[1]:
        h1 = x_nodes_nonuniform[1] - x_nodes_nonuniform[0]
        N0 = (x_nodes_nonuniform[1] - x) / h1
        N1 = (x - x_nodes_nonuniform[0]) / h1
        return uI_nodes_nonuniform[0]*N0 + uI_nodes_nonuniform[1]*N1
    else:
        h2 = x_nodes_nonuniform[2] - x_nodes_nonuniform[1]
        N1 = (x_nodes_nonuniform[2] - x) / h2
        N2 = (x - x_nodes_nonuniform[1]) / h2
        return uI_nodes_nonuniform[1]*N1 + uI_nodes_nonuniform[2]*N2

# -------------------------
# Plot both cases side by side
# -------------------------
x_plot = np.linspace(0, 2, 500)
u_exact_vals = u_exact(x_plot)
colors = sns.color_palette("deep")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Uniform mesh plot
u_fem_vals_uniform = np.array([u_fem_uniform(x) for x in x_plot])
u_interp_vals_uniform = np.array([u_interpolant_uniform(x) for x in x_plot])

ax1.plot(x_plot, u_exact_vals, '-', color=colors[0], label=r'Exact $u(x)=e^x$')
ax1.plot(x_plot, u_fem_vals_uniform, '--', color=colors[1], label=r'FEM $\hat{u}(x)$')
ax1.plot(x_plot, u_interp_vals_uniform, '-.', color=colors[2], label=r'Interpolant $u_I(x)$')
ax1.plot(x_nodes_uniform, u_hat_uniform, 'o', color=colors[1], markersize=8)
ax1.plot(x_nodes_uniform, uI_nodes_uniform, 's', color=colors[2], markersize=8)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$u(x)$')
ax1.legend()
ax1.set_title(r'(a) Uniform mesh')

# Non-uniform mesh plot
u_fem_vals_nonuniform = np.array([u_fem_nonuniform(x) for x in x_plot])
u_interp_vals_nonuniform = np.array([u_interpolant_nonuniform(x) for x in x_plot])

ax2.plot(x_plot, u_exact_vals, '-', color=colors[0], label=r'Exact $u(x)=e^x$')
ax2.plot(x_plot, u_fem_vals_nonuniform, '--', color=colors[1], label=r'FEM $\hat{u}(x)$')
ax2.plot(x_plot, u_interp_vals_nonuniform, '-.', color=colors[2], label=r'Interpolant $u_I(x)$')
ax2.plot(x_nodes_nonuniform, u_hat_nonuniform, 'o', color=colors[1], markersize=8)
ax2.plot(x_nodes_nonuniform, uI_nodes_nonuniform, 's', color=colors[2], markersize=8)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$u(x)$')
ax2.legend()
ax2.set_title(r'(b) Non-uniform mesh')

plt.tight_layout()
save_figure(fig, "ex_1_sol.pdf")
