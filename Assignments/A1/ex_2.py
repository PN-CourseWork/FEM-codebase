import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import spsolve

from FEM import line_mesh, diffusion, mass, assemble_matrix_1d, apply_dirichlet


def BVP1D(L, c, d, n_elem):
    """
    Solve u'' - u = 0 on [0,L] using linear FEM.

    Parameters
    ----------
    L : float
        Domain length
    c : float
        Left BC u(0) = c
    d : float
        Right BC u(L) = d
    n_elem : int
        Number of elements

    Returns
    -------
    x : ndarray
        Node coordinates
    u : ndarray
        FEM solution
    """
    mesh = line_mesh(L, n_elem)
    h = np.diff(mesh.VX)

    # Element matrices: -u'' + u (diffusion + mass)
    Ke_all = np.array([diffusion(hi) + mass(hi) for hi in h])

    A = assemble_matrix_1d(Ke_all, mesh.n_nodes)
    b = np.zeros(mesh.n_nodes)

    apply_dirichlet(A, b, 0, c)
    apply_dirichlet(A, b, mesh.n_nodes - 1, d)

    u = spsolve(A, b)
    return mesh.VX, u


# Problem data
L = 2.0
c = 1.0
d = np.exp(2)

# Reference solutions from Exercise 1.1
x_uniform = np.array([0.0, 1.0, 2.0])
u_hat_uniform = np.array([1.0, (5 / 16) * (1 + np.exp(2)), np.exp(2)])

x_non_uniform = np.array([0.0, 4 / 3, 2.0])
u_hat_non_uniform = np.array([1.0, 19 / 101 + (50 / 101) * np.exp(2), np.exp(2)])

# Solve with refined mesh
x, u = BVP1D(L, c, d, n_elem=10)
u_exact = np.exp(x)

colors = sns.color_palette("deep")

# Plot comparison
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x, u_exact, "-", color=colors[0], label=r"Exact $u(x)=e^x$")
ax.plot(x, u, ":", color=colors[1], label=r"FEM ($M=11$)")
ax.plot(x_uniform, u_hat_uniform, "-x", color=colors[2], label=r"Uniform (3 nodes)")
ax.plot(x_non_uniform, u_hat_non_uniform, ":^", color=colors[3], label=r"Non-uniform (3 nodes)")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u(x)$")
ax.legend()
ax.set_title(r"Comparison of FEM solutions")
fig.savefig("figures/A1/ex_2_sol.pdf")

# Convergence test
n_elem_values = [10, 20, 40, 80, 160]
h_values = np.array([L / n for n in n_elem_values])
error_values = np.array(
    [np.max(np.abs(BVP1D(L, c, d, n)[1] - np.exp(BVP1D(L, c, d, n)[0]))) for n in n_elem_values]
)

p_est = np.polyfit(np.log(h_values), np.log(error_values), 1)[0]
print(f"Estimated convergence rate: {p_est:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(h_values, error_values, "o-", color=colors[0], label="Numerical error")
ax.loglog(h_values, error_values[0] * (h_values / h_values[0]) ** 2, "--", color=colors[1], label=r"$O(h^2)$")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"$\max |u_{\mathrm{FEM}} - u_{\mathrm{exact}}|$")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
ax.set_title(r"Convergence of FEM solution")
fig.savefig("figures/A1/ex_2_conv.pdf")
