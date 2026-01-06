import numpy as np
import matplotlib.pyplot as plt
from FEM import (
    uniform_mesh,
    element_diffusion_reaction,
    assemble_1d,
    apply_dirichlet_bc,
    solve_symmetric,
)


def BVP1D(L, c, d, M):
    """
    Solve u'' - u = 0 on [0,L] using linear FEM on a uniform mesh.

    Parameters
    ----------
    L : float
        Domain length
    c : float
        Left boundary condition u(0) = c
    d : float
        Right boundary condition u(L) = d
    M : int
        Number of nodes

    Returns
    -------
    x : ndarray
        Node coordinates
    u : ndarray
        FEM solution at the nodes
    """
    x, h = uniform_mesh(L, M)
    A, b = assemble_1d(M, element_diffusion_reaction, h)
    apply_dirichlet_bc(A, b, 0, c)
    apply_dirichlet_bc(A, b, M - 1, d)
    u = solve_symmetric(A, b)
    return x, u


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
x, u = BVP1D(L, c, d, M=11)
u_exact = np.exp(x)

# Plot comparison
plt.figure(figsize=(8, 5))
plt.plot(x, u_exact, "k-", label="Exact $u(x)=e^x$", linewidth=2)
plt.plot(x, u, "r:", label="FEM (M=11)", linewidth=2)
plt.plot(x_uniform, u_hat_uniform, "b-x", label="Uniform (3 nodes)", linewidth=2)
plt.plot(x_non_uniform, u_hat_non_uniform, "g:^", label="Non-uniform (3 nodes)", linewidth=2)
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.legend()
plt.grid(True)
plt.title("Comparison of FEM solutions")
plt.savefig('figures/ex_2_sol.pdf')

# Convergence test
M_values = [11, 21, 41, 81, 161]
h_values = np.array([L / (M - 1) for M in M_values])
error_values = np.array([np.max(np.abs(BVP1D(L, c, d, M)[1] - np.exp(BVP1D(L, c, d, M)[0]))) for M in M_values])

p_est = np.polyfit(np.log(h_values), np.log(error_values), 1)[0]
print(f"Estimated convergence rate: {p_est:.2f}")

plt.figure(figsize=(7, 5))
plt.loglog(h_values, error_values, "o-", label="Numerical error")
plt.loglog(h_values, error_values[0] * (h_values / h_values[0]) ** 2, "--", label=r"$O(h^2)$")
plt.xlabel("$h$")
plt.ylabel(r"$\max |u_{FEM} - u_{exact}|$")
plt.legend()
plt.grid(True, which="both")
plt.title("Convergence of FEM solution")

plt.savefig('figures/ex_2_conv.pdf')

