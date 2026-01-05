"""
Exercise 1.2 - 1D Finite Element Method for Boundary Value Problem
02623 The Finite Element Method for PDEs

Solves: u'' - u = 0, 0 <= x <= L
With:   u(0) = c, u(L) = d

Authors: Philip Korsager Nickel (s214960), Aske Funch Schrøder Nielsen (s224409)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from solvers.FEM.bvp1d import BVP1D_solver

plt.style.use("src/fem.mplstyle")

# Figures saved to figures/A1/ (run from repo root)
FIGURE_DIR = Path("figures/A1")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def exact_solution(x: np.ndarray, c: float, d: float, L: float) -> np.ndarray:
    """Exact solution for u'' - u = 0 with u(0)=c, u(L)=d."""
    eL = np.exp(L)
    emL = np.exp(-L)
    det = emL - eL
    c1 = (c * emL - d) / det
    c2 = (d - c * eL) / det
    return c1 * np.exp(x) + c2 * np.exp(-x)


def compute_max_error(u_fem: np.ndarray, x: np.ndarray, c: float, d: float, L: float) -> float:
    """Compute max error, sampling within elements."""
    max_err = 0.0
    for i in range(len(x) - 1):
        xi = np.linspace(x[i], x[i + 1], 50)
        t = (xi - x[i]) / (x[i + 1] - x[i])
        ui = u_fem[i] * (1 - t) + u_fem[i + 1] * t
        ue = exact_solution(xi, c, d, L)
        max_err = max(max_err, np.max(np.abs(ui - ue)))
    return max_err


# Problem parameters
L = 2.0
c = 1.0
d = np.exp(2)

print("=" * 60)
print("Exercise 1.2 - 1D FEM Solver")
print("=" * 60)

# ========================================
# Part a) Non-uniform mesh
# ========================================
print("\n--- Part a) Non-uniform mesh ---")
x_test = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.9, 1.4, 1.5, 1.8, 1.9, 2.0])
u = BVP1D_solver(x_test, left_bc=c, right_bc=d)
u_exact = exact_solution(x_test, c, d, L)

print(f"Nodes: {len(x_test)}")
print(f"Max nodal error: {np.max(np.abs(u - u_exact)):.6e}")

# ========================================
# Part b) Uniform mesh
# ========================================
print("\n--- Part b) Uniform mesh (M=11) ---")
M = 11
x = np.linspace(0, L, M)
u = BVP1D_solver(x, left_bc=c, right_bc=d)
u_exact = exact_solution(x, c, d, L)

print(f"Max nodal error: {np.max(np.abs(u - u_exact)):.6e}")

# ========================================
# Part c) Validation plots
# ========================================
print("\n--- Part c) Validation ---")

x_fine = np.linspace(0, L, 500)
u_exact_fine = exact_solution(x_fine, c, d, L)
u_fem_fine = np.interp(x_fine, x, u)
u_interp_fine = np.interp(x_fine, x, exact_solution(x, c, d, L))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_fine, u_exact_fine, "k-", lw=2, label=r"Exact $u(x)$")
ax.plot(x_fine, u_fem_fine, "b--", lw=1.5, label=r"FEM $\hat{u}(x)$")
ax.plot(x_fine, u_interp_fine, "r:", lw=1.5, label=r"Interpolant $u_I(x)$")
ax.plot(x, u, "bo", ms=6, label="FEM nodes")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$u$")
ax.set_title(r"Exercise 1.2c: Validation ($M=11$, uniform mesh)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "exercise_1_2c.pdf")
print(f"Saved: {FIGURE_DIR / 'exercise_1_2c.pdf'}")

# ========================================
# Part d) Convergence study
# ========================================
print("\n--- Part d) Convergence rate ---")

M_vals = [5, 9, 17, 33, 65, 129, 257]
h_vals = []
errors = []

for M in M_vals:
    x = np.linspace(0, L, M)
    u = BVP1D_solver(x, left_bc=c, right_bc=d)
    h = L / (M - 1)
    err = compute_max_error(u, x, c, d, L)
    h_vals.append(h)
    errors.append(err)
    print(f"M={M:4d}, h={h:.5f}, error={err:.6e}")

h_vals = np.array(h_vals)
errors = np.array(errors)

p = np.polyfit(np.log(h_vals), np.log(errors), 1)[0]
print(f"\nConvergence rate: p = {p:.2f} (expected: 2)")

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h_vals, errors, "o-", lw=1.5, ms=7, label=rf"FEM ($p={p:.2f}$)")
ax.loglog(h_vals, errors[0] * (h_vals / h_vals[0]) ** 2, "k--", label=r"$O(h^2)$")
ax.set_xlabel(r"$h$")
ax.set_ylabel(r"Max error")
ax.set_title(r"Exercise 1.2d: Convergence")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "exercise_1_2d.pdf")
print(f"Saved: {FIGURE_DIR / 'exercise_1_2d.pdf'}")

plt.show()
