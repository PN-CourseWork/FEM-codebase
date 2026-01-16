"""
Exercise 2.7: Mixed Boundary Conditions

Demonstrates solving with Neumann BCs on left/bottom and Dirichlet on right/top.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

from FEM.datastructures import Mesh2d
from FEM.solvers import solve_mixed_bc_2d

# Use custom style
plt.style.use("src/FEM/fem.mplstyle")

# Output directory
output_dir = Path("figures/A2/ex_7")
output_dir.mkdir(parents=True, exist_ok=True)

print("Exercise 2.7: Mixed Boundary Conditions")
print("=" * 50)

x0, y0 = -2.5, -4.8
L1, L2 = 7.6, 5.9

# ============================================================
# TEST CASE 1: u(x,y) = 3x + 5y - 7 (linear, should be exact)
# ============================================================
print("\nCASE 1: u(x,y) = 3x + 5y - 7")
print("-" * 50)


def u_exact_1(x, y):
    return 3 * x + 5 * y - 7


def q_tilde_1(x, _y):
    return np.zeros_like(x)


def q_left_1(x, _y):
    return 3 * np.ones_like(x)


def q_bottom_1(x, _y):
    return 5 * np.ones_like(x)


# Triplots for Case 1 - both diagonal types
for diag in ["nw_se", "sw_ne"]:
    mesh1 = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=4, noelms2=3, diagonal=diag)
    u_h1 = solve_mixed_bc_2d(mesh1, q_tilde_1, q_left_1, q_bottom_1, u_exact_1)
    u_ex1 = u_exact_1(mesh1.VX, mesh1.VY)
    E1 = np.max(np.abs(u_h1 - u_ex1))

    print(f"\n  Diagonal: {diag}, Max error E = {E1:.6e}")

    # Print solution in 2-D array format (as specified in exercise)
    if diag == "sw_ne":
        u_2d = u_h1.reshape((mesh1.noelms2 + 1, mesh1.noelms1 + 1), order='F')
        print("  Solution u_h (2-D format):")
        print(u_2d)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tri = Triangulation(mesh1.VX, mesh1.VY, mesh1.EToV)

    c0 = axes[0].tripcolor(tri, u_h1)
    axes[0].set_title(r"FEM Solution $u_h$")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tripcolor(tri, u_ex1)
    axes[1].set_title(r"Exact Solution $u$")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tripcolor(tri, np.abs(u_h1 - u_ex1))
    axes[2].set_title(rf"Error (max={E1:.2e})")
    plt.colorbar(c2, ax=axes[2])

    fig.suptitle(rf"$u = 3x + 5y - 7$ with Mixed BC (diagonal: {diag})")
    fig.tight_layout()
    fig.savefig(output_dir / f"case1_triplot_{diag}.pdf")
    plt.close(fig)

# ============================================================
# TEST CASE 2: u(x,y) = sin(x)sin(y) - Compare both diagonal types
# ============================================================
print("\n\nCASE 2: u(x,y) = sin(x)sin(y)")
print("-" * 50)


def u_exact_2(x, y):
    return np.sin(x) * np.sin(y)


def q_tilde_2(x, y):
    return 2 * np.sin(x) * np.sin(y)


def q_left_2(x, y):
    return np.cos(x) * np.sin(y)


def q_bottom_2(x, y):
    return np.sin(x) * np.cos(y)


# ============================================================
# CONVERGENCE ANALYSIS - Both diagonal types
# ============================================================
print("\n\nCONVERGENCE ANALYSIS")
print("-" * 50)

p_values = range(5, 9)
diagonals = ["nw_se", "sw_ne"]
results = {}

for diag in diagonals:
    print(f"\n  Diagonal: {diag}")
    print(f"  {'p':<5} {'noelms':<10} {'h':<15} {'E_inf':<15} {'E_L2':<15}")

    errors = []
    errors_l2 = []
    h_values = []

    for p in p_values:
        noelms = 2**p
        mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=noelms, noelms2=noelms, diagonal=diag)
        h = np.sqrt((L1 / noelms) ** 2 + (L2 / noelms) ** 2)

        u_h = solve_mixed_bc_2d(mesh, q_tilde_2, q_left_2, q_bottom_2, u_exact_2)
        u_ex = u_exact_2(mesh.VX, mesh.VY)

        E_inf = np.max(np.abs(u_h - u_ex))
        E_l2 = np.sqrt(np.mean((u_h - u_ex) ** 2))

        errors.append(E_inf)
        errors_l2.append(E_l2)
        h_values.append(h)

        print(f"  {p:<5} {noelms:<10} {h:<15.6f} {E_inf:<15.6e} {E_l2:<15.6e}")

    errors = np.array(errors)
    errors_l2 = np.array(errors_l2)
    h_values = np.array(h_values)

    # Compute convergence rate
    log_h = np.log(h_values)
    log_E = np.log(errors)
    alpha = np.polyfit(log_h, log_E, 1)[0]
    alpha_l2 = np.polyfit(log_h, np.log(errors_l2), 1)[0]

    print(f"  Least squares fit: E_inf ~ h^{alpha:.2f}, E_L2 ~ h^{alpha_l2:.2f}")

    results[diag] = {
        "errors": errors,
        "errors_l2": errors_l2,
        "h_values": h_values,
        "alpha": alpha,
        "alpha_l2": alpha_l2,
    }

# ============================================================
# Side-by-side convergence plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, diag in zip(axes, diagonals):
    r = results[diag]
    ax.loglog(r["h_values"], r["errors"], "o-", label=rf"$L^\infty$ ($\alpha={r['alpha']:.2f}$)")
    ax.loglog(r["h_values"], r["errors_l2"], "s-", label=rf"$L^2$ ($\alpha={r['alpha_l2']:.2f}$)")
    ax.loglog(
        r["h_values"],
        r["errors"][0] * (r["h_values"] / r["h_values"][0]) ** 2,
        "--",
        color="gray",
        label=r"$O(h^2)$ reference",
    )
    ax.set_xlabel("h (max element edge length)")
    ax.set_ylabel("Error")
    ax.set_title(f"Diagonal: {diag}")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle(r"FEM Convergence for $u = \sin(x)\sin(y)$ (Mixed BC)")
fig.tight_layout()
fig.savefig(output_dir / "case2_convergence.pdf")
plt.close(fig)

# ============================================================
# Triplots for both diagonal types
# ============================================================
for diag in diagonals:
    mesh2 = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=32, noelms2=32, diagonal=diag)
    u_h2 = solve_mixed_bc_2d(mesh2, q_tilde_2, q_left_2, q_bottom_2, u_exact_2)
    u_ex2 = u_exact_2(mesh2.VX, mesh2.VY)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tri = Triangulation(mesh2.VX, mesh2.VY, mesh2.EToV)

    c0 = axes[0].tripcolor(tri, u_h2)
    axes[0].set_title(r"FEM Solution $u_h$")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tripcolor(tri, u_ex2)
    axes[1].set_title(r"Exact Solution $u$")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tripcolor(tri, np.abs(u_h2 - u_ex2))
    axes[2].set_title(rf"Error (max={np.max(np.abs(u_h2 - u_ex2)):.2e})")
    plt.colorbar(c2, ax=axes[2])

    fig.suptitle(rf"$u = \sin(x)\sin(y)$ with Mixed BC (diagonal: {diag})")
    fig.tight_layout()
    fig.savefig(output_dir / f"case2_triplot_{diag}.pdf")
    plt.close(fig)
