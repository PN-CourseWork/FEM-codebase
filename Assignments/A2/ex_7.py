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

p_values = range(3, 9)  # Extended range for better rate calculation
diagonals = ["nw_se", "sw_ne"]
results = {}

def calc_rate(h, err):
    """Calculate convergence rate from last 4 points."""
    if len(h) < 4:
        return np.polyfit(np.log(h), np.log(err), 1)[0]
    return np.polyfit(np.log(h[-4:]), np.log(err[-4:]), 1)[0]

for diag in diagonals:
    print(f"\n  Diagonal: {diag}")
    print(f"  {'p':<5} {'noelms':<10} {'h':<15} {'E_inf':<15}")

    errors = []
    h_values = []

    for p in p_values:
        noelms = 2**p
        mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=noelms, noelms2=noelms, diagonal=diag)
        h = np.sqrt((L1 / noelms) ** 2 + (L2 / noelms) ** 2)

        u_h = solve_mixed_bc_2d(mesh, q_tilde_2, q_left_2, q_bottom_2, u_exact_2)
        u_ex = u_exact_2(mesh.VX, mesh.VY)

        E_inf = np.max(np.abs(u_h - u_ex))

        errors.append(E_inf)
        h_values.append(h)

        print(f"  {p:<5} {noelms:<10} {h:<15.6f} {E_inf:<15.6e}")

    errors = np.array(errors)
    h_values = np.array(h_values)

    # Compute convergence rate from last 4 points
    alpha = calc_rate(h_values, errors)
    print(f"  Observed rate (last 4 pts): {alpha:.2f}")

    results[diag] = {
        "errors": errors,
        "h_values": h_values,
        "alpha": alpha,
    }

# ============================================================
# Combined convergence plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

styles = {
    "nw_se": {"color": "tab:blue", "marker": "o", "ls": "-"},
    "sw_ne": {"color": "tab:orange", "marker": "s", "ls": "--"},
}

for diag in diagonals:
    r = results[diag]
    s = styles[diag]
    ax.loglog(r["h_values"], r["errors"], 
               marker=s["marker"], linestyle=s["ls"], color=s["color"],
               label=rf"Mixed ({diag}) [rate={r['alpha']:.2f}]")

# Reference O(h^2) line
r0 = results[diagonals[0]]
h_ref = np.array([r0["h_values"][0], r0["h_values"][-1]])
err_ref = r0["errors"][0] * (h_ref / h_ref[0]) ** 2
ax.loglog(h_ref, err_ref, "k:", alpha=0.5, label=r"$O(h^2)$ reference")

ax.set_xlabel("h (max element edge length)")
ax.set_ylabel(r"$L^\infty$ error")

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "case2_convergence_combined.pdf")
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


    fig.tight_layout()
    fig.savefig(output_dir / f"case2_triplot_{diag}.pdf")
    plt.close(fig)
