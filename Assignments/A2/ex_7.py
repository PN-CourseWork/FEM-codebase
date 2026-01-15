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


def main():
    print("Exercise 2.7: Mixed Boundary Conditions")
    print("=" * 50)

    x0, y0 = -2.5, -4.8
    L1, L2 = 7.6, 5.9

    # ============================================================
    # TEST CASE 1: u(x,y) = 3x + 5y - 7 (linear, should be exact)
    # ============================================================
    print("\nCASE 1: u(x,y) = 3x + 5y - 7")
    print("-" * 50)

    mesh1 = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=4, noelms2=3)

    def u_exact_1(x, y):
        return 3 * x + 5 * y - 7

    def q_tilde_1(x, _y):
        return np.zeros_like(x)

    def q_left_1(x, _y):
        return 3 * np.ones_like(x)

    def q_bottom_1(x, _y):
        return 5 * np.ones_like(x)

    u_h1 = solve_mixed_bc_2d(mesh1, q_tilde_1, q_left_1, q_bottom_1, u_exact_1)

    # Reshape to 2D array format
    u_2d = u_h1.reshape((mesh1.nonodes2, mesh1.nonodes1), order="F")

    u_ex1 = u_exact_1(mesh1.VX, mesh1.VY)
    E1 = np.max(np.abs(u_h1 - u_ex1))
    print(f"\n  Max error E = {E1:.6e}")

    # Triplot for Case 1
    mesh1_fine = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=4, noelms2=3)
    u_h1_fine = solve_mixed_bc_2d(
        mesh1_fine, q_tilde_1, q_left_1, q_bottom_1, u_exact_1
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tri = Triangulation(mesh1_fine.VX, mesh1_fine.VY, mesh1_fine.EToV - 1)
    u_ex1_fine = u_exact_1(mesh1_fine.VX, mesh1_fine.VY)

    c0 = axes[0].tripcolor(tri, u_h1_fine)
    axes[0].set_title(r"FEM Solution $u_h$")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tripcolor(tri, u_ex1_fine)
    axes[1].set_title(r"Exact Solution $u$")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tripcolor(tri, np.abs(u_h1_fine - u_ex1_fine))
    axes[2].set_title(r"Error $|u_h - u|$")
    plt.colorbar(c2, ax=axes[2])

    fig.suptitle(r"$u = 3x + 5y - 7$ with Mixed BC")
    fig.tight_layout()
    fig.savefig(output_dir / "case1_triplot.pdf")
    plt.close(fig)

    # ============================================================
    # TEST CASE 2: u(x,y) = sin(x)sin(y)
    # ============================================================
    print("\n\nCASE 2: u(x,y) = sin(x)sin(y)")
    print("-" * 50)

    mesh2 = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=32, noelms2=32)

    def u_exact_2(x, y):
        return np.sin(x) * np.sin(y)

    def q_tilde_2(x, y):
        return 2 * np.sin(x) * np.sin(y)

    def q_left_2(x, y):
        return np.cos(x) * np.sin(y)

    def q_bottom_2(x, y):
        return np.sin(x) * np.cos(y)

    u_h2 = solve_mixed_bc_2d(mesh2, q_tilde_2, q_left_2, q_bottom_2, u_exact_2)
    u_ex2 = u_exact_2(mesh2.VX, mesh2.VY)
    E2 = np.max(np.abs(u_h2 - u_ex2))

    print(f"\n  Max error E = {E2:.6e}")

    # ============================================================
    # CONVERGENCE ANALYSIS
    # ============================================================
    print("\n\nCONVERGENCE ANALYSIS")
    print("-" * 50)

    p_values = range(5, 8)
    errors = []
    errors_l2 = []
    h_values = []

    for p in p_values:
        noelms = 2**p
        mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=noelms, noelms2=noelms)
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

    print("\n  Convergence rates (L-infinity / L2):")
    for i in range(len(errors) - 1):
        rate_inf = np.log(errors[i] / errors[i + 1]) / np.log(
            h_values[i] / h_values[i + 1]
        )
        rate_l2 = np.log(errors_l2[i] / errors_l2[i + 1]) / np.log(
            h_values[i] / h_values[i + 1]
        )
        print(
            f"    p={list(p_values)[i]} to p={list(p_values)[i + 1]}: {rate_inf:.2f} / {rate_l2:.2f}"
        )

    log_h = np.log(h_values)
    log_E = np.log(errors)
    coeffs = np.polyfit(log_h, log_E, 1)
    alpha = coeffs[0]

    log_E_l2 = np.log(errors_l2)
    coeffs_l2 = np.polyfit(log_h, log_E_l2, 1)
    alpha_l2 = coeffs_l2[0]

    print(f"\n  Least squares fit (L-inf): E ~ h^{alpha:.2f}")
    print(f"  Least squares fit (L2):    E ~ h^{alpha_l2:.2f}")

    # Convergence plot with rates in legend
    fig, ax = plt.subplots()
    ax.loglog(h_values, errors, "o-", label=rf"$L^\infty$ error ($\alpha={alpha:.2f}$)")
    ax.loglog(h_values, errors_l2, "s-", label=rf"$L^2$ error ($\alpha={alpha_l2:.2f}$)")
    ax.loglog(
        h_values,
        errors[0] * (h_values / h_values[0]) ** 2,
        "--",
        color="gray",
        label=r"$O(h^2)$ reference",
    )
    ax.set_xlabel("h (max element edge length)")
    ax.set_ylabel("Error")
    ax.set_title(r"FEM Convergence for $u = \sin(x)\sin(y)$ (Mixed BC)")
    ax.legend()
    fig.savefig(output_dir / "case2_convergence.pdf")
    plt.close(fig)

    # Triplot for Case 2
    mesh2_fine = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=32, noelms2=32)
    u_h2_fine = solve_mixed_bc_2d(
        mesh2_fine, q_tilde_2, q_left_2, q_bottom_2, u_exact_2
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tri = Triangulation(mesh2_fine.VX, mesh2_fine.VY, mesh2_fine.EToV - 1)
    u_ex2_fine = u_exact_2(mesh2_fine.VX, mesh2_fine.VY)

    c0 = axes[0].tripcolor(tri, u_h2_fine)
    axes[0].set_title(r"FEM Solution $u_h$")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].tripcolor(tri, u_ex2_fine)
    axes[1].set_title(r"Exact Solution $u$")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].tripcolor(tri, np.abs(u_h2_fine - u_ex2_fine))
    axes[2].set_title(r"Error $|u_h - u|$")
    plt.colorbar(c2, ax=axes[2])

    fig.suptitle(r"$u = \sin(x)\sin(y)$ with Mixed BC")
    fig.tight_layout()
    fig.savefig(output_dir / "case2_triplot.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
