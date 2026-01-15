"""
Exercise 2.5: Dirichlet Problem and Convergence Analysis

Demonstrates solving Poisson equation with Dirichlet BCs and O(h^2) convergence.
Run pytest tests/test_a2.py::TestEx5DirichletConvergence for validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from pathlib import Path

from FEM.datastructures import Mesh2d
from FEM.assembly import assembly_2d
from FEM.boundary import get_boundary_nodes, dirbc_2d

# Use custom style
plt.style.use("src/FEM/fem.mplstyle")

# Create output directory
output_dir = Path("figures/A2/ex_5")
output_dir.mkdir(parents=True, exist_ok=True)

# Case 3 uses MMS with sin(x)*sin(y) to demonstrate proper O(h^2) convergence


def main():
    print("Exercise 2.5: Dirichlet Problem and Convergence")
    print("=" * 50)

    # ============================================================
    # CASE 1: u(x,y) = x^3 - x^2*y + y^2 - 1
    # ============================================================
    print("\nCASE 1: u(x,y) = x^3 - x^2*y + y^2 - 1")
    print("-" * 50)

    x0, y0 = -2.5, -4.8
    L1, L2 = 7.6, 5.9

    mesh1 = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=4, noelms2=3)

    def u_exact_1(x, y):
        return x**3 - x**2 * y + y**2 - 1

    def q_1(x, y):
        return -6 * x + 2 * y - 2

    qt1 = q_1(mesh1.VX, mesh1.VY)

    A1, b1 = assembly_2d(mesh1, qt1)

    bnodes1 = get_boundary_nodes(mesh1)
    idx1 = bnodes1 - 1  # Convert to 0-based indices
    f1 = u_exact_1(mesh1.VX[idx1], mesh1.VY[idx1])
    A1, b1 = dirbc_2d(bnodes1, f1, A1, b1)

    u_h1 = spsolve(A1, b1)
    u_ex1 = u_exact_1(mesh1.VX, mesh1.VY)

    error1 = np.abs(u_h1 - u_ex1)
    E1 = np.max(error1)

    print(f"\n  Max error E (case 1)  = {E1:.6e}")

    # ============================================================
    # CASE 2: u(x,y) = x^2 * y^2 - Convergence Analysis
    # ============================================================
    print("\n\nCASE 2: u(x,y) = x^2 * y^2 - Convergence Analysis")
    print("-" * 50)

    def u_exact_2(x, y):
        return x**2 * y**2

    def q_2(x, y):
        return -2 * (x**2 + y**2)

    p_values = range(4, 8)
    errors = []
    h_values = []

    print("  " + "-" * 45)

    for p in p_values:
        noelms = 2**p
        mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=noelms, noelms2=noelms)
        h = np.sqrt((L1 / noelms) ** 2 + (L2 / noelms) ** 2)

        qt = q_2(mesh.VX, mesh.VY)
        A, b = assembly_2d(mesh, qt)

        bnodes = get_boundary_nodes(mesh)
        idx = bnodes - 1  # Convert to 0-based indices
        f = u_exact_2(mesh.VX[idx], mesh.VY[idx])
        A, b = dirbc_2d(bnodes, f, A, b)

        u_h = spsolve(A, b)
        u_ex = u_exact_2(mesh.VX, mesh.VY)
        E = np.max(np.abs(u_h - u_ex))

        errors.append(E)
        h_values.append(h)
        print(f"  {p:<5} {noelms:<10} {h:<15.6f} {E:<15.6e}")

    errors = np.array(errors)
    h_values = np.array(h_values)

    # Convergence rate
    print("\n  Convergence rates:")
    for i in range(len(errors) - 1):
        rate = np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
        print(
            f"    p={list(p_values)[i]} to p={list(p_values)[i + 1]}: rate = {rate:.2f}"
        )

    log_h = np.log(h_values)
    log_E = np.log(errors)
    coeffs = np.polyfit(log_h, log_E, 1)
    alpha = coeffs[0]
    C = np.exp(coeffs[1])

    print(f"\n  Least squares fit: E ~ {C:.4f} * h^{alpha:.2f}")

    # ============================================================
    # CASE 3: u(x,y) = sin(x)*sin(y) - MMS Convergence Analysis
    # ============================================================
    print("\n\nCASE 3: u(x,y) = sin(x)*sin(y) - MMS Convergence Analysis")
    print("-" * 50)

    # Domain [0, pi] x [0, pi] for nice boundary conditions (u=0 on boundary)
    x0_mms, y0_mms = 0.0, 0.0
    L1_mms, L2_mms = np.pi, np.pi

    def u_exact_mms(x, y):
        return np.sin(x) * np.sin(y)

    def q_mms(x, y):
        # -Laplacian of sin(x)*sin(y) = 2*sin(x)*sin(y)
        return 2 * np.sin(x) * np.sin(y)

    errors_mms = []
    h_values_mms = []

    print("  " + "-" * 45)

    for p in p_values:
        noelms = 2**p
        mesh = Mesh2d(x0=x0_mms, y0=y0_mms, L1=L1_mms, L2=L2_mms, noelms1=noelms, noelms2=noelms)
        h = np.sqrt((L1_mms / noelms) ** 2 + (L2_mms / noelms) ** 2)

        qt = q_mms(mesh.VX, mesh.VY)
        A, b = assembly_2d(mesh, qt)

        bnodes = get_boundary_nodes(mesh)
        idx = bnodes - 1
        f = u_exact_mms(mesh.VX[idx], mesh.VY[idx])
        A, b = dirbc_2d(bnodes, f, A, b)

        u_h = spsolve(A, b)
        u_ex = u_exact_mms(mesh.VX, mesh.VY)
        E = np.max(np.abs(u_h - u_ex))

        errors_mms.append(E)
        h_values_mms.append(h)
        print(f"  {p:<5} {noelms:<10} {h:<15.6f} {E:<15.6e}")

    errors_mms = np.array(errors_mms)
    h_values_mms = np.array(h_values_mms)

    # Convergence rate
    print("\n  Convergence rates:")
    for i in range(len(errors_mms) - 1):
        rate = np.log(errors_mms[i] / errors_mms[i + 1]) / np.log(h_values_mms[i] / h_values_mms[i + 1])
        print(
            f"    p={list(p_values)[i]} to p={list(p_values)[i + 1]}: rate = {rate:.2f}"
        )

    log_h_mms = np.log(h_values_mms)
    log_E_mms = np.log(errors_mms)
    coeffs_mms = np.polyfit(log_h_mms, log_E_mms, 1)
    alpha_mms = coeffs_mms[0]
    C_mms = np.exp(coeffs_mms[1])

    print(f"\n  Least squares fit: E ~ {C_mms:.4f} * h^{alpha_mms:.2f}")

    # ============================================================
    # Convergence Plots
    # ============================================================

    # Plot for Case 2 (polynomial)
    fig, ax = plt.subplots()
    ax.loglog(h_values, errors, "o-", label="Computed error")
    ax.loglog(h_values, C * h_values**2, "--", label=r"$O(h^2)$ reference")
    ax.set_xlabel("h (max element edge length)")
    ax.set_ylabel("E (max error)")
    ax.set_title(r"FEM Error Convergence for $u(x,y) = x^2 y^2$")
    ax.legend()
    fig.savefig(output_dir / "convergence_plot.pdf")
    plt.close(fig)

    # Plot for Case 3 (MMS with sin)
    fig, ax = plt.subplots()
    ax.loglog(h_values_mms, errors_mms, "o-", label="Computed error")
    ax.loglog(h_values_mms, C_mms * h_values_mms**2, "--", label=r"$O(h^2)$ reference")
    ax.set_xlabel("h (max element edge length)")
    ax.set_ylabel("E (max error)")
    ax.set_title(r"FEM Error Convergence for $u(x,y) = \sin(x)\sin(y)$ (MMS)")
    ax.legend()
    fig.savefig(output_dir / "convergence_plot_mms.pdf")
    plt.close(fig)

    # Combined comparison plot
    fig, ax = plt.subplots()
    ax.loglog(h_values, errors, "o-", label=r"$u = x^2 y^2$")
    ax.loglog(h_values_mms, errors_mms, "s-", label=r"$u = \sin(x)\sin(y)$")
    ax.loglog(h_values, C * h_values**2, "k--", alpha=0.5, label=r"$O(h^2)$ reference")
    ax.set_xlabel("h (max element edge length)")
    ax.set_ylabel("E (max error)")
    ax.set_title("FEM Error Convergence Comparison")
    ax.legend()
    fig.savefig(output_dir / "convergence_comparison.pdf")
    plt.close(fig)



if __name__ == "__main__":
    main()
