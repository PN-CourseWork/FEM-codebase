
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

from FEM.datastructures import Mesh2d
from FEM.solvers import solve_mixed_bc_2d, solve_dirichlet_bc_2d

# Use custom style
plt.style.use("src/FEM/fem.mplstyle")

# Output directory
output_dir = Path("figures/A2/ex_8")
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    print("Exercise 2.8: Exploiting Symmetry with Neumann BCs")
    print("=" * 60)

    # Test problem: u = cos(πx)cos(πy)
    # PDE: u_xx + u_yy = -2π²cos(πx)cos(πy)
    def u_exact(x, y):
        return np.cos(np.pi * x) * np.cos(np.pi * y)

    def q_tilde(x, y):
        return 2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)

    # Neumann BCs: ∂u/∂n on left (x=0) and bottom (y=0)
    # On x=0: ∂u/∂x = -π*sin(πx)*cos(πy) = 0
    # On y=0: ∂u/∂y = -π*cos(πx)*sin(πy) = 0
    def q_left(x, y):
        return np.zeros_like(x)  # Homogeneous Neumann (symmetry)

    def q_bottom(x, y):
        return np.zeros_like(x)  # Homogeneous Neumann (symmetry)

    # ============================================================
    # Program b): Quarter domain [0,1]² with mixed BCs, noelms = 2p
    # ============================================================
    print("\nProgram b): Quarter domain [0,1]², noelms = 2p")
    print("-" * 60)

    p_values = range(1, 10)
    errors_b = []
    errors_b_l2 = []
    dofs_b = []
    h_values = []
    times_b = []

    print(f"  {'p':<4} {'noelms':<8} {'h':<10} {'DOFs':<8} {'Time (ms)':<12} {'E_inf':<12} {'E_L2':<12}")
    print("  " + "-" * 75)

    for p in p_values:
        noelms = 2 * p  # Even: 2, 4, 6, 8, ...
        h = 1.0 / noelms  # Element size on [0,1]

        mesh = Mesh2d(x0=0.0, y0=0.0, L1=1.0, L2=1.0, noelms1=noelms, noelms2=noelms)
        n_dofs = len(mesh.VX)

        t0 = time.perf_counter()
        u_h = solve_mixed_bc_2d(mesh, q_tilde, q_left, q_bottom, u_exact)
        t_solve = (time.perf_counter() - t0) * 1000  # ms

        u_ex = u_exact(mesh.VX, mesh.VY)
        E_inf = np.max(np.abs(u_h - u_ex))
        E_l2 = np.sqrt(np.mean((u_h - u_ex) ** 2))

        errors_b.append(E_inf)
        errors_b_l2.append(E_l2)
        dofs_b.append(n_dofs)
        h_values.append(h)
        times_b.append(t_solve)

        print(f"  {p:<4} {noelms:<8} {h:<10.4f} {n_dofs:<8} {t_solve:<12.3f} {E_inf:<12.4e} {E_l2:<12.4e}")

    errors_b = np.array(errors_b)
    errors_b_l2 = np.array(errors_b_l2)
    dofs_b = np.array(dofs_b)
    h_values = np.array(h_values)
    times_b = np.array(times_b)

    # ============================================================
    # Program c): Full domain [-1,1]² with Dirichlet BCs, noelms = 2p
    # Same DOFs as program b (same computational cost)
    # ============================================================
    print("\n\nProgram c): Full domain [-1,1]², noelms = 2p (same DOFs)")
    print("-" * 60)

    errors_c = []
    errors_c_l2 = []
    dofs_c = []
    times_c = []

    print(f"  {'p':<4} {'noelms':<8} {'h':<10} {'DOFs':<8} {'Time (ms)':<12} {'E_inf':<12} {'E_L2':<12}")
    print("  " + "-" * 75)

    for p in p_values:
        noelms = 2 * p  # Same noelms as quarter → same DOFs
        h = 2.0 / noelms  # Element size on [-1,1]² (2x coarser than quarter)

        mesh = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=noelms, noelms2=noelms)
        n_dofs = len(mesh.VX)

        t0 = time.perf_counter()
        u_h = solve_dirichlet_bc_2d(mesh, q_tilde, u_exact)
        t_solve = (time.perf_counter() - t0) * 1000  # ms

        u_ex = u_exact(mesh.VX, mesh.VY)
        E_inf = np.max(np.abs(u_h - u_ex))
        E_l2 = np.sqrt(np.mean((u_h - u_ex) ** 2))

        errors_c.append(E_inf)
        errors_c_l2.append(E_l2)
        dofs_c.append(n_dofs)
        times_c.append(t_solve)

        print(f"  {p:<4} {noelms:<8} {h:<10.4f} {n_dofs:<8} {t_solve:<12.3f} {E_inf:<12.4e} {E_l2:<12.4e}")

    errors_c = np.array(errors_c)
    errors_c_l2 = np.array(errors_c_l2)
    dofs_c = np.array(dofs_c)
    times_c = np.array(times_c)

    # ============================================================
    # Comparison Plot: Error vs DOFs (same DOFs for both)
    # ============================================================
    print("\n\n" + "=" * 60)
    print("COMPARISON (same DOFs)")
    print("=" * 60)
    print(f"  Same DOFs, but quarter domain has finer mesh (h_quarter = h_full/2)")
    print(f"  Quarter DOFs: {dofs_b}")
    print(f"  Full DOFs:    {dofs_c}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reference line: O(DOFs^{-1}) = O(h^2) since DOFs ~ 1/h^2
    dofs_ref = np.array([dofs_b[0], dofs_b[-1]])
    ref_line = errors_b[0] * (dofs_ref / dofs_ref[0]) ** (-1)

    # L-infinity norm comparison
    ax1.loglog(dofs_b, errors_b, "s-", label="Mixed (quarter domain)")
    ax1.loglog(dofs_c, errors_c, "o-", label="Dirichlet (full domain)")
    ax1.loglog(dofs_ref, ref_line, "--", color="gray", label=r"$O(\mathrm{DOFs}^{-1})$")
    ax1.set_xlabel("DOFs")
    ax1.set_ylabel(r"$L^\infty$ error")
    ax1.set_title(r"$L^\infty$ Error vs DOFs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # L2 norm comparison
    ref_line_l2 = errors_b_l2[0] * (dofs_ref / dofs_ref[0]) ** (-1)
    ax2.loglog(dofs_b, errors_b_l2, "s-", label="Mixed (quarter domain)", markersize=8)
    ax2.loglog(dofs_c, errors_c_l2, "o-", label="Dirichlet (full domain)", markersize=8)
    ax2.loglog(dofs_ref, ref_line_l2, "--", color="gray", label=r"$O(\mathrm{DOFs}^{-1})$")
    ax2.set_xlabel("DOFs")
    ax2.set_ylabel(r"$L^2$ error")
    ax2.set_title(r"$L^2$ Error vs DOFs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(r"Same DOFs: Mixed BC (symmetry) vs Dirichlet ($u = \cos(\pi x)\cos(\pi y)$)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "even_vs_odd_convergence.pdf")
    plt.close(fig)

    # ============================================================
    # CPU Time Plot: Error vs CPU Time
    # ============================================================
    print("\n  CPU Time Comparison:")
    print(f"    Quarter domain times (ms): {times_b}")
    print(f"    Full domain times (ms):    {times_c}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # L-infinity error vs CPU time
    ax1.loglog(times_b, errors_b, "s-", label="Mixed (quarter domain)")
    ax1.loglog(times_c, errors_c, "o-", label="Dirichlet (full domain)")
    ax1.set_xlabel("CPU Time (ms)")
    ax1.set_ylabel(r"$L^\infty$ error")
    ax1.set_title(r"$L^\infty$ Error vs CPU Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # L2 error vs CPU time
    ax2.loglog(times_b, errors_b_l2, "s-", label="Mixed (quarter domain)")
    ax2.loglog(times_c, errors_c_l2, "o-", label="Dirichlet (full domain)")
    ax2.set_xlabel("CPU Time (ms)")
    ax2.set_ylabel(r"$L^2$ error")
    ax2.set_title(r"$L^2$ Error vs CPU Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(r"Computational Efficiency: Error vs CPU Time ($u = \cos(\pi x)\cos(\pi y)$)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_time.pdf")
    plt.close(fig)

    # ============================================================
    # Visual Comparison: Same DOFs for both methods
    # ============================================================
    from scipy.interpolate import LinearNDInterpolator

    # Use same noelms (same DOFs) for fair comparison
    n_elems = 32  # Same for both → same DOFs, but different h

    # 1. Quarter domain with mixed BCs (h = 1/32)
    mesh_quarter = Mesh2d(x0=0.0, y0=0.0, L1=1.0, L2=1.0, noelms1=n_elems, noelms2=n_elems)
    u_quarter = solve_mixed_bc_2d(mesh_quarter, q_tilde, q_left, q_bottom, u_exact)
    dofs_quarter = len(mesh_quarter.VX)
    h_quarter = 1.0 / n_elems

    # 2. Full domain with Dirichlet BCs (h = 2/32 = 1/16, same DOFs)
    mesh_dirichlet = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=n_elems, noelms2=n_elems)
    u_dirichlet = solve_dirichlet_bc_2d(mesh_dirichlet, q_tilde, u_exact)
    dofs_dirichlet = len(mesh_dirichlet.VX)
    h_dirichlet = 2.0 / n_elems

    # Create fine visualization mesh
    n_vis = 64
    mesh_vis = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=n_vis, noelms2=n_vis)
    u_ex_vis = u_exact(mesh_vis.VX, mesh_vis.VY)

    # Interpolate Dirichlet solution to vis mesh
    interp_dir = LinearNDInterpolator(
        list(zip(mesh_dirichlet.VX, mesh_dirichlet.VY)), u_dirichlet
    )
    u_dir_vis = interp_dir(mesh_vis.VX, mesh_vis.VY)

    # Create interpolator for quarter domain and reconstruct via symmetry
    interp_quarter = LinearNDInterpolator(
        list(zip(mesh_quarter.VX, mesh_quarter.VY)), u_quarter
    )

    def reconstruct_symmetric(x, y):
        """Reconstruct solution on full domain using even-even symmetry."""
        x_q, y_q = np.abs(x), np.abs(y)
        return interp_quarter(x_q, y_q)

    u_mixed_vis = reconstruct_symmetric(mesh_vis.VX, mesh_vis.VY)

    # Compute errors
    err_dirichlet = np.abs(u_dir_vis - u_ex_vis)
    err_mixed = np.abs(u_mixed_vis - u_ex_vis)

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    tri = Triangulation(mesh_vis.VX, mesh_vis.VY, mesh_vis.EToV - 1)

    # Top row: solutions
    c0 = axes[0, 0].tripcolor(tri, u_dir_vis)
    axes[0, 0].set_title(f"Dirichlet ({dofs_dirichlet} DOFs, h={h_dirichlet:.3f})")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(c0, ax=axes[0, 0])

    c1 = axes[0, 1].tripcolor(tri, u_mixed_vis)
    axes[0, 1].set_title(f"Mixed + Symmetry ({dofs_quarter} DOFs, h={h_quarter:.3f})")
    axes[0, 1].set_aspect("equal")
    plt.colorbar(c1, ax=axes[0, 1])

    # Bottom row: errors
    c2 = axes[1, 0].tripcolor(tri, err_dirichlet)
    axes[1, 0].set_title(f"Dirichlet Error (max={np.nanmax(err_dirichlet):.2e})")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(c2, ax=axes[1, 0])

    c3 = axes[1, 1].tripcolor(tri, err_mixed)
    axes[1, 1].set_title(f"Mixed Error (max={np.nanmax(err_mixed):.2e})")
    axes[1, 1].set_aspect("equal")
    plt.colorbar(c3, ax=axes[1, 1])

    fig.suptitle(r"$u = \cos(\pi x)\cos(\pi y)$: Same DOFs, Different $h$")
    fig.tight_layout()
    fig.savefig(output_dir / "solution_triplot.pdf")
    plt.close(fig)

    print(f"\n  Plots saved to: {output_dir}")
    print("\nAll even vs odd tests completed!")

#TODO: estimated convergence rates for the few largest amount of DOFs 

if __name__ == "__main__":
    main()
