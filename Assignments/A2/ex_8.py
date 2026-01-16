"""
Exercise 2.8
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
from scipy.interpolate import LinearNDInterpolator

from FEM.datastructures import Mesh2d
from FEM.solvers import solve_mixed_bc_2d, solve_dirichlet_bc_2d

np.set_printoptions(precision=6, suppress=True, linewidth=160)
# Use custom style
plt.style.use("src/FEM/fem.mplstyle")

# Output directory
output_dir = Path("figures/A2/ex_8")
output_dir.mkdir(parents=True, exist_ok=True)

print("Exercise 2.8: Exploiting Symmetry with Neumann BCs")
print("=" * 60)


# Test problem: u = cos(πx)cos(πy)
def u_exact(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def q_tilde(x, y):
    return 2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)


def q_left(x, _y):
    return np.zeros_like(x)  


def q_bottom(x, _y):
    return np.zeros_like(x) 


# ============================================================
# Run convergence studies for both diagonal types
# ============================================================
diagonals = ["nw_se", "sw_ne"]
p_values = range(1, 10)
results = {}

for diag in diagonals:
    print(f"\n{'='*60}")
    print(f"DIAGONAL TYPE: {diag}")
    print("=" * 60)

    # ============================================================
    # Program b): Quarter domain [0,1]² with mixed BCs
    # ============================================================
    print(f"\nProgram b): Quarter domain [0,1]², diagonal={diag}")
    print("-" * 60)

    # Print solution in 2-D format for noelms=3 (as specified in exercise)
    mesh_b = Mesh2d(x0=0.0, y0=0.0, L1=1.0, L2=1.0, noelms1=3, noelms2=3, diagonal=diag)
    u_b = solve_mixed_bc_2d(mesh_b, q_tilde, q_left, q_bottom, u_exact)
    u_b_2d = u_b.reshape((mesh_b.noelms2 + 1, mesh_b.noelms1 + 1), order='F')
    print("\n  Solution u_h (2-D format):")
    print(u_b_2d)

    # Convergence study
    print(f"\n  {'p':<4} {'noelms':<8} {'h':<10} {'DOFs':<8} {'Time (ms)':<12} {'u(0,0)':<14} {'E_inf':<12}")
    print("  " + "-" * 75)

    errors_b = []
    dofs_b = []
    h_values = []
    times_b = []

    for p in p_values:
        noelms = 2 * p
        h = 1.0 / noelms

        mesh = Mesh2d(x0=0.0, y0=0.0, L1=1.0, L2=1.0, noelms1=noelms, noelms2=noelms, diagonal=diag)
        n_dofs = len(mesh.VX)

        t0 = time.perf_counter()
        u_h = solve_mixed_bc_2d(mesh, q_tilde, q_left, q_bottom, u_exact)
        t_solve = (time.perf_counter() - t0) * 1000

        u_ex = u_exact(mesh.VX, mesh.VY)
        E_inf = np.max(np.abs(u_h - u_ex))

        # u(0,0) is at index 0 for quarter domain starting at origin
        u_00 = u_h[0]

        errors_b.append(E_inf)
        dofs_b.append(n_dofs)
        h_values.append(h)
        times_b.append(t_solve)

        print(f"  {p:<4} {noelms:<8} {h:<10.4f} {n_dofs:<8} {t_solve:<12.3f} {u_00:<14.6f} {E_inf:<12.4e}")

    # ============================================================
    # Program c): Full domain [-1,1]² with Dirichlet BCs
    # ============================================================
    print(f"\nProgram c): Full domain [-1,1]², diagonal={diag}")
    print("-" * 60)

    # Print solution in 2-D format for noelms=6
    mesh_c = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=6, noelms2=6, diagonal=diag)
    u_c = solve_dirichlet_bc_2d(mesh_c, q_tilde, u_exact)
    u_c_2d = u_c.reshape((mesh_c.noelms2 + 1, mesh_c.noelms1 + 1), order='F')
    print("\n  Solution u_h (2-D format):")
    print(u_c_2d)

    print(f"\n  {'p':<4} {'noelms':<8} {'h':<10} {'DOFs':<8} {'Time (ms)':<12} {'u(0,0)':<14} {'E_inf':<12}")
    print("  " + "-" * 75)

    errors_c = []
    dofs_c = []
    times_c = []

    for p in p_values:
        noelms = 2 * p
        h = 2.0 / noelms

        mesh = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=noelms, noelms2=noelms, diagonal=diag)
        n_dofs = len(mesh.VX)

        t0 = time.perf_counter()
        u_h = solve_dirichlet_bc_2d(mesh, q_tilde, u_exact)
        t_solve = (time.perf_counter() - t0) * 1000

        u_ex = u_exact(mesh.VX, mesh.VY)
        E_inf = np.max(np.abs(u_h - u_ex))

        # Find node at (0,0) - center of the full domain
        idx_00 = np.argmin(mesh.VX**2 + mesh.VY**2)
        u_00 = u_h[idx_00]

        errors_c.append(E_inf)
        dofs_c.append(n_dofs)
        times_c.append(t_solve)

        print(f"  {p:<4} {noelms:<8} {h:<10.4f} {n_dofs:<8} {t_solve:<12.3f} {u_00:<14.6f} {E_inf:<12.4e}")

    # Store results
    results[diag] = {
        "errors_b": np.array(errors_b),
        "dofs_b": np.array(dofs_b),
        "h_values": np.array(h_values),
        "times_b": np.array(times_b),
        "errors_c": np.array(errors_c),
        "dofs_c": np.array(dofs_c),
        "times_c": np.array(times_c),
    }

# ============================================================
# Comparison plots (Combined)
# ============================================================
def calc_rate(h, err):
    """Calculate convergence rate from last 4 points."""
    if len(h) < 4:
        return np.nan
    slope, _ = np.polyfit(np.log(h[-4:]), np.log(err[-4:]), 1)
    return slope

# Create a single figure with 2 subplots for all results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define unique styles for each combination (BC, Split)
styles = {
    ("Mixed", "nw_se"): {"color": "tab:blue", "marker": "o", "ls": "-"},
    ("Mixed", "sw_ne"): {"color": "tab:orange", "marker": "s", "ls": "--"},
    ("Dirichlet", "nw_se"): {"color": "tab:green", "marker": "D", "ls": "-"},
    ("Dirichlet", "sw_ne"): {"color": "tab:red", "marker": "^", "ls": "--"},
}

for diag in diagonals:
    r = results[diag]
    
    # Calculate rates
    h_b = r["h_values"]
    h_c = 2 * r["h_values"]
    
    rate_b = calc_rate(h_b, r["errors_b"])
    rate_c = calc_rate(h_c, r["errors_c"])

    # Plot Mixed
    s_m = styles[("Mixed", diag)]
    ax1.loglog(r["dofs_b"], r["errors_b"], 
               marker=s_m["marker"], linestyle=s_m["ls"], color=s_m["color"], 
               label=f"Mixed ({diag}) [rate={rate_b:.2f}]")
    ax2.loglog(r["times_b"], r["errors_b"], 
               marker=s_m["marker"], linestyle=s_m["ls"], color=s_m["color"],
               label=f"Mixed ({diag})")

    # Plot Dirichlet
    s_d = styles[("Dirichlet", diag)]
    ax1.loglog(r["dofs_c"], r["errors_c"], 
               marker=s_d["marker"], linestyle=s_d["ls"], color=s_d["color"],
               label=f"Dirichlet ({diag}) [rate={rate_c:.2f}]")
    ax2.loglog(r["times_c"], r["errors_c"], 
               marker=s_d["marker"], linestyle=s_d["ls"], color=s_d["color"],
               label=f"Dirichlet ({diag})")

# Add reference line (O(N^-1)) based on the first dataset
r0 = results[diagonals[0]]
dofs_ref = np.array([r0["dofs_b"][0], r0["dofs_b"][-1]])
ref_line = r0["errors_b"][0] * (dofs_ref / dofs_ref[0]) ** (-1)
ax1.loglog(dofs_ref, ref_line, "k:", alpha=0.5, label=r"$O(\mathrm{DOFs}^{-1})$")

# Formatting Ax1 (DOFs)
ax1.set_xlabel("DOFs")
ax1.set_ylabel(r"$L^\infty$ error")
ax1.set_title(r"$L^\infty$ Error vs DOFs")
ax1.legend(fontsize='small')
ax1.grid(True, alpha=0.3)

# Formatting Ax2 (Time)
ax2.set_xlabel("CPU Time (ms)")
ax2.set_ylabel(r"$L^\infty$ error")
ax2.set_title(r"$L^\infty$ Error vs CPU Time")
ax2.legend(fontsize='small')
ax2.grid(True, alpha=0.3)


fig.tight_layout()
fig.savefig(output_dir / "convergence_combined.pdf")
plt.close(fig)

# ============================================================
# Visual Comparison (triplots) for each diagonal type
# ============================================================
n_elems = 32
n_vis = 64

for diag in diagonals:
    # Quarter domain with mixed BCs
    mesh_quarter = Mesh2d(x0=0.0, y0=0.0, L1=1.0, L2=1.0, noelms1=n_elems, noelms2=n_elems, diagonal=diag)
    u_quarter = solve_mixed_bc_2d(mesh_quarter, q_tilde, q_left, q_bottom, u_exact)
    dofs_quarter = len(mesh_quarter.VX)
    h_quarter = 1.0 / n_elems

    # Full domain with Dirichlet BCs
    mesh_dirichlet = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=n_elems, noelms2=n_elems, diagonal=diag)
    u_dirichlet = solve_dirichlet_bc_2d(mesh_dirichlet, q_tilde, u_exact)
    dofs_dirichlet = len(mesh_dirichlet.VX)
    h_dirichlet = 2.0 / n_elems

    # Create fine visualization mesh
    mesh_vis = Mesh2d(x0=-1.0, y0=-1.0, L1=2.0, L2=2.0, noelms1=n_vis, noelms2=n_vis, diagonal=diag)
    u_ex_vis = u_exact(mesh_vis.VX, mesh_vis.VY)

    # Interpolate Dirichlet solution
    interp_dir = LinearNDInterpolator(list(zip(mesh_dirichlet.VX, mesh_dirichlet.VY)), u_dirichlet)
    u_dir_vis = interp_dir(mesh_vis.VX, mesh_vis.VY)

    # Reconstruct quarter solution via symmetry
    interp_quarter = LinearNDInterpolator(list(zip(mesh_quarter.VX, mesh_quarter.VY)), u_quarter)

    def reconstruct_symmetric(x, y):
        x_q, y_q = np.abs(x), np.abs(y)
        return interp_quarter(x_q, y_q)

    u_mixed_vis = reconstruct_symmetric(mesh_vis.VX, mesh_vis.VY)

    # Compute errors
    err_dirichlet = np.abs(u_dir_vis - u_ex_vis)
    err_mixed = np.abs(u_mixed_vis - u_ex_vis)

    # Create triangulation for visualization
    tri = Triangulation(mesh_vis.VX, mesh_vis.VY, mesh_vis.EToV)

    # Plot 1: Solutions
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

    c0 = axes1[0].tripcolor(tri, u_dir_vis)
    axes1[0].set_title(f"Dirichlet ({dofs_dirichlet} DOFs, h={h_dirichlet:.3f})")
    axes1[0].set_aspect("equal")
    plt.colorbar(c0, ax=axes1[0])

    c1 = axes1[1].tripcolor(tri, u_mixed_vis)
    axes1[1].set_title(f"Mixed + Symmetry ({dofs_quarter} DOFs, h={h_quarter:.3f})")
    axes1[1].set_aspect("equal")
    plt.colorbar(c1, ax=axes1[1])


    fig1.tight_layout()
    fig1.savefig(output_dir / f"solution_triplot_{diag}.pdf")
    plt.close(fig1)

    # Plot 2: Errors
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    c2 = axes2[0].tripcolor(tri, err_dirichlet)
    axes2[0].set_title(f"Dirichlet Error (max={np.nanmax(err_dirichlet):.2e})")
    axes2[0].set_aspect("equal")
    plt.colorbar(c2, ax=axes2[0])

    c3 = axes2[1].tripcolor(tri, err_mixed)
    axes2[1].set_title(f"Mixed Error (max={np.nanmax(err_mixed):.2e})")
    axes2[1].set_aspect("equal")
    plt.colorbar(c3, ax=axes2[1])


    fig2.tight_layout()
    fig2.savefig(output_dir / f"error_triplot_{diag}.pdf")
    plt.close(fig2)

