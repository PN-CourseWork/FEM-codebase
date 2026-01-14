"""
Test script for lid-driven cavity at Re=100.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from FEM.datastructures import Mesh2d
from FEM.navier_stokes import lid_driven_cavity, interpolate_fem, extract_centerline


def main():
    parser = argparse.ArgumentParser(description="Lid-driven cavity solver")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Path to mesh file (if not provided, uses structured mesh)",
    )
    parser.add_argument(
        "--n-elements",
        type=int,
        default=32,
        help="Number of elements per side (for structured mesh)",
    )
    args = parser.parse_args()

    # Create mesh
    if args.mesh is not None:
        print(f"Loading mesh from {args.mesh}")
        mesh = Mesh2d.from_meshio(args.mesh)
        mesh_label = args.mesh.stem
    else:
        noelms = args.n_elements
        mesh = Mesh2d(
            x0=0.0, y0=0.0,
            L1=1.0, L2=1.0,
            noelms1=noelms, noelms2=noelms
        )
        mesh_label = f"{noelms}x{noelms}"

    # Parameters
    Re = 100
    dt = 0.001  # Time step (stability requires dt < h^2 / nu for explicit parts)
    n_steps = 15000  # Run until steady state

    # Solve to steady state
    omega, psi, u, v = lid_driven_cavity(mesh, Re, dt, n_steps, print_every=1000, solver="lu", steady_tol=1e-6)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Triangulation for plotting (EToV is 1-based, matplotlib needs 0-based)
    triang = Triangulation(mesh.VX, mesh.VY, mesh.EToV - 1)

    # Stream function contours
    ax = axes[0, 0]
    levels = np.linspace(psi.min(), psi.max(), 20)
    cs = ax.tricontourf(triang, psi, levels=levels, cmap='RdBu_r')
    ax.tricontour(triang, psi, levels=levels, colors='k', linewidths=0.5)
    plt.colorbar(cs, ax=ax, label=r'$\psi$')
    ax.set_title('Stream Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # Vorticity
    ax = axes[0, 1]
    levels_omega = np.linspace(-10, 10, 21)
    cs = ax.tricontourf(triang, omega, levels=levels_omega, cmap='RdBu_r', extend='both')
    plt.colorbar(cs, ax=ax, label=r'$\omega$')
    ax.set_title('Vorticity')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # Velocity magnitude
    ax = axes[1, 0]
    speed = np.sqrt(u**2 + v**2)
    cs = ax.tricontourf(triang, speed, levels=20, cmap='viridis')
    plt.colorbar(cs, ax=ax, label='|u|')
    ax.set_title('Velocity Magnitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    # Velocity vectors (subsample for clarity)
    ax = axes[1, 1]
    # Create regular grid for quiver
    nx, ny = 20, 20
    xi = np.linspace(0, 1, nx)
    yi = np.linspace(0, 1, ny)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate to regular grid
    from scipy.interpolate import griddata
    Ui = griddata((mesh.VX, mesh.VY), u, (Xi, Yi), method='linear')
    Vi = griddata((mesh.VX, mesh.VY), v, (Xi, Yi), method='linear')

    ax.quiver(Xi, Yi, Ui, Vi)
    ax.set_title('Velocity Vectors')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(f'Lid-Driven Cavity, Re={Re}, {mesh_label} mesh', fontsize=14)
    fig.tight_layout()

    plt.savefig('figures/lid_driven_cavity_Re100.pdf')
    # plt.show()

    # Ghia et al. (1982) benchmark data for Re=100
    # u velocity along vertical centerline (x=0.5)
    ghia_y = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
                       0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
                       0.9688, 0.9766, 1.0000])
    ghia_u = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.00000])

    # v velocity along horizontal centerline (y=0.5)
    ghia_x = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266,
                       0.2344, 0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531,
                       0.9609, 0.9688, 1.0000])
    ghia_v = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                       0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                       -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])

    # Interpolate using proper FEM basis functions
    # u along vertical centerline (x=0.5)
    points_u = np.column_stack([np.full_like(ghia_y, 0.5), ghia_y])
    u_computed = interpolate_fem(mesh, u, points_u)

    # v along horizontal centerline (y=0.5)
    points_v = np.column_stack([ghia_x, np.full_like(ghia_x, 0.5)])
    v_computed = interpolate_fem(mesh, v, points_v)

    # Also extract full centerline profiles for smoother plotting
    y_center, u_centerline = extract_centerline(mesh, u, axis='x', position=0.5, n_points=100)
    x_center, v_centerline = extract_centerline(mesh, v, axis='y', position=0.5, n_points=100)

    # Create Ghia comparison plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # u velocity along vertical centerline
    ax = axes2[0]
    ax.plot(ghia_u, ghia_y, 'ro', markersize=8, label='Ghia et al. (1982)')
    ax.plot(u_centerline, y_center, 'b-', linewidth=2, label='FEM (this work)')
    ax.plot(u_computed, ghia_y, 'bx', markersize=6)  # FEM at Ghia points
    ax.set_xlabel('u')
    ax.set_ylabel('y')
    ax.set_title('u velocity along x=0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.4, 1.1)
    ax.set_ylim(0, 1)

    # v velocity along horizontal centerline
    ax = axes2[1]
    ax.plot(ghia_x, ghia_v, 'ro', markersize=8, label='Ghia et al. (1982)')
    ax.plot(x_center, v_centerline, 'b-', linewidth=2, label='FEM (this work)')
    ax.plot(ghia_x, v_computed, 'bx', markersize=6)  # FEM at Ghia points
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title('v velocity along y=0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.2)

    fig2.suptitle(f'Comparison with Ghia et al. (1982), Re={Re}', fontsize=14)
    fig2.tight_layout()

    plt.savefig('figures/lid_driven_cavity_ghia_comparison.pdf')
    # plt.show()

    # Print error metrics
    print("\nComparison with Ghia et al. (1982):")
    u_error = np.nanmean(np.abs(u_computed - ghia_u))
    v_error = np.nanmean(np.abs(v_computed - ghia_v))
    print(f"  Mean absolute error in u: {u_error:.4f}")
    print(f"  Mean absolute error in v: {v_error:.4f}")


if __name__ == "__main__":
    main()
