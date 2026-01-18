"""Spectral Element Method Convergence Study.

This script demonstrates the h-convergence and p-convergence properties
of the Spectral Element Method for solving the Poisson equation:

    -∇²u = f  on Ω = [0,1]²
    u = g     on ∂Ω

Using manufactured solution: u = sin(πx)sin(πy)

Run with: uv run python assignments/A3/sem_convergence_study.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from FEM.SEM import (
    SEMMesh2D,
    create_unit_square_mesh,
    solve_poisson_sem,
    l2_error_sem,
    linf_error_sem,
)

# Output directory (figures/A3/ from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "figures" / "A3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Manufactured solution
def u_exact(x, y):
    """Exact solution: u = sin(πx)sin(πy)."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f_rhs(x, y):
    """RHS: -∇²u = 2π²sin(πx)sin(πy)."""
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def bc_func(x, y):
    """Homogeneous Dirichlet BC."""
    return np.zeros_like(x)


def run_h_convergence(p_values, n_values):
    """Run h-convergence study for multiple polynomial orders.

    Parameters
    ----------
    p_values : list[int]
        Polynomial orders to test
    n_values : list[int]
        Number of elements per direction

    Returns
    -------
    dict
        Results keyed by p, each containing h, l2_error, linf_error, dof
    """
    results = {}

    for p in p_values:
        print(f"\n{'='*50}")
        print(f"Polynomial order p = {p}")
        print(f"{'='*50}")

        results[p] = {
            'n': [],
            'h': [],
            'dof': [],
            'l2_error': [],
            'linf_error': [],
        }

        for n in n_values:
            # Create mesh
            with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
                filepath = Path(f.name)

            create_unit_square_mesh(n, n, filepath)
            mesh = SEMMesh2D(filepath=filepath, polynomial_order=p)

            # Solve
            u = solve_poisson_sem(mesh, f_rhs, bc_func)

            # Compute errors
            l2_err = l2_error_sem(mesh, u, u_exact)
            linf_err = linf_error_sem(mesh, u, u_exact)

            h = 1.0 / n

            results[p]['n'].append(n)
            results[p]['h'].append(h)
            results[p]['dof'].append(mesh.nonodes)
            results[p]['l2_error'].append(l2_err)
            results[p]['linf_error'].append(linf_err)

            print(f"  n={n:2d}, h={h:.4f}, DOF={mesh.nonodes:5d}, "
                  f"L2={l2_err:.2e}, L∞={linf_err:.2e}")

            # Cleanup
            filepath.unlink()

    return results


def run_p_convergence(n_values, p_values):
    """Run p-convergence study for multiple mesh sizes.

    Parameters
    ----------
    n_values : list[int]
        Number of elements per direction (fixed mesh)
    p_values : list[int]
        Polynomial orders to test

    Returns
    -------
    dict
        Results keyed by n, each containing p, l2_error, linf_error, dof
    """
    results = {}

    for n in n_values:
        print(f"\n{'='*50}")
        print(f"Mesh: {n}×{n} elements")
        print(f"{'='*50}")

        results[n] = {
            'p': [],
            'dof': [],
            'l2_error': [],
            'linf_error': [],
        }

        for p in p_values:
            # Create mesh
            with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
                filepath = Path(f.name)

            create_unit_square_mesh(n, n, filepath)
            mesh = SEMMesh2D(filepath=filepath, polynomial_order=p)

            # Solve
            u = solve_poisson_sem(mesh, f_rhs, bc_func)

            # Compute errors
            l2_err = l2_error_sem(mesh, u, u_exact)
            linf_err = linf_error_sem(mesh, u, u_exact)

            results[n]['p'].append(p)
            results[n]['dof'].append(mesh.nonodes)
            results[n]['l2_error'].append(l2_err)
            results[n]['linf_error'].append(linf_err)

            print(f"  p={p:2d}, DOF={mesh.nonodes:5d}, "
                  f"L2={l2_err:.2e}, L∞={linf_err:.2e}")

            # Cleanup
            filepath.unlink()

    return results


def compute_convergence_rate(h, error):
    """Compute convergence rate from log-log slope."""
    if len(h) < 2:
        return np.nan
    log_h = np.log(h)
    log_err = np.log(error)
    # Linear regression
    coeffs = np.polyfit(log_h, log_err, 1)
    return coeffs[0]


def plot_h_convergence(results, filename="h_convergence.pdf"):
    """Plot h-convergence for different polynomial orders."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for idx, (p, data) in enumerate(sorted(results.items())):
        h = np.array(data['h'])
        l2_err = np.array(data['l2_error'])
        linf_err = np.array(data['linf_error'])

        # Compute convergence rates
        rate_l2 = compute_convergence_rate(h, l2_err)
        rate_linf = compute_convergence_rate(h, linf_err)

        color = colors[idx]
        marker = markers[idx % len(markers)]

        # L2 error
        ax1.loglog(h, l2_err, f'{marker}-', color=color,
                   label=f'p={p} (rate={rate_l2:.1f})',
                   markersize=8, linewidth=1.5)

        # L∞ error
        ax2.loglog(h, linf_err, f'{marker}-', color=color,
                   label=f'p={p} (rate={rate_linf:.1f})',
                   markersize=8, linewidth=1.5)

    # Reference slopes
    h_ref = np.array([0.5, 0.05])
    for ax in [ax1, ax2]:
        for slope, ls in [(2, ':'), (4, '--'), (6, '-.')]:
            y_ref = h_ref**slope * 0.5
            ax.loglog(h_ref, y_ref, ls, color='gray', alpha=0.5,
                     label=f'O(h^{slope})' if ax == ax1 else None)

    ax1.set_xlabel('Element size h', fontsize=12)
    ax1.set_ylabel('L² error', fontsize=12)
    ax1.set_title('L² Error Convergence', fontsize=14)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.03, 0.6])

    ax2.set_xlabel('Element size h', fontsize=12)
    ax2.set_ylabel('L∞ error', fontsize=12)
    ax2.set_title('L∞ Error Convergence', fontsize=14)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.03, 0.6])

    plt.suptitle('SEM h-Convergence: $-\\nabla^2 u = f$ on $[0,1]^2$\n'
                 '$u = \\sin(\\pi x)\\sin(\\pi y)$', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filepath}")
    plt.close()


def plot_p_convergence(results, filename="p_convergence.pdf"):
    """Plot p-convergence (exponential) for different mesh sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(results)))
    markers = ['o', 's', '^', 'D']

    for idx, (n, data) in enumerate(sorted(results.items())):
        p = np.array(data['p'])
        l2_err = np.array(data['l2_error'])
        linf_err = np.array(data['linf_error'])

        color = colors[idx]
        marker = markers[idx % len(markers)]

        # L2 error (semilogy for exponential convergence)
        ax1.semilogy(p, l2_err, f'{marker}-', color=color,
                     label=f'{n}×{n} elements',
                     markersize=8, linewidth=1.5)

        # L∞ error
        ax2.semilogy(p, linf_err, f'{marker}-', color=color,
                     label=f'{n}×{n} elements',
                     markersize=8, linewidth=1.5)

    ax1.set_xlabel('Polynomial order p', fontsize=12)
    ax1.set_ylabel('L² error', fontsize=12)
    ax1.set_title('L² Error (p-convergence)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(list(results.values())[0]['p'])

    ax2.set_xlabel('Polynomial order p', fontsize=12)
    ax2.set_ylabel('L∞ error', fontsize=12)
    ax2.set_title('L∞ Error (p-convergence)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(list(results.values())[0]['p'])

    plt.suptitle('SEM p-Convergence (Spectral Accuracy)\n'
                 '$u = \\sin(\\pi x)\\sin(\\pi y)$', fontsize=14, y=1.02)
    plt.tight_layout()

    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_error_vs_dof(h_results, filename="error_vs_dof.pdf"):
    """Plot error vs DOF (efficiency plot)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(h_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for idx, (p, data) in enumerate(sorted(h_results.items())):
        dof = np.array(data['dof'])
        l2_err = np.array(data['l2_error'])

        color = colors[idx]
        marker = markers[idx % len(markers)]

        ax.loglog(dof, l2_err, f'{marker}-', color=color,
                  label=f'p={p}', markersize=8, linewidth=1.5)

    ax.set_xlabel('Degrees of Freedom', fontsize=12)
    ax.set_ylabel('L² error', fontsize=12)
    ax.set_title('SEM Efficiency: Error vs DOF', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def create_convergence_table(results):
    """Print convergence table in LaTeX format."""
    print("\n" + "="*70)
    print("CONVERGENCE TABLE (LaTeX)")
    print("="*70)

    print("\n% h-convergence table")
    print("\\begin{tabular}{c|ccc|ccc}")
    print("\\hline")
    print("$p$ & $n$ & DOF & $h$ & $\\|e\\|_{L^2}$ & Rate & $\\|e\\|_{L^\\infty}$ \\\\")
    print("\\hline")

    for p, data in sorted(results.items()):
        for i, (n, dof, h, l2, linf) in enumerate(zip(
            data['n'], data['dof'], data['h'], data['l2_error'], data['linf_error']
        )):
            if i == 0:
                rate = "-"
            else:
                prev_h, prev_l2 = data['h'][i-1], data['l2_error'][i-1]
                rate = f"{np.log(prev_l2/l2) / np.log(prev_h/h):.2f}"

            p_str = str(p) if i == 0 else ""
            print(f"{p_str} & {n} & {dof} & {h:.3f} & {l2:.2e} & {rate} & {linf:.2e} \\\\")
        print("\\hline")

    print("\\end{tabular}")


def main():
    """Run the full convergence study."""
    print("="*70)
    print("SPECTRAL ELEMENT METHOD CONVERGENCE STUDY")
    print("="*70)
    print("\nManufactured solution: u = sin(πx)sin(πy)")
    print("Domain: [0,1]²")
    print("Boundary conditions: Homogeneous Dirichlet")

    # h-convergence study
    print("\n" + "="*70)
    print("H-CONVERGENCE STUDY")
    print("="*70)

    p_values = [2, 3, 4, 5, 6]
    n_values = [2, 4, 8, 16]

    h_results = run_h_convergence(p_values, n_values)

    # p-convergence study
    print("\n" + "="*70)
    print("P-CONVERGENCE STUDY")
    print("="*70)

    n_fixed = [2, 4, 8]
    p_range = [2, 3, 4, 5, 6, 7, 8]

    p_results = run_p_convergence(n_fixed, p_range)

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    plot_h_convergence(h_results)
    plot_p_convergence(p_results)
    plot_error_vs_dof(h_results)

    # Print LaTeX table
    create_convergence_table(h_results)

    print("\n" + "="*70)
    print("STUDY COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
