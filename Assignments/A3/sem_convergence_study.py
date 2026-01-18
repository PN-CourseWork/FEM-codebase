"""SEM Convergence Study using Method of Manufactured Solutions (MMS)."""

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

OUTPUT_DIR = Path(__file__).parent.parent.parent / "figures" / "A3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MMS: Manufactured solution with non-homogeneous Dirichlet BCs
#   u(x,y) = exp(x) * sin(πy) + x²y(1-y)
#
# Laplacian:
#   ∂²u/∂x² = exp(x)sin(πy) + 2y(1-y)
#   ∂²u/∂y² = -π²exp(x)sin(πy) + x²(-2)
#   -∇²u = (π²-1)exp(x)sin(πy) - 2y(1-y) + 2x²
# =============================================================================
def u_exact(x, y):
    return np.exp(x) * np.sin(np.pi * y) + x**2 * y * (1 - y)

def f_rhs(x, y):
    return (np.pi**2 - 1) * np.exp(x) * np.sin(np.pi * y) - 2*y*(1-y) + 2*x**2

def bc_func(x, y):
    return u_exact(x, y)

print("=" * 60)
print("SEM CONVERGENCE STUDY (MMS)")
print("=" * 60)
print("u(x,y) = exp(x)sin(πy) + x²y(1-y)")

# =============================================================================
# H-CONVERGENCE
# =============================================================================
print("\nH-CONVERGENCE")
print("-" * 60)

p_values = [2, 3, 4, 5, 6]
n_values = [2, 4, 8, 16]
h_results = {}

for p in p_values:
    print(f"\np = {p}")
    print(f"  {'n':<4} {'h':<8} {'DOF':<6} {'L2':<12} {'Linf':<12}")
    print(f"  {'-'*50}")

    h_results[p] = {'h': [], 'dof': [], 'l2': [], 'linf': []}

    for n in n_values:
        with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
            filepath = Path(f.name)

        create_unit_square_mesh(n, n, filepath)
        mesh = SEMMesh2D(filepath=filepath, polynomial_order=p)
        u = solve_poisson_sem(mesh, f_rhs, bc_func)

        l2_err = l2_error_sem(mesh, u, u_exact)
        linf_err = linf_error_sem(mesh, u, u_exact)
        h = 1.0 / n

        h_results[p]['h'].append(h)
        h_results[p]['dof'].append(mesh.nonodes)
        h_results[p]['l2'].append(l2_err)
        h_results[p]['linf'].append(linf_err)

        print(f"  {n:<4} {h:<8.4f} {mesh.nonodes:<6} {l2_err:<12.2e} {linf_err:<12.2e}")
        filepath.unlink()

# =============================================================================
# P-CONVERGENCE
# =============================================================================
print("\n" + "=" * 60)
print("P-CONVERGENCE")
print("-" * 60)

n_fixed = [2, 4, 8]
p_range = [2, 3, 4, 5, 6, 7, 8]
p_results = {}

for n in n_fixed:
    print(f"\n{n}x{n} mesh")
    print(f"  {'p':<4} {'DOF':<6} {'L2':<12} {'Linf':<12}")
    print(f"  {'-'*40}")

    p_results[n] = {'p': [], 'dof': [], 'l2': [], 'linf': []}

    for p in p_range:
        with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
            filepath = Path(f.name)

        create_unit_square_mesh(n, n, filepath)
        mesh = SEMMesh2D(filepath=filepath, polynomial_order=p)
        u = solve_poisson_sem(mesh, f_rhs, bc_func)

        l2_err = l2_error_sem(mesh, u, u_exact)
        linf_err = linf_error_sem(mesh, u, u_exact)

        p_results[n]['p'].append(p)
        p_results[n]['dof'].append(mesh.nonodes)
        p_results[n]['l2'].append(l2_err)
        p_results[n]['linf'].append(linf_err)

        print(f"  {p:<4} {mesh.nonodes:<6} {l2_err:<12.2e} {linf_err:<12.2e}")
        filepath.unlink()

# =============================================================================
# PLOTS
# =============================================================================
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("-" * 60)

cmap_h = plt.get_cmap('viridis')
cmap_p = plt.get_cmap('plasma')
markers = ['o', 's', '^', 'D', 'v']

# H-convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = [cmap_h(x) for x in np.linspace(0.2, 0.8, len(p_values))]

for idx, p in enumerate(p_values):
    h = np.array(h_results[p]['h'])
    l2 = np.array(h_results[p]['l2'])
    linf = np.array(h_results[p]['linf'])
    rate = np.polyfit(np.log(h), np.log(l2), 1)[0]

    ax1.loglog(h, l2, f'{markers[idx]}-', color=colors[idx],
               label=f'p={p} (rate={rate:.1f})', markersize=8)
    ax2.loglog(h, linf, f'{markers[idx]}-', color=colors[idx], label=f'p={p}', markersize=8)

h_ref = np.array([0.5, 0.05])
for slope, ls in [(2, ':'), (4, '--'), (6, '-.')]:
    ax1.loglog(h_ref, 0.5 * h_ref**slope, ls, color='gray', alpha=0.5, label=f'O(h^{slope})')

ax1.set(xlabel='h', ylabel='L² error', title='L² Error (h-convergence)')
ax2.set(xlabel='h', ylabel='L∞ error', title='L∞ Error (h-convergence)')
for ax in (ax1, ax2):
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "h_convergence.pdf")
plt.close(fig)
print(f"Saved: {OUTPUT_DIR / 'h_convergence.pdf'}")

# P-convergence plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = [cmap_p(x) for x in np.linspace(0.2, 0.8, len(n_fixed))]

for idx, n in enumerate(n_fixed):
    p_arr = np.array(p_results[n]['p'])
    l2 = np.array(p_results[n]['l2'])
    linf = np.array(p_results[n]['linf'])

    ax1.semilogy(p_arr, l2, f'{markers[idx]}-', color=colors[idx], label=f'{n}x{n}', markersize=8)
    ax2.semilogy(p_arr, linf, f'{markers[idx]}-', color=colors[idx], label=f'{n}x{n}', markersize=8)

ax1.set(xlabel='p', ylabel='L² error', title='L² Error (p-convergence)')
ax2.set(xlabel='p', ylabel='L∞ error', title='L∞ Error (p-convergence)')
for ax in (ax1, ax2):
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "p_convergence.pdf")
plt.close(fig)
print(f"Saved: {OUTPUT_DIR / 'p_convergence.pdf'}")

# Error vs DOF
fig, ax = plt.subplots(figsize=(8, 6))
colors = [cmap_h(x) for x in np.linspace(0.2, 0.8, len(p_values))]

for idx, p in enumerate(p_values):
    dof = np.array(h_results[p]['dof'])
    l2 = np.array(h_results[p]['l2'])
    ax.loglog(dof, l2, f'{markers[idx]}-', color=colors[idx], label=f'p={p}', markersize=8)

ax.set(xlabel='DOF', ylabel='L² error', title='Efficiency: Error vs DOF')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "error_vs_dof.pdf")
plt.close(fig)
print(f"Saved: {OUTPUT_DIR / 'error_vs_dof.pdf'}")

print("\nDONE")
