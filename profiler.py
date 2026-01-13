"""Profile FEM solver to identify performance bottlenecks.

Profiles the same problem as SusDriverW2.py (exercises 2.8b/c) and reports
cumulative time with percentage of total time for each function.
"""

import cProfile
import pstats
from io import StringIO

import numpy as np

from FEM.datastructures import Mesh2d
from FEM.solvers import solve_mixed_bc_2d


def _q_zero(x: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: ARG001
    """Zero Neumann boundary condition."""
    return np.zeros_like(x)


def run_solver(n_runs: int = 1, reuse_mesh: bool = True) -> None:
    """Run the FEM solver n_runs times for more accurate profiling.

    Args:
        n_runs: Number of iterations
        reuse_mesh: If True, create mesh once and reuse (realistic usage).
                   If False, create new mesh each iteration.
    """
    # Problem parameters (same as SusDriverW2.py)
    noelms1, noelms2 = 40, 50
    lam1, lam2 = 1.0, 1.0

    def fun(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.cos(np.pi * x) * np.cos(np.pi * y)

    def qt(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)

    if reuse_mesh:
        # Create meshes once (realistic: mesh created once, solve called many times)
        mesh_b = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=noelms1, noelms2=noelms2)
        mesh_c = Mesh2d(x0=-1, y0=-1, L1=2, L2=2, noelms1=noelms1, noelms2=noelms2)
        for _ in range(n_runs):
            solve_mixed_bc_2d(mesh_b, qt, _q_zero, _q_zero, fun, lam1, lam2)
            solve_mixed_bc_2d(mesh_c, qt, _q_zero, _q_zero, fun, lam1, lam2)
    else:
        # Create new mesh each iteration (for profiling mesh creation)
        for _ in range(n_runs):
            mesh_b = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=noelms1, noelms2=noelms2)
            solve_mixed_bc_2d(mesh_b, qt, _q_zero, _q_zero, fun, lam1, lam2)
            mesh_c = Mesh2d(x0=-1, y0=-1, L1=2, L2=2, noelms1=noelms1, noelms2=noelms2)
            solve_mixed_bc_2d(mesh_c, qt, _q_zero, _q_zero, fun, lam1, lam2)


def profile_solver(n_runs: int = 10) -> None:
    """Profile the solver and print results sorted by cumulative time."""
    print("=" * 70)
    print("FEM Solver Profiler")
    print("=" * 70)
    print("\nProblem: 40x50 mesh, Poisson equation with mixed BCs")
    print(f"Running {n_runs} iterations for statistical accuracy...\n")

    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()
    run_solver(n_runs)
    profiler.disable()

    # Capture stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    # Get total time (attributes exist but aren't in type stubs)
    total_time: float = stats.total_tt  # type: ignore[attr-defined]
    total_calls: int = stats.total_calls  # type: ignore[attr-defined]

    print(f"Total time: {total_time:.4f}s ({total_time/n_runs:.4f}s per iteration)")
    print(f"Total function calls: {total_calls:,}")
    print("\n" + "=" * 70)
    print("TOP 25 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)
    print(f"{'Function':<45} {'Calls':>8} {'CumTime':>10} {'%Total':>8}")
    print("-" * 70)

    # Extract and format top functions (stats dict: func -> (cc, nc, tt, ct, callers))
    func_list = []
    func_stats: dict = stats.stats  # type: ignore[attr-defined]
    for func, stat_tuple in func_stats.items():
        nc, ct = stat_tuple[1], stat_tuple[3]  # call count, cumulative time
        filename, _, name = func
        # Skip built-in methods that aren't informative
        if name.startswith("<") and name.endswith(">"):
            continue
        func_list.append((name, filename, nc, ct, ct / total_time * 100))

    # Sort by cumulative time
    func_list.sort(key=lambda x: x[3], reverse=True)

    for name, filename, calls, cum_time, pct in func_list[:25]:
        # Truncate long names
        display_name = f"{name}"
        if len(display_name) > 44:
            display_name = display_name[:41] + "..."
        print(f"{display_name:<45} {calls:>8} {cum_time:>10.4f} {pct:>7.1f}%")

    # Print FEM-specific breakdown
    print("\n" + "=" * 70)
    print("FEM MODULE BREAKDOWN")
    print("=" * 70)

    fem_funcs = [
        ("Mesh2d.__post_init__", "_compute"),
        ("assembly_2d", "assembly"),
        ("solve_mixed_bc_2d", "solve"),
        ("dirbc_2d", "dirbc"),
        ("neubc_2d", "neubc"),
        ("spsolve", "spsolve"),
    ]

    print(f"{'Component':<30} {'CumTime':>10} {'%Total':>8} {'Per Iter':>12}")
    print("-" * 70)

    for search_name, display in fem_funcs:
        for name, filename, calls, cum_time, pct in func_list:
            if search_name in name or name == search_name:
                per_iter = cum_time / n_runs
                print(f"{display:<30} {cum_time:>10.4f} {pct:>7.1f}% {per_iter:>10.6f}s")
                break

    # Print detailed stats to file
    print("\n" + "=" * 70)
    print("DETAILED STATS")
    print("=" * 70)
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    profile_solver(n_runs=10)
