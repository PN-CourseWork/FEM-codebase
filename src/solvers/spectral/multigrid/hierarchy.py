"""Multigrid hierarchy construction and FSG solve driver.

Implements Full Single Grid (FSG) multigrid:
- Grid hierarchy with nested Gauss-Lobatto grids
- Sequential solve from coarse to fine
"""

import logging
from typing import List, Tuple, Optional

import numpy as np

from solvers.spectral.core.level import SpectralLevel, build_spectral_level
from solvers.spectral.core.timestepping import MultigridSmoother
from solvers.spectral.operators.transfer_operators import (
    TransferOperators,
    create_transfer_operators,
)
from solvers.spectral.operators.corner import (
    CornerTreatment,
    create_corner_treatment,
)
from .transfers import prolongate_solution

log = logging.getLogger(__name__)


def build_hierarchy(
    n_fine: int,
    n_levels: int,
    basis_x,
    basis_y,
    Lx: float = 1.0,
    Ly: float = 1.0,
    coarsest_n: int = 12,
) -> List[SpectralLevel]:
    """Build multigrid hierarchy from fine to coarse.

    Parameters
    ----------
    n_fine : int
        Polynomial order on finest grid
    n_levels : int
        Maximum number of multigrid levels (may use fewer if coarsest_n limit reached)
    basis_x, basis_y : Basis objects
        Spectral basis objects
    Lx, Ly : float
        Domain dimensions
    coarsest_n : int
        Minimum polynomial order for coarsest grid (default 12).
        Coarse grids need sufficient resolution to capture physics.

    Returns
    -------
    List[SpectralLevel]
        List of levels, index 0 = coarsest, index -1 = finest
    """
    # Compute polynomial orders for each level (full coarsening: N/2)
    # Stop when next coarsening would go below coarsest_n
    orders = []
    n = n_fine
    for _ in range(n_levels):
        orders.append(n)
        n_next = n // 2
        if n_next < coarsest_n:
            break
        n = n_next

    # Reverse so coarsest is first
    orders = orders[::-1]

    log.info(f"Building {len(orders)}-level hierarchy: N = {orders}")

    levels = []
    for idx, n in enumerate(orders):
        level = build_spectral_level(n, idx, basis_x, basis_y, Lx, Ly)
        levels.append(level)

    return levels


def solve_fsg(
    levels: List[SpectralLevel],
    Re: float,
    beta_squared: float,
    lid_velocity: float,
    CFL: float,
    tolerance: float,
    max_iterations: int,
    transfer_ops: Optional[TransferOperators] = None,
    corner_treatment: Optional[CornerTreatment] = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    coarse_tolerance_factor: float = 1.0,
) -> Tuple[SpectralLevel, int, bool]:
    """Solve using Full Single Grid (FSG) multigrid.

    Solves sequentially from coarsest to finest level, using the converged
    solution on each level as initial guess for the next finer level.

    Per Zhang & Xi (2010): Uses the SAME tolerance on ALL levels.

    Parameters
    ----------
    levels : List[SpectralLevel]
        Grid hierarchy (index 0 = coarsest)
    Re, beta_squared, lid_velocity, CFL : float
        Solver parameters
    tolerance : float
        Global convergence tolerance (used on ALL levels)
    max_iterations : int
        Max iterations per level
    transfer_ops : TransferOperators, optional
        Configured transfer operators. If None, uses default FFT operators.
    corner_treatment : CornerTreatment, optional
        Corner treatment handler. If None, uses default smoothing.
    Lx, Ly : float
        Domain dimensions
    coarse_tolerance_factor : float
        Factor to loosen tolerance on coarser levels

    Returns
    -------
    tuple
        (finest_level, total_iterations, converged)
    """
    # Create default transfer operators if not provided
    if transfer_ops is None:
        transfer_ops = create_transfer_operators(
            prolongation_method="fft",
            restriction_method="fft",
        )

    # Create default corner treatment if not provided
    if corner_treatment is None:
        corner_treatment = create_corner_treatment(method="smoothing")

    # Check if using subtraction method (needs fallback to smoothing on coarse grids)
    uses_subtraction = corner_treatment.uses_modified_convection()
    if uses_subtraction:
        smoothing_treatment = create_corner_treatment(method="smoothing")
        # Minimum N for subtraction method - use smoothing below this
        min_n_for_subtraction = 8

    total_iterations = 0
    n_levels = len(levels)

    for level_idx, level in enumerate(levels):
        is_finest = level_idx == n_levels - 1

        # Use LOOSER tolerance on coarser levels for efficiency
        # coarse_tolerance_factor=10 means: coarsest gets 100x looser (for 3 levels)
        levels_from_finest = n_levels - 1 - level_idx
        level_tol = tolerance * (coarse_tolerance_factor**levels_from_finest)

        log.info(
            f"FSG Level {level_idx}/{n_levels - 1}: N={level.n}, "
            f"tolerance={level_tol:.2e}"
        )

        # Select corner treatment for this level
        # For subtraction: use smoothing on coarse levels for stability
        if uses_subtraction and level.n < min_n_for_subtraction:
            level_corner_treatment = smoothing_treatment
            log.debug(
                f"  Level {level_idx} (N={level.n}): using smoothing (N < {min_n_for_subtraction})"
            )
        else:
            level_corner_treatment = corner_treatment

        # Initialize from previous level or zeros
        if level_idx == 0:
            # Coarsest level: start from zeros
            level.u[:] = 0.0
            level.v[:] = 0.0
            level.p[:] = 0.0
        else:
            # Prolongate from previous (coarser) level
            prolongate_solution(levels[level_idx - 1], level, transfer_ops, lid_velocity)

        # Create smoother for this level
        smoother = MultigridSmoother(
            level=level,
            Re=Re,
            beta_squared=beta_squared,
            lid_velocity=lid_velocity,
            CFL=CFL,
            corner_treatment=level_corner_treatment,
            Lx=Lx,
            Ly=Ly,
        )
        smoother.initialize_lid()

        # Solve on this level
        converged = False
        level_iters = 0

        diverged = False
        for iteration in range(max_iterations):
            u_res, v_res = smoother.step()
            level_iters += 1
            total_iterations += 1

            # Check convergence
            max_res = max(u_res, v_res)
            if max_res < level_tol:
                converged = True
                cont_res = smoother.get_continuity_residual()
                log.info(
                    f"  Level {level_idx} converged in {level_iters} iterations, "
                    f"residual={max_res:.2e}, continuity={cont_res:.2e}"
                )
                break

            # Early exit on NaN/Inf (diverged)
            if not np.isfinite(max_res):
                diverged = True
                log.warning(
                    f"  Level {level_idx} diverged (NaN/Inf) at iteration {level_iters}, exiting early"
                )
                break

            # Logging every 100 iterations
            if iteration > 0 and iteration % 100 == 0:
                cont_res = smoother.get_continuity_residual()
                log.debug(
                    f"  Level {level_idx} iter {iteration}: "
                    f"u_res={u_res:.2e}, v_res={v_res:.2e}, cont={cont_res:.2e}"
                )

        # Exit early if diverged (NaN/Inf detected)
        if diverged:
            break

        if not converged and not is_finest:
            log.warning(
                f"  Level {level_idx} did not converge after {level_iters} iterations, "
                f"continuing to next level..."
            )
        elif not converged and is_finest:
            log.warning(
                f"  Finest level did not converge after {level_iters} iterations"
            )

    finest_level = levels[-1]
    final_converged = converged and not diverged

    log.info(
        f"FSG completed: {total_iterations} total iterations, converged={final_converged}"
    )

    return finest_level, total_iterations, final_converged
