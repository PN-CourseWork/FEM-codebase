"""Transfer operators for multigrid (prolongation and restriction).

Handles coarse-to-fine (prolongation) and fine-to-coarse (restriction)
solution and residual transfers between multigrid levels.
"""

import logging

import numpy as np

from solvers.spectral.core.level import SpectralLevel
from solvers.spectral.operators.transfer_operators import (
    TransferOperators,
    InjectionRestriction,
)

log = logging.getLogger(__name__)


def prolongate_solution(
    level_coarse: SpectralLevel,
    level_fine: SpectralLevel,
    transfer_ops: TransferOperators,
    lid_velocity: float = 1.0,
) -> None:
    """Prolongate solution (u, v, p) from coarse level to fine level.

    Modifies level_fine.u, level_fine.v, level_fine.p in place.

    IMPORTANT: After spectral interpolation, boundary conditions are
    re-enforced explicitly to avoid Gibbs-type oscillations at boundaries.

    Parameters
    ----------
    level_coarse : SpectralLevel
        Source (coarse) level with converged solution
    level_fine : SpectralLevel
        Target (fine) level to receive interpolated solution
    transfer_ops : TransferOperators
        Configured transfer operators for prolongation
    lid_velocity : float
        Lid velocity for boundary condition (default: 1.0)
    """
    # Prolongate velocities (full grid)
    u_coarse_2d = level_coarse.u.reshape(level_coarse.shape_full)
    v_coarse_2d = level_coarse.v.reshape(level_coarse.shape_full)

    u_fine_2d = transfer_ops.prolongation.prolongate_2d(
        u_coarse_2d, level_fine.shape_full
    )
    v_fine_2d = transfer_ops.prolongation.prolongate_2d(
        v_coarse_2d, level_fine.shape_full
    )

    # Re-enforce boundary conditions after interpolation
    # (spectral interpolation can introduce Gibbs oscillations at boundaries)
    # Bottom: u=0, v=0
    u_fine_2d[0, :] = 0.0
    v_fine_2d[0, :] = 0.0
    # Top: u=lid_velocity, v=0
    u_fine_2d[-1, :] = lid_velocity
    v_fine_2d[-1, :] = 0.0
    # Left: u=0, v=0
    u_fine_2d[:, 0] = 0.0
    v_fine_2d[:, 0] = 0.0
    # Right: u=0, v=0
    u_fine_2d[:, -1] = 0.0
    v_fine_2d[:, -1] = 0.0

    level_fine.u[:] = u_fine_2d.ravel()
    level_fine.v[:] = v_fine_2d.ravel()

    # Prolongate pressure (inner grid - no boundary conditions needed)
    p_coarse_2d = level_coarse.p.reshape(level_coarse.shape_inner)
    p_fine_2d = transfer_ops.prolongation.prolongate_2d(
        p_coarse_2d, level_fine.shape_inner
    )
    level_fine.p[:] = p_fine_2d.ravel()

    log.debug(
        f"Prolongated solution from level {level_coarse.level_idx} "
        f"(N={level_coarse.n}) to level {level_fine.level_idx} (N={level_fine.n})"
    )


def restrict_solution(
    level_fine: SpectralLevel,
    level_coarse: SpectralLevel,
    transfer_ops: TransferOperators,
) -> None:
    """Restrict solution (u, v, p) from fine level to coarse level.

    Uses direct injection for variables (FAS scheme requirement).
    This is critical: coarse GLL points are subsets of fine GLL points,
    so injection preserves the exact solution values.

    Parameters
    ----------
    level_fine : SpectralLevel
        Source (fine) level
    level_coarse : SpectralLevel
        Target (coarse) level
    transfer_ops : TransferOperators
        Configured transfer operators (not used - always uses injection)
    """
    # FAS requires direct injection for solution restriction
    # (coarse GLL points are subsets of fine GLL points)
    injection = InjectionRestriction()

    # Restrict velocities (full grid)
    u_fine_2d = level_fine.u.reshape(level_fine.shape_full)
    v_fine_2d = level_fine.v.reshape(level_fine.shape_full)

    u_coarse_2d = injection.restrict_2d(u_fine_2d, level_coarse.shape_full)
    v_coarse_2d = injection.restrict_2d(v_fine_2d, level_coarse.shape_full)

    level_coarse.u[:] = u_coarse_2d.ravel()
    level_coarse.v[:] = v_coarse_2d.ravel()

    # Restrict pressure (inner grid)
    p_fine_2d = level_fine.p.reshape(level_fine.shape_inner)
    p_coarse_2d = injection.restrict_2d(p_fine_2d, level_coarse.shape_inner)
    level_coarse.p[:] = p_coarse_2d.ravel()

    log.debug(
        f"Restricted solution from level {level_fine.level_idx} "
        f"(N={level_fine.n}) to level {level_coarse.level_idx} (N={level_coarse.n})"
    )


def restrict_residual(
    level_fine: SpectralLevel,
    level_coarse: SpectralLevel,
    transfer_ops: TransferOperators,
) -> None:
    """Restrict residuals (R_u, R_v, R_p) from fine to coarse level.

    Uses FFT-based restriction for residuals (spectral truncation).

    Per Zhang & Xi (2010), Section 3.3:
    "In the PN − PN−2 method, the boundary values are already known for
    velocities and unnecessary for pressure, so the residuals and corrections
    on the boundary points are all set to zero."

    Parameters
    ----------
    level_fine : SpectralLevel
        Source (fine) level with computed residuals
    level_coarse : SpectralLevel
        Target (coarse) level to receive restricted residuals
    transfer_ops : TransferOperators
        Configured transfer operators
    """
    # Restrict momentum residuals (full grid)
    # IMPORTANT: Zero FINE grid boundaries BEFORE restriction!
    # The residuals at boundary nodes are garbage (BCs are enforced separately).
    # If we don't zero them before FFT restriction, they pollute interior values
    # through spectral truncation.
    R_u_fine_2d = level_fine.R_u.reshape(level_fine.shape_full).copy()
    R_v_fine_2d = level_fine.R_v.reshape(level_fine.shape_full).copy()

    # Zero fine grid boundaries BEFORE restriction
    R_u_fine_2d[0, :] = 0.0
    R_u_fine_2d[-1, :] = 0.0
    R_u_fine_2d[:, 0] = 0.0
    R_u_fine_2d[:, -1] = 0.0
    R_v_fine_2d[0, :] = 0.0
    R_v_fine_2d[-1, :] = 0.0
    R_v_fine_2d[:, 0] = 0.0
    R_v_fine_2d[:, -1] = 0.0

    R_u_coarse_2d = transfer_ops.restriction.restrict_2d(
        R_u_fine_2d, level_coarse.shape_full
    )
    R_v_coarse_2d = transfer_ops.restriction.restrict_2d(
        R_v_fine_2d, level_coarse.shape_full
    )

    # Also zero coarse grid boundaries after restriction (belt and suspenders)
    # "residuals and corrections on the boundary points are all set to zero"
    R_u_coarse_2d[0, :] = 0.0
    R_u_coarse_2d[-1, :] = 0.0
    R_u_coarse_2d[:, 0] = 0.0
    R_u_coarse_2d[:, -1] = 0.0
    R_v_coarse_2d[0, :] = 0.0
    R_v_coarse_2d[-1, :] = 0.0
    R_v_coarse_2d[:, 0] = 0.0
    R_v_coarse_2d[:, -1] = 0.0

    level_coarse.R_u[:] = R_u_coarse_2d.ravel()
    level_coarse.R_v[:] = R_v_coarse_2d.ravel()

    # Restrict continuity residual (inner grid - already excludes boundaries)
    R_p_fine_2d = level_fine.R_p.reshape(level_fine.shape_inner)
    R_p_coarse_2d = transfer_ops.restriction.restrict_2d(
        R_p_fine_2d, level_coarse.shape_inner
    )
    level_coarse.R_p[:] = R_p_coarse_2d.ravel()
