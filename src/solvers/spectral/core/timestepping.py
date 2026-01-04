"""MultigridSmoother class for RK4 time-stepping on a single level.

Encapsulates the time-stepping logic for one multigrid level.
Supports FAS scheme by allowing external forcing terms (tau correction).
"""

from typing import Tuple

import numpy as np

from .level import SpectralLevel
from .residuals import (
    compute_derivatives_and_laplacian,
    compute_momentum_residuals,
    compute_continuity_residual,
    interpolate_and_differentiate_pressure,
)
from solvers.spectral.operators.corner import CornerTreatment


class MultigridSmoother:
    """Performs RK4 smoothing iterations on a single level.

    Encapsulates the time-stepping logic for one multigrid level.
    Supports FAS scheme by allowing external forcing terms (tau correction).
    """

    def __init__(
        self,
        level: SpectralLevel,
        Re: float,
        beta_squared: float,
        lid_velocity: float,
        CFL: float,
        corner_treatment: CornerTreatment,
        Lx: float = 1.0,
        Ly: float = 1.0,
    ):
        self.level = level
        self.Re = Re
        self.nu = 1.0 / Re  # Cache viscosity
        self.beta_squared = beta_squared
        self.lid_velocity = lid_velocity
        self.CFL = CFL
        self.Lx = Lx
        self.Ly = Ly

        # FAS forcing terms (tau correction from fine grid)
        # These are ADDED to the computed residuals during coarse grid solve
        self.tau_u = None
        self.tau_v = None
        self.tau_p = None

        self.corner_treatment = corner_treatment

        # ========== OPTIMIZATION: Cache boundary conditions ==========
        # Pre-compute and cache lid velocity (it never changes during solve)
        x_lid = level.X[:, -1]
        y_lid = level.Y[:, -1]
        self._cached_u_lid, self._cached_v_lid = corner_treatment.get_lid_velocity(
            x_lid, y_lid, lid_velocity=lid_velocity, Lx=Lx, Ly=Ly
        )

        # Pre-compute wall velocities (zeros)
        # West boundary
        self._cached_u_west, self._cached_v_west = corner_treatment.get_wall_velocity(
            level.X[0, :], level.Y[0, :], Lx, Ly
        )
        # East boundary
        self._cached_u_east, self._cached_v_east = corner_treatment.get_wall_velocity(
            level.X[-1, :], level.Y[-1, :], Lx, Ly
        )
        # South boundary
        self._cached_u_south, self._cached_v_south = corner_treatment.get_wall_velocity(
            level.X[:, 0], level.Y[:, 0], Lx, Ly
        )

        # ========== OPTIMIZATION: Cache transposed matrices ==========
        self._Dy_1d_T = level.Dy_1d.T.copy()
        self._Dyy_1d_T = level.Dyy_1d.T.copy()
        self._Interp_y_T = level.Interp_y.T.copy()

    def _apply_lid_boundary(self, u_2d: np.ndarray, v_2d: np.ndarray):
        """Apply lid boundary condition using cached values."""
        u_2d[:, -1] = self._cached_u_lid
        v_2d[:, -1] = self._cached_v_lid

    def _interpolate_pressure_gradient(self, p: np.ndarray):
        """Compute pressure gradient on full grid from inner-grid pressure.

        Uses spectral interpolation (Chebyshev polynomial fit) to extend
        pressure from inner grid to full grid before differentiation.
        """
        lvl = self.level

        # JIT-compiled combined interpolation and gradient computation
        interpolate_and_differentiate_pressure(
            p,
            lvl.Interp_x,
            self._Interp_y_T,
            lvl.Dx_1d,
            self._Dy_1d_T,
            lvl.dp_dx,
            lvl.dp_dy,
            lvl.shape_inner,
        )

    def _compute_residuals(self, u: np.ndarray, v: np.ndarray, p: np.ndarray):
        """Compute RHS residuals for RK4 pseudo time-stepping.

        Uses O(N³) tensor product operations instead of O(N⁴) Kronecker products.
        """
        lvl = self.level

        # Reshape to 2D for tensor product operations (views, no copy)
        u_2d = u.reshape(lvl.shape_full)
        v_2d = v.reshape(lvl.shape_full)

        # JIT-compiled: Compute all derivatives and Laplacians in one call
        du_dx_2d, dv_dy_2d = compute_derivatives_and_laplacian(
            u_2d,
            v_2d,
            lvl.Dx_1d,
            self._Dy_1d_T,
            lvl.Dxx_1d,
            self._Dyy_1d_T,
            lvl.du_dx,
            lvl.du_dy,
            lvl.dv_dx,
            lvl.dv_dy,
            lvl.lap_u,
            lvl.lap_v,
        )

        # Compute pressure gradient
        self._interpolate_pressure_gradient(p)

        # JIT-compiled: Compute momentum residuals
        compute_momentum_residuals(
            u,
            v,
            lvl.du_dx,
            lvl.du_dy,
            lvl.dv_dx,
            lvl.dv_dy,
            lvl.lap_u,
            lvl.lap_v,
            lvl.dp_dx,
            lvl.dp_dy,
            self.nu,
            lvl.R_u,
            lvl.R_v,
        )

        # JIT-compiled: Continuity residual on inner grid
        compute_continuity_residual(du_dx_2d, dv_dy_2d, self.beta_squared, lvl.R_p)

        # Add FAS tau correction if set (for coarse grid solves in V-cycle)
        if self.tau_u is not None:
            lvl.R_u[:] += self.tau_u
        if self.tau_v is not None:
            lvl.R_v[:] += self.tau_v
        if self.tau_p is not None:
            lvl.R_p[:] += self.tau_p

    def _enforce_boundary_conditions(self, u: np.ndarray, v: np.ndarray):
        """Enforce boundary conditions using cached values."""
        u_2d = u.reshape(self.level.shape_full)
        v_2d = v.reshape(self.level.shape_full)

        # Use pre-computed wall velocities
        # West boundary
        u_2d[0, :] = self._cached_u_west
        v_2d[0, :] = self._cached_v_west

        # East boundary
        u_2d[-1, :] = self._cached_u_east
        v_2d[-1, :] = self._cached_v_east

        # South boundary
        u_2d[:, 0] = self._cached_u_south
        v_2d[:, 0] = self._cached_v_south

        # North boundary (moving lid)
        u_2d[:, -1] = self._cached_u_lid
        v_2d[:, -1] = self._cached_v_lid

    def _compute_adaptive_timestep(self) -> float:
        """Compute adaptive timestep based on CFL."""
        lvl = self.level
        u_max = max(np.max(np.abs(lvl.u)), abs(self.lid_velocity))
        v_max = max(np.max(np.abs(lvl.v)), 1e-10)

        lambda_x = (
            u_max + np.sqrt(u_max**2 + self.beta_squared)
        ) / lvl.dx_min + self.nu / lvl.dx_min**2
        lambda_y = (
            v_max + np.sqrt(v_max**2 + self.beta_squared)
        ) / lvl.dy_min + self.nu / lvl.dy_min**2

        return self.CFL / (lambda_x + lambda_y)

    def initialize_lid(self):
        """Initialize lid velocity boundary condition using corner treatment."""
        u_2d = self.level.u.reshape(self.level.shape_full)
        v_2d = self.level.v.reshape(self.level.shape_full)
        self._apply_lid_boundary(u_2d, v_2d)

    def step(self) -> Tuple[float, float]:
        """Perform one RK4 pseudo time-step.

        Returns
        -------
        tuple
            (u_residual, v_residual) - relative L2 norms of velocity change
        """
        lvl = self.level

        # Save previous for convergence check
        lvl.u_prev[:] = lvl.u
        lvl.v_prev[:] = lvl.v

        dt = self._compute_adaptive_timestep()

        # 4-stage RK4
        rk4_coeffs = [0.25, 1.0 / 3.0, 0.5, 1.0]
        u_in, v_in, p_in = lvl.u, lvl.v, lvl.p

        for i, alpha in enumerate(rk4_coeffs):
            self._compute_residuals(u_in, v_in, p_in)

            if i < 3:
                lvl.u_stage[:] = lvl.u + alpha * dt * lvl.R_u
                lvl.v_stage[:] = lvl.v + alpha * dt * lvl.R_v
                lvl.p_stage[:] = lvl.p + alpha * dt * lvl.R_p
                self._enforce_boundary_conditions(lvl.u_stage, lvl.v_stage)
                u_in, v_in, p_in = lvl.u_stage, lvl.v_stage, lvl.p_stage
            else:
                lvl.u[:] = lvl.u + alpha * dt * lvl.R_u
                lvl.v[:] = lvl.v + alpha * dt * lvl.R_v
                lvl.p[:] = lvl.p + alpha * dt * lvl.R_p
                self._enforce_boundary_conditions(lvl.u, lvl.v)

        # Compute RELATIVE residuals for convergence check
        u_res = np.linalg.norm(lvl.u - lvl.u_prev) / (np.linalg.norm(lvl.u_prev) + 1e-12)
        v_res = np.linalg.norm(lvl.v - lvl.v_prev) / (np.linalg.norm(lvl.v_prev) + 1e-12)

        return u_res, v_res

    def smooth(self, n_steps: int) -> Tuple[float, float]:
        """Perform multiple RK4 smoothing steps.

        Parameters
        ----------
        n_steps : int
            Number of RK4 steps

        Returns
        -------
        tuple
            Final (u_residual, v_residual)
        """
        u_res, v_res = 0.0, 0.0
        for _ in range(n_steps):
            u_res, v_res = self.step()
        return u_res, v_res

    def get_continuity_residual(self) -> float:
        """Get L2 norm of continuity residual."""
        return np.linalg.norm(self.level.R_p)

    def set_tau_correction(
        self,
        tau_u: np.ndarray,
        tau_v: np.ndarray,
        tau_p: np.ndarray,
    ):
        """Set FAS tau correction terms for coarse grid solve.

        These terms are added to the computed residuals during RK4 steps.
        Call clear_tau_correction() after the coarse grid solve is complete.

        Parameters
        ----------
        tau_u, tau_v : np.ndarray
            Momentum tau corrections (full grid size)
        tau_p : np.ndarray
            Pressure tau correction (inner grid size)
        """
        self.tau_u = tau_u
        self.tau_v = tau_v
        self.tau_p = tau_p

    def clear_tau_correction(self):
        """Clear FAS tau correction terms."""
        self.tau_u = None
        self.tau_v = None
        self.tau_p = None
