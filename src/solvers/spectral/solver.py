"""Unified Spectral solver for lid-driven cavity.

This module provides a single SpectralSolver class that handles both
single-grid and multigrid (FSG) solving:
- n_levels=1: Single-grid solve (no multigrid overhead)
- n_levels>1: FSG multigrid with coarse-to-fine acceleration

Features:
- Velocities on full (Nx+1)×(Ny+1) Legendre/Chebyshev-Gauss-Lobatto grid
- Pressure on reduced (Nx-1)×(Ny-1) inner grid (PN-PN-2 method)
- Artificial compressibility for pressure-velocity coupling
- 4-stage RK4 explicit time stepping with adaptive CFL
- Corner singularity treatment options
"""

import logging
import time

import numpy as np

from ..base import LidDrivenCavitySolver
from ..datastructures import SpectralParameters
from .basis.spectral import LegendreLobattoBasis, ChebyshevLobattoBasis
from .operators.corner import create_corner_treatment
from .operators.transfer_operators import create_transfer_operators
from .core.level import SpectralLevel, build_spectral_level
from .core.timestepping import MultigridSmoother
from .multigrid.hierarchy import build_hierarchy, solve_fsg

log = logging.getLogger(__name__)


class SpectralSolver(LidDrivenCavitySolver):
    """Unified Spectral solver for lid-driven cavity problem.

    Supports both single-grid (n_levels=1) and multigrid (n_levels>1) modes.

    Parameters
    ----------
    params : SpectralParameters
        Parameters with physics (Re, lid velocity, domain size) and
        spectral-specific settings (Nx, Ny, CFL, beta_squared, n_levels, etc.).
    """

    Parameters = SpectralParameters

    def __init__(self, **kwargs):
        """Initialize spectral solver."""
        super().__init__(**kwargs)

        # Create spectral basis based on params
        if self.params.basis_type.lower() == "chebyshev":
            self.basis_x = ChebyshevLobattoBasis(domain=(0.0, self.params.Lx))
            self.basis_y = ChebyshevLobattoBasis(domain=(0.0, self.params.Ly))
            log.info("Using Chebyshev-Gauss-Lobatto basis")
        elif self.params.basis_type.lower() == "legendre":
            self.basis_x = LegendreLobattoBasis(domain=(0.0, self.params.Lx))
            self.basis_y = LegendreLobattoBasis(domain=(0.0, self.params.Ly))
            log.info("Using Legendre-Gauss-Lobatto basis")
        else:
            raise ValueError(
                f"Unknown basis_type: {self.params.basis_type}. Use 'legendre' or 'chebyshev'"
            )

        # Build the finest level (used for both single-grid and multigrid)
        self.level = build_spectral_level(
            n=self.params.nx,
            level_idx=0,
            basis_x=self.basis_x,
            basis_y=self.basis_y,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
        )

        # Cache shapes for convenience
        self.shape_full = self.level.shape_full
        self.shape_inner = self.level.shape_inner

        # Initialize output fields (base class handles this)
        self._init_fields(x=self.level.X.ravel(), y=self.level.Y.ravel())

        # Create corner treatment handler
        self.corner_treatment = create_corner_treatment(
            method=self.params.corner_treatment,
            smoothing_width=self.params.corner_smoothing,
        )
        log.info(f"Using corner treatment: {self.params.corner_treatment}")

        # Setup quadrature weights for proper integration (energy, enstrophy, etc.)
        self._setup_quadrature_weights()

        # Log solver mode
        if self.params.n_levels == 1:
            log.info("Spectral solver initialized in single-grid mode")
        else:
            log.info(f"Spectral solver initialized with {self.params.n_levels} multigrid levels")

        # Create smoother for step() method (lazy initialization)
        self._smoother = None

    def step(self):
        """Perform one RK4 pseudo time-step.

        This method is required by the abstract base class but the spectral solver
        primarily uses solve() directly. This is provided for API compatibility.

        Returns
        -------
        u, v, p : np.ndarray
            Updated velocity and pressure fields
        """
        # Lazy initialize smoother on first call
        if self._smoother is None:
            self._smoother = MultigridSmoother(
                level=self.level,
                Re=self.params.Re,
                beta_squared=self.params.beta_squared,
                lid_velocity=self.params.lid_velocity,
                CFL=self.params.CFL,
                corner_treatment=self.corner_treatment,
                Lx=self.params.Lx,
                Ly=self.params.Ly,
            )
            self._smoother.initialize_lid()

        # Perform one step
        self._smoother.step()

        return self.level.u, self.level.v, self.level.p

    def _setup_quadrature_weights(self):
        """Setup 2D quadrature weight matrix for proper spectral integration."""
        Nx, Ny = self.params.nx + 1, self.params.ny + 1

        # Get 1D quadrature weights from basis
        self.w_x = self.basis_x.quadrature_weights(Nx)
        self.w_y = self.basis_y.quadrature_weights(Ny)

        # Create 2D weight matrix via outer product
        self.W_2d = np.outer(self.w_x, self.w_y)

    def solve(self, tolerance: float = None, max_iter: int = None):
        """Solve the lid-driven cavity problem.

        Automatically selects single-grid or multigrid based on n_levels parameter.

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance. If None, uses params.tolerance.
        max_iter : int, optional
            Maximum iterations. If None, uses params.max_iterations.
        """
        if tolerance is None:
            tolerance = self.params.tolerance
        if max_iter is None:
            max_iter = self.params.max_iterations

        time_start = time.time()

        if self.params.n_levels == 1:
            # Single-grid solve (no multigrid overhead)
            total_iters, converged, history = self._solve_single_grid(tolerance, max_iter)
        else:
            # Multigrid (FSG) solve
            total_iters, converged = self._solve_multigrid(tolerance, max_iter)
            # Multigrid doesn't return history yet, create minimal one
            history = {
                "rel_iter": [tolerance if converged else tolerance * 10],
                "u_eq": [np.linalg.norm(self.level.R_u)],
                "v_eq": [np.linalg.norm(self.level.R_v)],
                "continuity": [np.linalg.norm(self.level.R_p)],
                "energy": [self._compute_energy()],
                "enstrophy": [self._compute_enstrophy()],
                "palinstrophy": [self._compute_palinstrophy()],
            }

        wall_time = time.time() - time_start

        # Convert history dict to list of dicts for _store_results
        residual_history = [
            {
                "rel_iter": history["rel_iter"][i],
                "u_eq": history["u_eq"][i],
                "v_eq": history["v_eq"][i],
                "continuity": history["continuity"][i],
            }
            for i in range(len(history["rel_iter"]))
        ]

        self._store_results(
            residual_history=residual_history,
            final_iter_count=total_iters,
            is_converged=converged,
            wall_time=wall_time,
            energy_history=history["energy"],
            enstrophy_history=history["enstrophy"],
            palinstrophy_history=history["palinstrophy"],
            iteration_history=history.get("iteration"),
        )

        log.info(
            f"Solver completed in {wall_time:.2f}s: {total_iters} iterations, "
            f"converged={converged}"
        )

    def _solve_single_grid(self, tolerance: float, max_iter: int) -> tuple:
        """Solve using single-grid (no multigrid).

        Returns
        -------
        tuple
            (total_iterations, converged, history)
        """
        log.info("Using single-grid spectral solver")

        # Initialize solution to zero
        self.level.u[:] = 0.0
        self.level.v[:] = 0.0
        self.level.p[:] = 0.0

        # Create smoother
        smoother = MultigridSmoother(
            level=self.level,
            Re=self.params.Re,
            beta_squared=self.params.beta_squared,
            lid_velocity=self.params.lid_velocity,
            CFL=self.params.CFL,
            corner_treatment=self.corner_treatment,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
        )
        smoother.initialize_lid()

        # History tracking
        history = {
            "iteration": [],
            "rel_iter": [],
            "u_eq": [],
            "v_eq": [],
            "continuity": [],
            "energy": [],
            "enstrophy": [],
            "palinstrophy": [],
        }

        # Iterate to convergence
        converged = False
        # Log at least every 50 iterations, but aim for ~100 points total
        log_interval = max(1, min(50, max_iter // 100))
        for iteration in range(max_iter):
            u_res, v_res = smoother.step()
            max_res = max(u_res, v_res)

            # Record history at regular intervals
            if iteration % log_interval == 0 or iteration == max_iter - 1:
                cont_res = smoother.get_continuity_residual()
                history["iteration"].append(iteration)
                history["rel_iter"].append(max_res)
                history["u_eq"].append(np.linalg.norm(self.level.R_u))
                history["v_eq"].append(np.linalg.norm(self.level.R_v))
                history["continuity"].append(cont_res)
                history["energy"].append(self._compute_energy())
                history["enstrophy"].append(self._compute_enstrophy())
                history["palinstrophy"].append(self._compute_palinstrophy())

            if max_res < tolerance:
                converged = True
                cont_res = smoother.get_continuity_residual()
                log.info(
                    f"Converged in {iteration + 1} iterations, "
                    f"residual={max_res:.2e}, continuity={cont_res:.2e}"
                )
                break

            if not np.isfinite(max_res):
                log.warning(f"Diverged (NaN/Inf) at iteration {iteration + 1}")
                break

            if iteration > 0 and iteration % 100 == 0:
                cont_res = smoother.get_continuity_residual()
                log.debug(
                    f"Iter {iteration}: u_res={u_res:.2e}, v_res={v_res:.2e}, cont={cont_res:.2e}"
                )

        return iteration + 1, converged, history

    def _solve_multigrid(self, tolerance: float, max_iter: int) -> tuple:
        """Solve using FSG multigrid.

        Returns
        -------
        tuple
            (total_iterations, converged)
        """
        log.info(f"Using FSG multigrid with {self.params.n_levels} levels")
        log.info(
            f"Transfer operators: prolongation={self.params.prolongation_method}, "
            f"restriction={self.params.restriction_method}"
        )

        # Create transfer operators
        transfer_ops = create_transfer_operators(
            prolongation_method=self.params.prolongation_method,
            restriction_method=self.params.restriction_method,
        )

        # Build grid hierarchy
        levels = build_hierarchy(
            n_fine=self.params.nx,
            n_levels=self.params.n_levels,
            basis_x=self.basis_x,
            basis_y=self.basis_y,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
        )

        # Solve using FSG
        finest_level, total_iters, converged = solve_fsg(
            levels=levels,
            Re=self.params.Re,
            beta_squared=self.params.beta_squared,
            lid_velocity=self.params.lid_velocity,
            CFL=self.params.CFL,
            tolerance=tolerance,
            max_iterations=max_iter,
            transfer_ops=transfer_ops,
            corner_treatment=self.corner_treatment,
            Lx=self.params.Lx,
            Ly=self.params.Ly,
            coarse_tolerance_factor=self.params.coarse_tolerance_factor,
        )

        # Copy solution from finest level to our level
        self.level.u[:] = finest_level.u
        self.level.v[:] = finest_level.v
        self.level.p[:] = finest_level.p

        return total_iters, converged

    def _finalize_fields(self):
        """Copy final solution to output fields.

        Pressure lives on inner grid and needs extrapolation to full grid.
        """
        self.fields.u[:] = self.level.u
        self.fields.v[:] = self.level.v
        # Extrapolate pressure from inner to full grid
        p_full_2d = self._extrapolate_to_full_grid(
            self.level.p.reshape(self.shape_inner)
        )
        self.fields.p[:] = p_full_2d.ravel()

    def _extrapolate_to_full_grid(self, inner_2d):
        """Extrapolate field from inner grid (Nx-1, Ny-1) to full grid (Nx+1, Ny+1)."""
        full_2d = np.zeros(self.shape_full)

        # Copy interior values
        full_2d[1:-1, 1:-1] = inner_2d

        # Extrapolate to boundaries (linear extrapolation)
        full_2d[0, 1:-1] = 2 * full_2d[1, 1:-1] - full_2d[2, 1:-1]
        full_2d[-1, 1:-1] = 2 * full_2d[-2, 1:-1] - full_2d[-3, 1:-1]
        full_2d[1:-1, 0] = 2 * full_2d[1:-1, 1] - full_2d[1:-1, 2]
        full_2d[1:-1, -1] = 2 * full_2d[1:-1, -2] - full_2d[1:-1, -3]

        # Corners (average of neighbors)
        full_2d[0, 0] = 0.5 * (full_2d[0, 1] + full_2d[1, 0])
        full_2d[0, -1] = 0.5 * (full_2d[0, -2] + full_2d[1, -1])
        full_2d[-1, 0] = 0.5 * (full_2d[-1, 1] + full_2d[-2, 0])
        full_2d[-1, -1] = 0.5 * (full_2d[-1, -2] + full_2d[-2, -1])

        return full_2d

    # =========================================================================
    # Quadrature-Based Integration
    # =========================================================================

    def _compute_energy(self) -> float:
        """Compute kinetic energy using spectral quadrature."""
        u_2d = self.level.u.reshape(self.shape_full)
        v_2d = self.level.v.reshape(self.shape_full)
        integrand = u_2d * u_2d + v_2d * v_2d
        return 0.5 * float(np.sum(self.W_2d * integrand))

    def _compute_vorticity(self) -> np.ndarray:
        """Compute vorticity ω = ∂v/∂x - ∂u/∂y using spectral differentiation."""
        u_2d = self.level.u.reshape(self.shape_full)
        v_2d = self.level.v.reshape(self.shape_full)

        # Spectral differentiation (tensor product form)
        dv_dx_2d = self.level.Dx_1d @ v_2d
        du_dy_2d = u_2d @ self.level.Dy_1d.T

        return (dv_dx_2d - du_dy_2d).ravel()

    def _compute_enstrophy(self) -> float:
        """Compute enstrophy Z = 0.5 * ∫∫ ω² dA."""
        omega_2d = self._compute_vorticity().reshape(self.shape_full)
        return 0.5 * float(np.sum(self.W_2d * omega_2d * omega_2d))

    def _compute_palinstrophy(self) -> float:
        """Compute palinstrophy P = 0.5 * ∫∫ ||∇ω||² dA."""
        omega_2d = self._compute_vorticity().reshape(self.shape_full)

        # Spectral differentiation of vorticity
        domega_dx_2d = self.level.Dx_1d @ omega_2d
        domega_dy_2d = omega_2d @ self.level.Dy_1d.T

        grad_omega_sq = domega_dx_2d**2 + domega_dy_2d**2
        return 0.5 * float(np.sum(self.W_2d * grad_omega_sq))

    def _compute_algebraic_residuals(self):
        """Return algebraic residuals from pseudo time-stepping."""
        return {
            "u_residual": np.linalg.norm(self.level.R_u),
            "v_residual": np.linalg.norm(self.level.R_v),
            "continuity_residual": np.linalg.norm(self.level.R_p),
        }

    # =========================================================================
    # Streamfunction and Vortex Detection
    # =========================================================================

    def _compute_streamfunction(self) -> tuple:
        """Compute streamfunction ψ by solving ∇²ψ = -ω.

        Uses spectral Laplacian with sparse linear solver.
        Homogeneous Dirichlet BCs (ψ=0 on walls).

        Returns
        -------
        psi_2d, x_2d, y_2d : np.ndarray
            Streamfunction and coordinate grids
        """
        from scipy.sparse import kron, eye, csr_matrix
        from scipy.sparse.linalg import spsolve

        # Get vorticity using spectral differentiation
        omega = self._compute_vorticity()
        omega_2d = omega.reshape(self.shape_full)

        Nx, Ny = self.shape_full

        # Build 2D Laplacian using Kronecker products
        Dxx_1d = self.level.Dx_1d @ self.level.Dx_1d
        Dyy_1d = self.level.Dy_1d @ self.level.Dy_1d

        Dxx_sparse = csr_matrix(Dxx_1d)
        Dyy_sparse = csr_matrix(Dyy_1d)
        Ix = eye(Nx, format="csr")
        Iy = eye(Ny, format="csr")

        L = kron(Dxx_sparse, Iy) + kron(Ix, Dyy_sparse)

        # Right-hand side: -omega (flattened)
        rhs = -omega_2d.ravel()

        # Apply homogeneous Dirichlet BCs
        boundary_mask = np.zeros((Nx, Ny), dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        boundary_idx = np.where(boundary_mask.ravel())[0]

        L_bc = L.tolil()
        for idx in boundary_idx:
            L_bc[idx, :] = 0
            L_bc[idx, idx] = 1.0
            rhs[idx] = 0.0
        L_bc = L_bc.tocsr()

        # Solve
        psi_flat = spsolve(L_bc, rhs)
        psi_2d = psi_flat.reshape((Nx, Ny))

        return psi_2d, self.level.X, self.level.Y

    def _find_primary_vortex(self) -> dict:
        """Find the primary vortex (global minimum of streamfunction)."""
        psi_2d, x_2d, y_2d = self._compute_streamfunction()

        min_idx = np.unravel_index(np.argmin(psi_2d), psi_2d.shape)
        psi_min = psi_2d[min_idx]
        x_min = x_2d[min_idx]
        y_min = y_2d[min_idx]

        omega_2d = self._compute_vorticity().reshape(self.shape_full)
        omega_center = omega_2d[min_idx]

        return {
            "psi_min": float(psi_min),
            "x": float(x_min),
            "y": float(y_min),
            "omega_center": float(omega_center),
        }

    def _find_corner_vortices(self) -> dict:
        """Find secondary corner vortices (BR, BL, TL)."""
        psi_2d, x_2d, y_2d = self._compute_streamfunction()
        omega_2d = self._compute_vorticity().reshape(self.shape_full)

        results = {}

        regions = {
            "BR": (x_2d > 0.5) & (y_2d < 0.5),
            "BL": (x_2d < 0.5) & (y_2d < 0.5),
            "TL": (x_2d < 0.5) & (y_2d > 0.5),
        }

        for name, mask in regions.items():
            psi_region = np.where(mask, psi_2d, -np.inf)
            max_idx = np.unravel_index(np.argmax(psi_region), psi_2d.shape)
            psi_val = psi_2d[max_idx]

            if psi_val > 0:
                results[name] = {
                    "psi": float(psi_val),
                    "omega": float(omega_2d[max_idx]),
                    "x": float(x_2d[max_idx]),
                    "y": float(y_2d[max_idx]),
                }
            else:
                results[name] = {"psi": 0.0, "omega": 0.0, "x": 0.0, "y": 0.0}

        return results

    def _find_max_vorticity(self) -> dict:
        """Find maximum vorticity and its location."""
        omega_2d = self._compute_vorticity().reshape(self.shape_full)
        max_abs_idx = np.unravel_index(np.argmax(np.abs(omega_2d)), omega_2d.shape)
        omega_max = omega_2d[max_abs_idx]

        return {
            "omega_max": float(omega_max),
            "x": float(self.level.X[max_abs_idx]),
            "y": float(self.level.Y[max_abs_idx]),
        }

    def compute_vortex_metrics(self) -> dict:
        """Compute all vortex-related metrics for validation."""
        primary = self._find_primary_vortex()
        corners = self._find_corner_vortices()
        max_omega = self._find_max_vorticity()

        return {
            "psi_min": primary["psi_min"],
            "psi_min_x": primary["x"],
            "psi_min_y": primary["y"],
            "omega_center": primary["omega_center"],
            "omega_max": max_omega["omega_max"],
            "omega_max_x": max_omega["x"],
            "omega_max_y": max_omega["y"],
            "psi_BR": corners["BR"]["psi"],
            "omega_BR": corners["BR"]["omega"],
            "psi_BR_x": corners["BR"]["x"],
            "psi_BR_y": corners["BR"]["y"],
            "psi_BL": corners["BL"]["psi"],
            "omega_BL": corners["BL"]["omega"],
            "psi_BL_x": corners["BL"]["x"],
            "psi_BL_y": corners["BL"]["y"],
            "psi_TL": corners["TL"]["psi"],
            "omega_TL": corners["TL"]["omega"],
            "psi_TL_x": corners["TL"]["x"],
            "psi_TL_y": corners["TL"]["y"],
        }
