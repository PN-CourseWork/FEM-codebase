"""
Navier-Stokes solver using stream function-vorticity formulation.

Solves the incompressible Navier-Stokes equations:
    ∂ω/∂t + u·∇ω = ν∇²ω    (vorticity transport)
    ∇²ψ = -ω                 (stream function Poisson)

where u = ∂ψ/∂y, v = -∂ψ/∂x

Uses IMEX time stepping:
- Implicit: diffusion (ν∇²ω)
- Explicit: convection (u·∇ω)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, Literal, Callable

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False

from .datastructures import Mesh2d, BOUNDARY_TOL
from .assembly import assembly_2d
from .boundary import dirbc_2d, get_boundary_nodes

if TYPE_CHECKING:
    from scipy.sparse.linalg import SuperLU


# =============================================================================
# Mass Matrix Assembly
# =============================================================================
def mass_matrix_2d(mesh: Mesh2d) -> csr_matrix:
    """
    Assemble the mass matrix M where M_ij = ∫ φ_i φ_j dΩ.
    """
    delta = np.abs(mesh.delta)

    # Local mass matrix entries (scaled by |Δ|/12)
    diag = 2 * delta / 12      # M_ii
    off_diag = delta / 12      # M_ij, i≠j

    # Pack entries: [M11, M12, M13, M21, M22, M23, M31, M32, M33]
    noelms = mesh.noelms
    n2 = 9
    element_data = np.empty(noelms * n2)
    element_data[0::n2] = diag      # M11
    element_data[1::n2] = off_diag  # M12
    element_data[2::n2] = off_diag  # M13
    element_data[3::n2] = off_diag  # M21
    element_data[4::n2] = diag      # M22
    element_data[5::n2] = off_diag  # M23
    element_data[6::n2] = off_diag  # M31
    element_data[7::n2] = off_diag  # M32
    element_data[8::n2] = diag      # M33

    # Assemble using pre-computed CSR pattern
    nnz = len(mesh._csr_indices)
    csr_data = np.zeros(nnz, dtype=np.float64)
    np.add.at(csr_data, mesh._csr_data_map, element_data)

    return csr_matrix(
        (csr_data, mesh._csr_indices, mesh._csr_indptr),
        shape=(mesh.nonodes, mesh.nonodes),
    )


# =============================================================================
# Advection Matrix Assembly (Vectorized)
# =============================================================================
def advection_matrix_2d(
    mesh: Mesh2d,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> csr_matrix:
    """
    Assemble the advection matrix C where C_ij = ∫ φ_i (u·∇φ_j) dΩ.
    """
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    x1, y1, x2, y2, x3, y3 = mesh.vertex_coords

    # Element-averaged velocities
    u_avg = (u[v1] + u[v2] + u[v3]) / 3
    v_avg = (v[v1] + v[v2] + v[v3]) / 3

    # Basis function gradients
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

    # Advection contributions
    adv1 = (u_avg * b1 + v_avg * c1) / 6
    adv2 = (u_avg * b2 + v_avg * c2) / 6
    adv3 = (u_avg * b3 + v_avg * c3) / 6

    # Pack entries
    noelms = mesh.noelms
    n2 = 9
    element_data = np.empty(noelms * n2)
    element_data[0::n2] = adv1
    element_data[1::n2] = adv2
    element_data[2::n2] = adv3
    element_data[3::n2] = adv1
    element_data[4::n2] = adv2
    element_data[5::n2] = adv3
    element_data[6::n2] = adv1
    element_data[7::n2] = adv2
    element_data[8::n2] = adv3

    nnz = len(mesh._csr_indices)
    csr_data = np.zeros(nnz, dtype=np.float64)
    np.add.at(csr_data, mesh._csr_data_map, element_data)

    return csr_matrix(
        (csr_data, mesh._csr_indices, mesh._csr_indptr),
        shape=(mesh.nonodes, mesh.nonodes),
    )


# =============================================================================
# FEM Interpolation
# =============================================================================
def interpolate_fem(
    mesh: Mesh2d,
    field: NDArray[np.float64],
    points: NDArray[np.float64],
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Interpolate FEM field at arbitrary points.
    """
    n_points = len(points)
    values = np.full(n_points, np.nan)

    # Get element vertex coordinates
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    x1, y1 = mesh.VX[v1], mesh.VY[v1]
    x2, y2 = mesh.VX[v2], mesh.VY[v2]
    x3, y3 = mesh.VX[v3], mesh.VY[v3]

    for i, (px, py) in enumerate(points):
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        lam1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
        lam2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
        lam3 = 1.0 - lam1 - lam2

        inside = (lam1 >= -tol) & (lam1 <= 1 + tol) & \
                 (lam2 >= -tol) & (lam2 <= 1 + tol) & \
                 (lam3 >= -tol) & (lam3 <= 1 + tol)

        elem_idx = np.where(inside)[0]
        if len(elem_idx) > 0:
            e = elem_idx[0]
            values[i] = lam1[e] * field[v1[e]] + lam2[e] * field[v2[e]] + lam3[e] * field[v3[e]]

    return values


def extract_centerline(
    mesh: Mesh2d,
    field: NDArray[np.float64],
    axis: str,
    position: float,
    n_points: int = 100,
    tol: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract field values along a centerline."""
    if axis == 'x':
        coords = np.linspace(mesh.y0, mesh.y0 + mesh.L2, n_points)
        points = np.column_stack([np.full(n_points, position), coords])
    else:
        coords = np.linspace(mesh.x0, mesh.x0 + mesh.L1, n_points)
        points = np.column_stack([coords, np.full(n_points, position)])

    values = interpolate_fem(mesh, field, points, tol=tol)
    return coords, values


# =============================================================================
# Velocity Recovery (Generic)
# =============================================================================
def compute_velocity(
    psi: NDArray[np.float64],
    mesh: Mesh2d,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute velocity from stream function: u = ∂ψ/∂y, v = -∂ψ/∂x.
    Uses generic element-averaged gradients interpolated to nodes.
    Does NOT enforce boundary conditions (slip/no-slip must be handled by solver if needed).
    """
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    delta = mesh.delta
    abs_delta = np.abs(delta)

    # Basis function coefficients from mesh.abc
    # abc[:, i, 1] corresponds to 'b' (coeff of y? No, see datastructures.py)
    # datastructures: abc[:, 0, 1] = y2-y3
    # standard P1: phi = (a + b*x + c*y)/(2A) -> grad = (b, c)/(2A)
    # navier_stokes original: b = y2-y3, c = x3-x2.
    # dpsi/dx = sum psi_i * b_i / 2A
    # dpsi/dy = sum psi_i * c_i / 2A
    
    b1 = mesh.abc[:, 0, 1]
    b2 = mesh.abc[:, 1, 1]
    b3 = mesh.abc[:, 2, 1]
    c1 = mesh.abc[:, 0, 2]
    c2 = mesh.abc[:, 1, 2]
    c3 = mesh.abc[:, 2, 2]

    inv_2delta = 1.0 / (2 * delta)
    
    # Gradients per element
    dpsi_dx = (psi[v1] * b1 + psi[v2] * b2 + psi[v3] * b3) * inv_2delta
    dpsi_dy = (psi[v1] * c1 + psi[v2] * c2 + psi[v3] * c3) * inv_2delta

    # u = ∂ψ/∂y, v = -∂ψ/∂x
    u_elem = dpsi_dy
    v_elem = -dpsi_dx

    # Area-weighted average to nodes
    u_weighted = u_elem * abs_delta
    v_weighted = v_elem * abs_delta

    u = np.zeros(mesh.nonodes)
    v = np.zeros(mesh.nonodes)
    weights = np.zeros(mesh.nonodes)

    np.add.at(u, v1, u_weighted)
    np.add.at(u, v2, u_weighted)
    np.add.at(u, v3, u_weighted)
    np.add.at(v, v1, v_weighted)
    np.add.at(v, v2, v_weighted)
    np.add.at(v, v3, v_weighted)
    np.add.at(weights, v1, abs_delta)
    np.add.at(weights, v2, abs_delta)
    np.add.at(weights, v3, abs_delta)

    # Avoid divide by zero
    mask = weights > 0
    u[mask] /= weights[mask]
    v[mask] /= weights[mask]
    
    return u, v


# =============================================================================
# Vorticity Boundary Handler (Generic)
# =============================================================================
class VorticityBoundaryHandler:
    """
    Handles vorticity boundary conditions for generic unstructured meshes.
    Uses neighbor finding to apply Thom's formula: ω_w = -2(ψ_int - ψ_w) / d².
    """

    def __init__(self, mesh: Mesh2d):
        self.mesh = mesh
        self.map = {}  # boundary_node_idx -> (interior_node_idx, dist_squared)
        
        # Identify boundary nodes (relies on correct boundary.py logic)
        self.bnodes = get_boundary_nodes(mesh)
        self.bnode_mask = np.zeros(mesh.nonodes, dtype=bool)
        self.bnode_mask[self.bnodes - 1] = True

        # Build adjacency graph
        adj = {i: set() for i in range(mesh.nonodes)}
        for i in range(mesh.noelms):
            nodes = mesh.EToV[i] - 1
            for u, v in [(0, 1), (1, 2), (2, 0)]:
                n1, n2 = nodes[u], nodes[v]
                adj[n1].add(n2)
                adj[n2].add(n1)

        # Find best interior neighbor for each boundary node
        for bnode in self.bnodes - 1:
            neighbors = list(adj[bnode])
            interior_neighbors = [n for n in neighbors if not self.bnode_mask[n]]
            
            if not interior_neighbors:
                # If no direct interior neighbor, this node might be a corner or isolated on boundary
                # Fallback: check all nodes in elements containing this node
                # For now, skip (usually rare in good meshes)
                continue

            dists = [
                (mesh.VX[n] - mesh.VX[bnode])**2 + (mesh.VY[n] - mesh.VY[bnode])**2
                for n in interior_neighbors
            ]
            best_idx = np.argmin(dists)
            self.map[bnode] = (interior_neighbors[best_idx], dists[best_idx])

    def apply(self, omega: NDArray[np.float64], psi: NDArray[np.float64], target_nodes: NDArray[np.int64] | None = None):
        """
        Update vorticity at boundary nodes using Thom's formula. 
        
        Parameters
        ----------
        omega : vorticity field (modified in-place)
        psi : stream function field
        target_nodes : optional subset of boundary nodes to update (1-based indices).
                       If None, updates all boundary nodes in the map.
        """
        nodes_to_update = target_nodes - 1 if target_nodes is not None else self.map.keys()
        
        for idx in nodes_to_update:
            if idx in self.map:
                int_node, d2 = self.map[idx]
                # Formula: omega_wall = -2 * (psi_int - psi_wall) / d^2
                omega[idx] = -2 * (psi[int_node] - psi[idx]) / d2


# =============================================================================
# Cached Poisson Solver
# =============================================================================
class StreamFunctionSolver:
    """Cached solver for stream function Poisson equation with Dirichlet BCs."""

    def __init__(self, mesh: Mesh2d, dirichlet_nodes: NDArray[np.int64], solver: Literal["lu", "amg"] = "lu", tol: float = 1e-10):
        # Assemble Poisson matrix (Laplacian)
        # Note: assemble returns -Laplacian ? No, usually K is stiffness.
        # assembly_2d returns A, b. A is stiffness.
        # ∇²ψ = -ω  =>  -K ψ = -M ω  => K ψ = M ω.
        # So we use K.
        
        self.K, _ = assembly_2d(mesh, 1.0, 1.0, np.zeros(mesh.nonodes))
        self.M = mass_matrix_2d(mesh)
        self.solver = solver
        self.tol = tol
        self._x_prev = np.zeros(mesh.nonodes)
        self.dirichlet_nodes = dirichlet_nodes

        # Apply structure for Dirichlet BCs (1 on diag, 0 elsewhere) to K
        # We perform this once. The RHS will need to be modified at each solve step.
        dummy_rhs = np.zeros(mesh.nonodes)
        self.A_fixed, _ = dirbc_2d(dirichlet_nodes, np.zeros(len(dirichlet_nodes)), self.K, dummy_rhs)

        if solver == "lu":
            self.lu: SuperLU = splu(self.A_fixed.tocsc())
        elif solver == "amg":
            if not HAS_PYAMG:
                raise ImportError("pyamg required for AMG solver.")
            self.ml = pyamg.smoothed_aggregation_solver(self.A_fixed)

    def solve(self, omega: NDArray[np.float64], psi_bc_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Solve ∇²ψ = -ω with Dirichlet BCs. 
        
        Parameters
        ----------
        omega : vorticity source
        psi_bc_values : array containing the fixed psi values at boundary nodes
        """
        # RHS = M @ omega
        rhs = self.M @ omega
        
        # Apply Dirichlet BCs to RHS
        # rhs[bnodes] = bc_values
        rhs[self.dirichlet_nodes - 1] = psi_bc_values[self.dirichlet_nodes - 1]

        if self.solver == "lu":
            return self.lu.solve(rhs)
        else:
            residuals: list[float] = []
            x = self.ml.solve(rhs, x0=self._x_prev, tol=self.tol, maxiter=200, residuals=residuals)
            self._x_prev = x
            return x


# =============================================================================
# IMEX Time Stepper (Generic)
# =============================================================================
class IMEXStepper:
    """IMEX time stepper with generic BC support."""

    def __init__(self, mesh: Mesh2d, dt: float, nu: float, omega_dirichlet_nodes: NDArray[np.int64], solver: Literal["lu", "amg"] = "lu", tol: float = 1e-10):
        self.mesh = mesh
        self.dt = dt
        self.nu = nu
        self.solver = solver
        self.tol = tol
        self.omega_dirichlet_nodes = omega_dirichlet_nodes

        # Matrices
        self.M = mass_matrix_2d(mesh)
        self.K, _ = assembly_2d(mesh, 1.0, 1.0, np.zeros(mesh.nonodes))

        # Implicit system: M + ν*dt*K
        self.A_implicit = (self.M + nu * dt * self.K).tocsr()
        
        # Pre-apply Dirichlet structure to implicit matrix
        dummy_rhs = np.zeros(mesh.nonodes)
        self.A_implicit_fixed, _ = dirbc_2d(omega_dirichlet_nodes, np.zeros(len(omega_dirichlet_nodes)), self.A_implicit, dummy_rhs)

        # Warm start
        self._omega_prev = np.zeros(mesh.nonodes)

        if solver == "lu":
            self.lu_implicit: SuperLU = splu(self.A_implicit_fixed.tocsc())
        elif solver == "amg":
            if not HAS_PYAMG:
                raise ImportError("pyamg required for AMG solver.")
            self.ml = pyamg.smoothed_aggregation_solver(self.A_implicit_fixed)

    def step(
        self, 
        omega: NDArray[np.float64],
        psi: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        psi_solver: StreamFunctionSolver,
        bc_updater: Callable[[NDArray[np.float64], NDArray[np.float64]], None],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Perform one IMEX time step.
        
        Parameters
        ----------
        bc_updater : function(omega, psi) -> None
            Callback to update boundary conditions for Omega (e.g. Thom's formula)
            BEFORE the next step or AFTER? 
            Standard:
            1. Solve Omega_new (using Omega_old and BCs derived from Psi_old)
            2. Solve Psi_new (from Omega_new)
            3. Update Omega BCs (from Psi_new) for next step
        """
        # 1. Advection matrix C(u)
        C = advection_matrix_2d(self.mesh, u, v)

        # 2. RHS for Omega: M*omega - dt * C * omega
        rhs = self.M @ omega - self.dt * C @ omega
        
        # Apply Dirichlet BCs to RHS (using current omega values as prescribed BCs)
        # Note: Omega values at boundary should have been updated by bc_updater in previous step
        rhs[self.omega_dirichlet_nodes - 1] = omega[self.omega_dirichlet_nodes - 1]

        # 3. Solve implicit system for Omega
        if self.solver == "lu":
            omega_new = self.lu_implicit.solve(rhs)
        else:
            residuals: list[float] = []
            omega_new = self.ml.solve(rhs, x0=self._omega_prev, tol=self.tol, maxiter=200, residuals=residuals)
            self._omega_prev = omega_new

        # 4. Solve for new Stream Function
        # Note: Psi BCs are assumed constant or handled externally? 
        # Actually psi_solver needs the boundary values. 
        # We assume `psi` vector passed in has the correct BC values set for the *new* step 
        # (or they are constant).
        psi_new = psi_solver.solve(omega_new, psi)

        # 5. Compute new velocity
        u_new, v_new = compute_velocity(psi_new, self.mesh)

        # 6. Update Vorticity BCs (Thom's formula) for next step
        # This modifies omega_new in-place at boundary nodes
        bc_updater(omega_new, psi_new)

        return omega_new, psi_new, u_new, v_new