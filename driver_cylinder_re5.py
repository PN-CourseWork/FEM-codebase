"""
Driver for Schäfer and Turek 'Flow past a Cylinder' benchmark (Re=20).
Solves Navier-Stokes using Streamfunction-Vorticity formulation on an unstructured mesh.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from FEM.datastructures import Mesh2d, INLET, OUTLET, WALLS, CYLINDER
from FEM.assembly import assembly_2d
from FEM.boundary import dirbc_2d, get_boundary_nodes
from FEM.navier_stokes import (
    mass_matrix_2d,
    advection_matrix_2d,
    compute_velocity,
)

# Parameters
H = 0.41
D = 0.1
U_MEAN = 1.0  # Mean velocity
U_MAX = 1.5 * U_MEAN  # Max velocity for parabolic profile

def inlet_profile(y):
    """Parabolic inlet profile: u(y) = 4 * U_m * y * (H - y) / H^2"""
    return 4 * U_MAX * y * (H - y) / (H * H)

def inlet_psi(y):
    """Streamfunction at inlet: integral of u(y).
    psi(y) = 4*Um/H^2 * (H*y^2/2 - y^3/3)
    """
    return 4 * U_MAX / (H * H) * (H * y**2 / 2 - y**3 / 3)

def inlet_vorticity(y):
    """Vorticity at inlet: omega = -du/dy (since v=0)
    u(y) = C * (Hy - y^2), C = 4*Um/H^2
    u'(y) = C * (H - 2y)
    omega = -C * (H - 2y)
    """
    C = 4 * U_MAX / (H * H)
    return -C * (H - 2 * y)


def compute_velocity_generic(psi, mesh):
    """
    Compute velocity from stream function: u = dpsi/dy, v = -dpsi/dx.
    No hardcoded BCs.
    """
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    delta = mesh.delta
    abs_delta = np.abs(delta)

    # Basis function gradients
    # b_i = dphi_i/dy * 2*area ? No.
    # In datastructures:
    # abc[:, 0, 1] is b1 = y2-y3 = -dphi1/dx * 2A?
    # Standard P1: phi1 = (a1 + b1*x + c1*y) / (2A)
    # dphi1/dx = b1 / (2A)
    # dphi1/dy = c1 / (2A)
    
    # We want u = dpsi/dy = sum psi_i * dphi_i/dy
    # u_elem = (psi1*c1 + psi2*c2 + psi3*c3) / (2A)
    
    # From mesh.abc:
    # abc[:, i, 1] is "b_i" (coeff of x? No, usually b_i is coeff of x in numer).
    # mesh._compute_basis:
    # abc[:, 0, 1] = y2 - y3.
    # abc[:, 0, 2] = x3 - x2.
    # delta is area (signed?).
    # Formula used in navier_stokes.py:
    # dpsi_dx = (psi[v1] * b1 + psi[v2] * b2 + psi[v3] * b3) * inv_2delta
    # dpsi_dy = (psi[v1] * c1 + psi[v2] * c2 + psi[v3] * c3) * inv_2delta
    # AND: u = dpsi_dy, v = -dpsi_dx.
    
    # Let's verify definitions in datastructures.py:
    # abc[:, 0, 1] = y2 - y3. This is what navier_stokes calls b1.
    # abc[:, 0, 2] = x3 - x2. This is what navier_stokes calls c1.
    
    b1 = mesh.abc[:, 0, 1]
    b2 = mesh.abc[:, 1, 1]
    b3 = mesh.abc[:, 2, 1]
    c1 = mesh.abc[:, 0, 2]
    c2 = mesh.abc[:, 1, 2]
    c3 = mesh.abc[:, 2, 2]
    
    inv_2delta = 1.0 / (2 * delta)
    
    dpsi_dx = (psi[v1] * b1 + psi[v2] * b2 + psi[v3] * b3) * inv_2delta
    dpsi_dy = (psi[v1] * c1 + psi[v2] * c2 + psi[v3] * c3) * inv_2delta

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

    # Avoid divide by zero if any node has no elements (unlikely)
    mask = weights > 0
    u[mask] /= weights[mask]
    v[mask] /= weights[mask]
    
    return u, v


class UnstructuredBoundaryData:
    """Pre-computed neighbor data for unstructured vorticity BCs."""

    def __init__(self, mesh: Mesh2d):
        self.mesh = mesh
        self.map = {}  # boundary_node_idx -> (interior_node_idx, dist_squared)

        # identify boundary nodes and their tags
        self.bnodes = get_boundary_nodes(mesh)
        self.bnode_mask = np.zeros(mesh.nonodes, dtype=bool)
        self.bnode_mask[self.bnodes - 1] = True

        # Build adjacency (node -> connected nodes)
        # We iterate over edges in EToV
        adj = {i: set() for i in range(mesh.nonodes)}
        
        for i in range(mesh.noelms):
            nodes = mesh.EToV[i] - 1
            for u, v in [(0, 1), (1, 2), (2, 0)]:
                n1, n2 = nodes[u], nodes[v]
                adj[n1].add(n2)
                adj[n2].add(n1)

        # Find best interior neighbor for each boundary node
        # "Best" = connected and NOT on boundary
        for bnode in self.bnodes - 1:
            neighbors = list(adj[bnode])
            interior_neighbors = [n for n in neighbors if not self.bnode_mask[n]]
            
            if not interior_neighbors:
                # Corner case: connected only to other boundary nodes? 
                # Fallback: look for 2-hop neighbors or just pick any non-boundary
                # For this mesh, usually there is at least one interior connection.
                print(f"Warning: Boundary node {bnode} has no direct interior neighbors.")
                continue

            # Pick closest interior neighbor
            dists = [
                (mesh.VX[n] - mesh.VX[bnode])**2 + (mesh.VY[n] - mesh.VY[bnode])**2
                for n in interior_neighbors
            ]
            best_idx = np.argmin(dists)
            self.map[bnode] = (interior_neighbors[best_idx], dists[best_idx])
        
        # Debug distances
        dists_all = [d for _, d in self.map.values()]
        if dists_all:
             print(f"Boundary neighbor dists: min={np.sqrt(min(dists_all)):.2e}, max={np.sqrt(max(dists_all)):.2e}")
            
    def apply_vorticity_bc(self, omega, psi, boundary_nodes):
        """Apply omega = -2(psi_int - psi_wall) / d^2"""
        for bnode in boundary_nodes:
            idx = bnode - 1
            if idx in self.map:
                int_node, d2 = self.map[idx]
                # Formula: omega_wall = -2 * (psi_int - psi_wall) / d^2
                # Note: psi_wall is usually 0 or constant, but we use the current value in psi vector
                omega[idx] = -2 * (psi[int_node] - psi[idx]) / d2


def solve_cylinder(
    mesh_path: Path,
    Re: float,
    dt: float = 1e-3,
    n_steps: int = 1000,
):
    print(f"Loading mesh from {mesh_path}...")
    mesh = Mesh2d.from_meshio(mesh_path)
    print(f"Mesh: {mesh.nonodes} nodes, {mesh.noelms} elements")

    # Physical parameters
    nu = U_MEAN * D / Re
    print(f"Re={Re}, nu={nu:.6f}")

    # --- Identify Boundary Nodes by Tag ---
    # We loop over boundary_edges and collect nodes for each tag
    nodes_by_tag = {INLET: set(), OUTLET: set(), WALLS: set(), CYLINDER: set()}
    
    for (elem, edge), tag in zip(mesh.boundary_edges, mesh.boundary_sides):
        # element index is 1-based, EToV is 1-based in mesh object but 0-based in _v1 etc?
        # mesh.EToV is stored as 1-based in datastructures.py.
        # Let's use mesh.EToV (1-based)
        nodes = mesh.EToV[elem - 1] # [n1, n2, n3]
        
        # edge is 1, 2, 3.
        # EDGE_VERTICES = [[0, 1], [1, 2], [2, 0]]
        # So edge 1 connects nodes[0] and nodes[1]
        local_edge_indices = [[0, 1], [1, 2], [2, 0]]
        v1_idx, v2_idx = local_edge_indices[edge - 1]
        
        n1 = nodes[v1_idx]
        n2 = nodes[v2_idx]
        
        if tag in nodes_by_tag:
            nodes_by_tag[tag].add(n1)
            nodes_by_tag[tag].add(n2)

    inlet_nodes = np.array(list(nodes_by_tag[INLET]), dtype=np.int64)
    outlet_nodes = np.array(list(nodes_by_tag[OUTLET]), dtype=np.int64)
    wall_nodes = np.array(list(nodes_by_tag[WALLS]), dtype=np.int64)
    cyl_nodes = np.array(list(nodes_by_tag[CYLINDER]), dtype=np.int64)
    
    print(f"Nodes: Inlet {len(inlet_nodes)}, Outlet {len(outlet_nodes)}, Walls {len(wall_nodes)}, Cylinder {len(cyl_nodes)}")

    # --- Pre-compute Matrices ---
    print("Assembling matrices...")
    M = mass_matrix_2d(mesh)
    K, _ = assembly_2d(mesh, 1.0, 1.0, np.zeros(mesh.nonodes))
    
    # Implicit operator for Vorticity: M + nu*dt*K
    A_omega = (M + nu * dt * K).tocsc()
    lu_omega = splu(A_omega)

    # Operator for Streamfunction: K (Laplacian)
    # We fix psi at Inlet, Walls, Cylinder.
    # Outlet: Neumann (do nothing in matrix, but usually we fix one node or use regularized bc?
    # Actually, for psi, we usually set Dirichlet everywhere for cavity, but here Outlet is open.
    # If we leave Outlet "natural", it implies dpsi/dn = 0 (v_tangential = 0? No).
    # Natural BC for Laplacian is dpsi/dn = 0.
    # n = (1, 0). dpsi/dx = 0 => v = 0. This implies no vertical velocity at outlet. Reasonable.
    
    # Dirichlet nodes for PSI: Inlet + Walls + Cylinder
    dirichlet_nodes_psi = np.unique(np.concatenate([inlet_nodes, wall_nodes, cyl_nodes]))
    
    # Construct PSI matrix with Dirichlet BCs
    A_psi = K.copy()
    # RHS placeholder
    b_psi_dummy = np.zeros(mesh.nonodes)
    # We will modify A_psi to identity rows for Dirichlet nodes
    # But wait, dirbc_2d modifies b as well. We need a solver that can take new RHS.
    # So we zero out rows/cols for dirichlet nodes in K, and set diag=1.
    
    # Standard dirbc_2d handling:
    # A_psi, _ = dirbc_2d(dirichlet_nodes_psi, np.zeros(len(dirichlet_nodes_psi)), A_psi, b_psi_dummy)
    # But we need to update the RHS at each step with the vorticity source.
    # So let's just pre-factorize the matrix structure.
    # The standard `dirbc_2d` puts 1 on diagonal and 0 on row.
    # It also modifies RHS.
    # We can pre-assemble the matrix part.
    
    A_psi_fixed, _ = dirbc_2d(dirichlet_nodes_psi, np.zeros(len(dirichlet_nodes_psi)), A_psi, b_psi_dummy)
    lu_psi = splu(A_psi_fixed.tocsc())

    # --- Initialization ---
    omega = np.zeros(mesh.nonodes)
    psi = np.zeros(mesh.nonodes)
    u = np.zeros(mesh.nonodes)
    v = np.zeros(mesh.nonodes)
    
    # --- Set Fixed PSI BCs ---
    # Inlet
    psi[inlet_nodes - 1] = inlet_psi(mesh.VY[inlet_nodes - 1])
    # Walls: Bottom (y=0) -> psi=0. Top (y=H) -> psi=Q.
    # Check coords to identify top/bottom subset of walls
    # Floating point tolerance
    is_bottom = np.abs(mesh.VY[wall_nodes - 1]) < 1e-5
    is_top = np.abs(mesh.VY[wall_nodes - 1] - H) < 1e-5
    
    psi[wall_nodes[is_bottom] - 1] = 0.0
    psi[wall_nodes[is_top] - 1] = inlet_psi(H) # Q
    
    # Cylinder
    # Approximation: psi = psi_inlet(y_center)
    psi_cyl_val = inlet_psi(0.2)
    psi[cyl_nodes - 1] = psi_cyl_val

    # --- Set Fixed Omega BCs (Inlet only) ---
    omega[inlet_nodes - 1] = inlet_vorticity(mesh.VY[inlet_nodes - 1])

    # Unstructured BC helper
    bd_data = UnstructuredBoundaryData(mesh)

    print("Starting time stepping...")
    for step in range(n_steps):
        # 1. Advection matrix C(u)
        C = advection_matrix_2d(mesh, u, v)
        
        # 2. RHS for Omega: M*omega_n - dt * C * omega_n
        rhs_omega = M @ omega - dt * (C @ omega)
        
        # 3. Apply BCs for Omega
        # Dirichlet at Inlet (fixed)
        # For Walls/Cylinder, we use previous step's value (explicit update)?
        # Or we solve for interior and update boundary after?
        # Commonly: update boundary condition based on psi_n, then solve omega_n+1?
        # Or solve omega_n+1 with fixed boundary, then update boundary?
        # The `lid_driven_cavity` does: Solve Omega (BCs implicitly handled?), then Solve Psi, then Update Omega BC.
        # Actually `lid_driven_cavity` solves linear system for omega_new. What happens at boundary nodes?
        # In `IMEXStepper`, `interior_mask` is used? No, it's used to filter?
        # Ah, `dirbc_2d` is NOT called on `A_implicit`. So it solves for boundary nodes too as if they were interior?
        # No, that would be wrong.
        # Wait, `lid_driven_cavity` calls `stepper.step`.
        # `step` calls `lu_implicit.solve(rhs)`.
        # The `A_implicit` is `M + nu*dt*K`.
        # If we don't enforce Dirichlet on Omega at walls, we are enforcing "Natural BC" which is dOmega/dn = 0?
        # No, FEM assembly without BC means natural BC.
        # So it's effectively dOmega/dn = 0 at walls?
        # Then `vorticity_wall_bc_fast` OVERWRITES the values at the boundary.
        # So the boundary values computed by the solver are discarded.
        # This is an "Operator Splitting" / "Decoupled" approach.
        
        # Correct approach here:
        # A. Solve for Omega (Interior + Outlet). Fix Inlet and Walls/Cylinder to "known" values.
        #    Wait, Walls/Cylinder values are NOT known for implicit step n+1.
        #    We use values from step n (lagged BC).
        # B. Solve Psi.
        # C. Update Omega on Walls/Cylinder using new Psi.
        
        # Apply Dirichlet BCs to A_omega system?
        # Yes, we must fix the boundary values in the linear solve.
        # Since we reuse LU, we must have pre-applied the rows zeroing.
        # But the values on RHS change (because lagged BC changes).
        # We need `dirbc_2d` that works on RHS only, assuming Matrix is already fixed.
        
        # Let's fix A_omega structure once.
        # Dirichlet nodes for Omega: Inlet, Walls, Cylinder. (Outlet is natural/free).
        dirichlet_nodes_omega = np.unique(np.concatenate([inlet_nodes, wall_nodes, cyl_nodes]))
        
        # We need to rebuild A_omega or use a method to handle changing BC values with fixed matrix.
        # The standard way: Matrix has 1 on diagonal, 0 elsewhere.
        # RHS has the prescribed value.
        # So we can pre-process A_omega once.
        if step == 0:
             # Just do this once.
             # This is a bit inefficient if A_omega changes, but here A_omega is constant (M and K constant).
             # Wait, A_implicit is constant.
             # So we apply dirbc to A_omega now.
             A_omega, _ = dirbc_2d(dirichlet_nodes_omega, np.zeros(len(dirichlet_nodes_omega)), A_omega, np.zeros(mesh.nonodes))
             lu_omega = splu(A_omega.tocsc())

        # Now for the step:
        # Apply BC values to RHS
        # The values in `omega` array at boundary nodes are the "current prescribed values".
        # We want to force the solution to have these values.
        # So for Dirichlet rows i, RHS[i] = omega[i].
        # The solver will give x[i] = RHS[i] because diag is 1.
        
        rhs_omega[dirichlet_nodes_omega - 1] = omega[dirichlet_nodes_omega - 1]
        
        # Solve Omega
        omega_new = lu_omega.solve(rhs_omega)
        
        # Solve Psi
        # RHS = M * omega_new
        rhs_psi = M @ omega_new
        
        # Apply Dirichlet BCs for Psi (Fixed values: inlet, walls, cyl)
        # We need the values array
        psi_bc_values = psi.copy() # Contains the fixed setup
        # But we only need the values at Dirichlet nodes.
        # Construct full RHS with BCs applied
        # We assume A_psi_fixed is already 1-diag on boundary rows.
        # We just set RHS[i] = psi_bc_values[i]
        rhs_psi[dirichlet_nodes_psi - 1] = psi_bc_values[dirichlet_nodes_psi - 1]
        
        psi_new = lu_psi.solve(rhs_psi)
        
    print("Starting time stepping...")
    steady_tol = 1e-5
    for step in range(n_steps):
        # Ramp inlet velocity to avoid shock
        ramp = min(1.0, (step + 1) / 500)
        curr_u_max = U_MAX * ramp
        
        # 1. Advection matrix C(u)
        C = advection_matrix_2d(mesh, u, v)
        
        # 2. RHS for Omega: M*omega_n - dt * C * omega_n
        rhs_omega = M @ omega - dt * (C @ omega)
        
        # Update Dirichlet values for this step
        # Omega at Inlet
        curr_inlet_vort = - (4 * curr_u_max / (H * H)) * (H - 2 * mesh.VY[inlet_nodes - 1])
        omega[inlet_nodes - 1] = curr_inlet_vort
        
        # Psi at boundaries
        curr_inlet_psi = (4 * curr_u_max / (H * H)) * (H * mesh.VY[inlet_nodes - 1]**2 / 2 - mesh.VY[inlet_nodes - 1]**3 / 3)
        psi[inlet_nodes - 1] = curr_inlet_psi
        
        is_top = np.abs(mesh.VY[wall_nodes - 1] - H) < 1e-5
        q_curr = (2/3) * curr_u_max * H
        psi[wall_nodes[is_top] - 1] = q_curr
        
        # Cylinder psi (approx split)
        # Using y=0.2 in the ramped profile
        psi_cyl_curr = (4 * curr_u_max / (H * H)) * (H * 0.2**2 / 2 - 0.2**3 / 3)
        psi[cyl_nodes - 1] = psi_cyl_curr

        # 3. Apply BCs for Omega (Dirichlet rows)
        rhs_omega[dirichlet_nodes_omega - 1] = omega[dirichlet_nodes_omega - 1]
        
        # Solve Omega
        omega_new = lu_omega.solve(rhs_omega)
        
        # Solve Psi
        rhs_psi = M @ omega_new
        rhs_psi[dirichlet_nodes_psi - 1] = psi[dirichlet_nodes_psi - 1]
        psi_new = lu_psi.solve(rhs_psi)
        
        # Recover Velocity
        u_new, v_new = compute_velocity_generic(psi_new, mesh)
        
        # Update Wall Vorticity (Thom/Jensen)
        bd_data.apply_vorticity_bc(omega_new, psi_new, wall_nodes)
        bd_data.apply_vorticity_bc(omega_new, psi_new, cyl_nodes)
        omega_new[inlet_nodes - 1] = curr_inlet_vort

        # Check Convergence
        diff = np.max(np.abs(omega_new - omega))
        
        # Update state
        omega, psi, u, v = omega_new, psi_new, u_new, v_new
        
        if step % 500 == 0:
            print(f"Step {step:5d}: ramp={ramp:.2f}, max|u|={np.max(np.abs(u)):.4f}, diff={diff:.2e}")
            
        if diff < steady_tol and step > 1000:
            print(f"Converged at step {step} (diff={diff:.2e})")
            break

    print("Saving results...")
    # Use meshio to save
    points = np.column_stack([mesh.VX, mesh.VY, np.zeros(mesh.nonodes)])
    cells = [("triangle", mesh.EToV - 1)]
    
    meshio.write(
        "results_cylinder.vtk",
        meshio.Mesh(
            points,
            cells,
            point_data={
                "u": u,
                "v": v,
                "psi": psi,
                "omega": omega,
                "velocity": np.column_stack([u, v, np.zeros_like(u)])
            }
        )
    )
    print("Saved to results_cylinder.vtk")

import meshio
solve_cylinder(Path("meshing/cylinder_medium.msh"), Re=5.0, dt=1e-4, n_steps=10000)
