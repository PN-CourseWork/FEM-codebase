"""
Stokes Flow in Lid-Driven Cavity using petsc4py with Manual Assembly (Q2-Q1 Quads).
"""

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

def precompute_quadrature():
    """Precompute quadrature points, weights, and basis functions."""
    # 3x3 Gauss rule (tensor product)
    s06 = np.sqrt(0.6)
    pts_1d = np.array([-s06, 0, s06])
    wts_1d = np.array([5/9, 8/9, 5/9])

    # Tensor product quadrature
    xi_q, eta_q = np.meshgrid(pts_1d, pts_1d)
    xi_q = xi_q.flatten()
    eta_q = eta_q.flatten()
    wts_q = np.outer(wts_1d, wts_1d).flatten()
    n_qp = len(wts_q)

    # Q2 tensor indices: Vertices(4), Edges(4), Cell(1)
    q2_idx = np.array([
        [0, 0], [2, 0], [2, 2], [0, 2],  # Vertices
        [1, 0], [2, 1], [1, 2], [0, 1],  # Edges
        [1, 1]                            # Center
    ])

    # Q1 tensor indices: Vertices(4)
    q1_idx = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1]
    ])

    # Precompute all basis functions at all quadrature points
    # Q2: 9 basis functions
    N2 = np.zeros((n_qp, 9))
    dN2_dxi = np.zeros((n_qp, 9))
    dN2_deta = np.zeros((n_qp, 9))

    # Q1: 4 basis functions
    N1 = np.zeros((n_qp, 4))
    dN1_dxi = np.zeros((n_qp, 4))
    dN1_deta = np.zeros((n_qp, 4))

    for q in range(n_qp):
        xi, eta = xi_q[q], eta_q[q]

        # 1D P2 basis at xi, eta
        p2_xi = np.array([0.5*xi*(xi-1), 1-xi**2, 0.5*xi*(xi+1)])
        p2_eta = np.array([0.5*eta*(eta-1), 1-eta**2, 0.5*eta*(eta+1)])
        dp2_xi = np.array([xi-0.5, -2*xi, xi+0.5])
        dp2_eta = np.array([eta-0.5, -2*eta, eta+0.5])

        # 1D P1 basis
        p1_xi = np.array([0.5*(1-xi), 0.5*(1+xi)])
        p1_eta = np.array([0.5*(1-eta), 0.5*(1+eta)])
        dp1_xi = np.array([-0.5, 0.5])
        dp1_eta = np.array([-0.5, 0.5])

        # Q2 basis (tensor product)
        for i, (ix, iy) in enumerate(q2_idx):
            N2[q, i] = p2_xi[ix] * p2_eta[iy]
            dN2_dxi[q, i] = dp2_xi[ix] * p2_eta[iy]
            dN2_deta[q, i] = p2_xi[ix] * dp2_eta[iy]

        # Q1 basis (tensor product)
        for i, (ix, iy) in enumerate(q1_idx):
            N1[q, i] = p1_xi[ix] * p1_eta[iy]
            dN1_dxi[q, i] = dp1_xi[ix] * p1_eta[iy]
            dN1_deta[q, i] = p1_xi[ix] * dp1_eta[iy]

    return {
        'wts': wts_q,
        'N2': N2, 'dN2_dxi': dN2_dxi, 'dN2_deta': dN2_deta,
        'N1': N1, 'dN1_dxi': dN1_dxi, 'dN1_deta': dN1_deta
    }

# Precompute once at module load
QUAD_DATA = precompute_quadrature()


def element_stokes_quad(coords, nu=1.0):
    """Compute local stiffness for Stokes Q2-Q1 (vectorized)."""

    # Get precomputed quadrature data
    wts = QUAD_DATA['wts']
    N2 = QUAD_DATA['N2']           # (n_qp, 9)
    dN2_dxi = QUAD_DATA['dN2_dxi']
    dN2_deta = QUAD_DATA['dN2_deta']
    N1 = QUAD_DATA['N1']           # (n_qp, 4)
    dN1_dxi = QUAD_DATA['dN1_dxi']
    dN1_deta = QUAD_DATA['dN1_deta']

    x_nodes = coords[:4, 0]  # Q1 geometry uses 4 vertices
    y_nodes = coords[:4, 1]

    # Jacobian at all quadrature points (vectorized)
    dx_dxi = dN1_dxi @ x_nodes      # (n_qp,)
    dx_deta = dN1_deta @ x_nodes
    dy_dxi = dN1_dxi @ y_nodes
    dy_deta = dN1_deta @ y_nodes

    detJ = dx_dxi * dy_deta - dx_deta * dy_dxi  # (n_qp,)

    # Inverse Jacobian components
    dxi_dx = dy_deta / detJ
    dxi_dy = -dx_deta / detJ
    deta_dx = -dy_dxi / detJ
    deta_dy = dx_dxi / detJ

    # Weighted Jacobian determinant
    wt_detJ = wts * detJ  # (n_qp,)

    # Physical gradients of Q2 basis at all quadrature points
    # dN2_dx[q, i] = dN2_dxi[q, i] * dxi_dx[q] + dN2_deta[q, i] * deta_dx[q]
    dN2_dx = dN2_dxi * dxi_dx[:, None] + dN2_deta * deta_dx[:, None]  # (n_qp, 9)
    dN2_dy = dN2_dxi * dxi_dy[:, None] + dN2_deta * deta_dy[:, None]

    # Build element matrix
    Ke = np.zeros((22, 22))

    # A block (Laplacian): Ke[2*a, 2*b] += nu * (dN2_dx[a]*dN2_dx[b] + dN2_dy[a]*dN2_dy[b]) * wt
    # Vectorized: sum over quadrature points
    # A_ij = nu * sum_q (dN2_dx[q,i] * dN2_dx[q,j] + dN2_dy[q,i] * dN2_dy[q,j]) * wt_detJ[q]
    A = nu * (dN2_dx.T @ (dN2_dx * wt_detJ[:, None]) +
              dN2_dy.T @ (dN2_dy * wt_detJ[:, None]))  # (9, 9)

    # Place A in velocity blocks (u-u and v-v)
    Ke[0:18:2, 0:18:2] = A   # u-u block
    Ke[1:18:2, 1:18:2] = A   # v-v block

    # B block (Divergence/Gradient)
    # B[a, b] = sum_q N1[q, a] * dN2_dx[q, b] * wt_detJ[q]  (for u component)
    # B_v[a, b] = sum_q N1[q, a] * dN2_dy[q, b] * wt_detJ[q]  (for v component)
    B_u = N1.T @ (dN2_dx * wt_detJ[:, None])  # (4, 9)
    B_v = N1.T @ (dN2_dy * wt_detJ[:, None])  # (4, 9)

    # Place B in pressure-velocity blocks
    # Rows 18..21 (pressure), Cols 0,2,4,... (u) and 1,3,5,... (v)
    Ke[18:22, 0:18:2] = B_u   # Continuity: q * du/dx
    Ke[18:22, 1:18:2] = B_v   # Continuity: q * dv/dy

    # Transpose for momentum equation: -p * div(v)
    Ke[0:18:2, 18:22] = B_u.T
    Ke[1:18:2, 18:22] = B_v.T

    return Ke

def run():
    # Get MPI info
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    size = comm.getSize()

    if rank == 0:
        print(f"Running Stokes solver on {size} MPI process(es)")

    # 1. Mesh
    faces = [8, 8]  # 8x8 grid
    dm = PETSc.DMPlex().createBoxMesh(faces=faces, simplex=False, interpolate=True, comm=comm)
    dm.distribute()
    
    # 2. Section
    # Fields: u(2), p(1)
    # Q2: Verts(1), Edges(1), Cells(1).
    # Q1: Verts(1).
    
    s = PETSc.Section().create()
    s.setNumFields(2)
    s.setFieldName(0, "velocity")
    s.setFieldName(1, "pressure")
    s.setFieldComponents(0, 2)
    s.setFieldComponents(1, 1)
    
    pStart, pEnd = dm.getChart()
    s.setChart(pStart, pEnd)
    
    # Set DOFs
    # Depth 0: Vertices
    vStart, vEnd = dm.getDepthStratum(0)
    for p in range(vStart, vEnd):
        s.setDof(p, 3) # 2(u) + 1(p)
        s.setFieldDof(p, 0, 2)
        s.setFieldDof(p, 1, 1)
        
    # Depth 1: Edges
    eStart, eEnd = dm.getDepthStratum(1)
    for p in range(eStart, eEnd):
        s.setDof(p, 2) # 2(u) (mid-edge)
        s.setFieldDof(p, 0, 2)
        
    # Depth 2: Cells
    cStart, cEnd = dm.getHeightStratum(0)
    for p in range(cStart, cEnd):
        s.setDof(p, 2) # 2(u) (center)
        s.setFieldDof(p, 0, 2)
        
    s.setUp()
    dm.setSection(s)

    # 3. Matrix
    J = dm.createMat()
    
    # 4. Assembly
    coords_sec = dm.getCoordinateSection()
    coords_vec = dm.getCoordinatesLocal()
    
    def get_local_indices(dm, sec, cell):
        """Get LOCAL DOF indices for a cell in element matrix order.

        Element matrix order (Q2-Q1):
        - Velocity (18 DOFs): 4 vertices (8), 4 edges (8), 1 cell center (2)
        - Pressure (4 DOFs): 4 vertices

        DMPlex closure order: [cell, edges..., vertices...]
        We need to reorder to: [vertices(u), edges(u), cell(u), vertices(p)]
        """
        points = dm.getTransitiveClosure(cell)[0]

        # Separate by depth/type
        cell_pt = points[0]  # First is always the cell itself
        edge_pts = []
        vert_pts = []

        for p in points[1:]:  # Skip cell
            depth = dm.getLabelValue("depth", p)
            if depth == 0:  # Vertex
                vert_pts.append(p)
            elif depth == 1:  # Edge
                edge_pts.append(p)

        # Build indices in element matrix order (using LOCAL offsets)
        indices = []

        # 1. Velocity DOFs at vertices (2 per vertex)
        for v in vert_pts:
            off = sec.getOffset(v)
            indices.extend([off, off + 1])  # u, v components

        # 2. Velocity DOFs at edge midpoints (2 per edge)
        for e in edge_pts:
            off = sec.getOffset(e)
            indices.extend([off, off + 1])  # u, v components

        # 3. Velocity DOFs at cell center (2)
        off = sec.getOffset(cell_pt)
        indices.extend([off, off + 1])  # u, v components

        # 4. Pressure DOFs at vertices (1 per vertex)
        for v in vert_pts:
            off = sec.getOffset(v) + 2  # Pressure after 2 velocity components
            indices.append(off)

        return np.array(indices, dtype=np.int32)

    for c in range(cStart, cEnd):
        # Coordinates (Q1 vertices)
        c_coords_flat = dm.getVecClosure(coords_sec, coords_vec, c)
        c_coords = c_coords_flat.reshape(-1, 2)

        # Compute element stiffness
        Ke = element_stokes_quad(c_coords)

        # Get LOCAL DOF indices in element matrix order
        indices = get_local_indices(dm, s, c)

        # Use setValuesLocal with LOCAL indices (DMPlex handles ghost communication)
        J.setValuesLocal(indices, indices, Ke, PETSc.InsertMode.ADD_VALUES)

    J.assemblyBegin()
    J.assemblyEnd()

    if rank == 0:
        m, n = J.getSize()
        print(f"Matrix assembled: {m}x{n}")

    # 5. RHS & BCs
    b = dm.createGlobalVec()
    b.set(0.0)

    # Get global section for owned DOF ranges
    gsec = dm.getGlobalSection()

    # Dirichlet BCs - use "Face Sets" label for boundary markers
    face_sets = dm.getLabel("Face Sets")
    if not face_sets:
        raise RuntimeError("No 'Face Sets' label found - cannot apply BCs")

    # Collect GLOBAL indices and boundary values (only for owned DOFs)
    global_rows = []
    global_vals = []
    constrained_points = set()  # Track which points already have BCs

    def add_constraint(p, u_val, v_val):
        """Add velocity BC constraint for point p (only if owned by this rank)."""
        if p in constrained_points:
            return  # Already constrained (e.g., corner shared by two sides)
        if s.getFieldDof(p, 0) > 0:
            # Check if this point is owned by this rank (global offset >= 0)
            goff = gsec.getOffset(p)
            if goff >= 0:  # Owned by this rank
                global_rows.extend([goff, goff + 1])
                global_vals.extend([u_val, v_val])
                constrained_points.add(p)

    # Apply BCs: Top lid (y=1): u=1, v=0. All other walls: u=0, v=0
    # Face Sets: 1=bottom, 2=right, 3=top, 4=left
    bc_values = {
        3: (1.0, 0.0),  # Top lid
        1: (0.0, 0.0),  # Bottom
        2: (0.0, 0.0),  # Right
        4: (0.0, 0.0),  # Left
    }

    for side, (u_val, v_val) in bc_values.items():
        is_pts = face_sets.getStratumIS(side)
        if is_pts:
            faces = is_pts.getIndices()
            for face in faces:
                closure = dm.getTransitiveClosure(face)[0]
                for p in closure:
                    add_constraint(p, u_val, v_val)

    # Set BC values directly in global vector using GLOBAL indices
    global_rows_arr = np.array(global_rows, dtype=np.int32)
    global_vals_arr = np.array(global_vals, dtype=np.float64)

    if len(global_rows_arr) > 0:
        b.setValues(global_rows_arr, global_vals_arr, PETSc.InsertMode.INSERT_VALUES)
    b.assemblyBegin()
    b.assemblyEnd()

    # Zero BC rows in matrix using GLOBAL indices (zeroRows is collective)
    # All ranks must participate, each with their owned BC DOFs
    rows_is = PETSc.IS().createGeneral(global_rows_arr, comm=PETSc.COMM_SELF)
    J.zeroRows(rows_is, diag=1.0)

    # Pin pressure at one vertex to remove constant nullspace
    # Only rank 0 pins (vertex at origin should be on rank 0)
    pin_rows = []
    for v in range(vStart, vEnd):
        goff = gsec.getOffset(v)
        if goff >= 0:  # Only if owned
            v_coords = dm.getVecClosure(coords_sec, coords_vec, v)
            if abs(v_coords[0]) < 1e-10 and abs(v_coords[1]) < 1e-10:
                p_off = goff + 2  # Pressure DOF (after 2 velocity components)
                pin_rows.append(p_off)
                break

    # zeroRows is collective - all ranks must call even with empty IS
    pin_is = PETSc.IS().createGeneral(pin_rows, comm=PETSc.COMM_SELF)
    J.zeroRows(pin_is, diag=1.0)

    n_bc_local = len(global_rows) // 2
    if rank == 0:
        print(f"Rank 0: Applied Dirichlet BCs to {n_bc_local} boundary points")

    # 6. Solve
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(J)

    # Use GMRES with Jacobi preconditioner for parallel
    if size > 1:
        ksp.setType("gmres")
        ksp.getPC().setType("jacobi")  # Diagonal scaling - works for saddle-point
        ksp.setTolerances(rtol=1e-5, atol=1e-8, max_it=3000)
        ksp.setGMRESRestart(100)
    else:
        # Use direct LU solver for serial
        ksp.setType("preonly")
        ksp.getPC().setType("lu")

    ksp.setFromOptions()

    x = b.duplicate()
    if rank == 0:
        print("Solving linear system...")

    ksp.solve(b, x)

    if rank == 0:
        its = ksp.getIterationNumber()
        sol_norm = x.norm()
        if size > 1:
            print(f"Converged in {its} iterations, solution norm: {sol_norm:.6e}")
        else:
            print(f"Direct solve complete, solution norm: {sol_norm:.6e}")

    # Skip VTK viewer for parallel debugging
    if size == 1:
        viewer = PETSc.Viewer().createVTK("stokes_sol.vtu", "w")
        dm.view(viewer)
        x.view(viewer)
        viewer.destroy()

if __name__ == "__main__":
    run()
