"""
Navier-Stokes (Re=100) Lid-Driven Cavity using petsc4py with Manual Assembly (Q2-Q1).
Method: Picard Iteration (Linearization).
"""

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# ... Basis functions (Same as before) ...
def basis_q2_quad(xi, eta):
    """Q2 basis functions on reference quad [-1, 1]^2."""
    def p2_1d(z):
        return np.array([0.5*z*(z-1), 1-z**2, 0.5*z*(z+1)])
    def dp2_1d(z):
        return np.array([z-0.5, -2*z, z+0.5])
        
    phi_x = p2_1d(xi)
    phi_y = p2_1d(eta)
    dphi_x = dp2_1d(xi)
    dphi_y = dp2_1d(eta)
    
    # Vertices (4), Edges (4), Center (1)
    tensor_idx = [
        (0,0), (2,0), (2,2), (0,2), # Vertices
        (1,0), (2,1), (1,2), (0,1), # Edges
        (1,1)                       # Center
    ]
    
    N = np.zeros(9)
    dN_dxi = np.zeros(9)
    dN_deta = np.zeros(9)
    
    for i, (ix, iy) in enumerate(tensor_idx):
        N[i] = phi_x[ix] * phi_y[iy]
        dN_dxi[i] = dphi_x[ix] * phi_y[iy]
        dN_deta[i] = phi_x[ix] * dphi_y[iy]
        
    return N, dN_dxi, dN_deta

def basis_q1_quad(xi, eta):
    """Q1 basis functions on reference quad [-1, 1]^2."""
    def p1_1d(z):
        return np.array([0.5*(1-z), 0.5*(1+z)])
    def dp1_1d(z):
        return np.array([-0.5, 0.5])
        
    phi_x = p1_1d(xi)
    phi_y = p1_1d(eta)
    dphi_x = dp1_1d(xi)
    dphi_y = dp1_1d(eta)
    
    tensor_idx = [(0,0), (1,0), (1,1), (0,1)]
    
    N = np.zeros(4)
    dN_dxi = np.zeros(4)
    dN_deta = np.zeros(4)
    
    for i, (ix, iy) in enumerate(tensor_idx):
        N[i] = phi_x[ix] * phi_y[iy]
        dN_dxi[i] = dphi_x[ix] * phi_y[iy]
        dN_deta[i] = phi_x[ix] * dphi_y[iy]
        
    return N, dN_dxi, dN_deta

def element_ns_quad(coords, u_old_dofs, nu=0.01):
    """
    Compute Oseen matrix for Navier-Stokes Q2-Q1.
    A(u_old) * u_new + B.T * p_new = 0
    B * u_new = 0
    
    u_old_dofs: (9, 2) array of velocity values at nodes.
    """
    # 3x3 Gauss rule
    s06 = np.sqrt(0.6)
    pts_1d = [-s06, 0, s06]
    wts_1d = [5/9, 8/9, 5/9]
    
    q_pts = []
    q_wts = []
    for i in range(3):
        for j in range(3):
            q_pts.append([pts_1d[i], pts_1d[j]])
            q_wts.append(wts_1d[i] * wts_1d[j])
            
    x_nodes = coords[:, 0]
    y_nodes = coords[:, 1]
    
    # 22x22 Matrix
    Ke = np.zeros((22, 22))
    
    for i in range(len(q_wts)):
        xi, eta = q_pts[i]
        wt = q_wts[i]
        
        # Map (Q1)
        N1, dN1_dxi, dN1_deta = basis_q1_quad(xi, eta)
        dx_dxi = np.dot(dN1_dxi, x_nodes)
        dx_deta = np.dot(dN1_deta, x_nodes)
        dy_dxi = np.dot(dN1_dxi, y_nodes)
        dy_deta = np.dot(dN1_deta, y_nodes)
        detJ = dx_dxi * dy_deta - dx_deta * dy_dxi
        dxi_dx = dy_deta / detJ
        dxi_dy = -dx_deta / detJ
        deta_dx = -dy_dxi / detJ
        deta_dy = dx_dxi / detJ
        wt *= detJ
        
        # Bases Q2
        N2, dN2_dxi, dN2_deta = basis_q2_quad(xi, eta)
        dN2_dx = dN2_dxi * dxi_dx + dN2_deta * deta_dx
        dN2_dy = dN2_dxi * dxi_dy + dN2_deta * deta_dy
        
        # Compute convection velocity u_k at quad point
        # u_k = sum N2_i * u_old_i
        # u_old_dofs shape (9, 2)
        uk = np.dot(N2, u_old_dofs) # (2,) [u, v]
        
        # Assemble
        for a in range(9):
            # Test functions phi_a
            phi_a = N2[a]
            dphi_a_dx = dN2_dx[a]
            dphi_a_dy = dN2_dy[a]
            
            for b in range(9):
                # Trial functions phi_b
                phi_b = N2[b]
                dphi_b_dx = dN2_dx[b]
                dphi_b_dy = dN2_dy[b]
                
                # Diffusion: nu * grad(u):grad(v)
                diff = nu * (dphi_b_dx * dphi_a_dx + dphi_b_dy * dphi_a_dy)
                
                # Convection: (u_k . grad(u)) . v
                # u_k . grad(phi_b) * phi_a
                # (uk[0]*dphi_b_dx + uk[1]*dphi_b_dy) * phi_a
                conv = (uk[0] * dphi_b_dx + uk[1] * dphi_b_dy) * phi_a
                
                val = (diff + conv) * wt
                
                Ke[2*a, 2*b] += val
                Ke[2*a+1, 2*b+1] += val
                
        # Pressure blocks
        for a in range(4): # Pressure nodes
            psi_a = N1[a]
            for b in range(9): # Velocity nodes
                phi_b = N2[b]
                dphi_b_dx = dN2_dx[b]
                dphi_b_dy = dN2_dy[b]
                
                # Divergence B: -psi_a * div(u_b)
                # B[a, 2b]   += -psi_a * dphi_b_dx
                # B[a, 2b+1] += -psi_a * dphi_b_dy
                # Note: Integration by parts usually gives positive if grad(p).v?
                # Continuity: integral(q div u) = 0.
                # Momentum: -integral(p div v).
                # Let's stick to standard:
                # K_up = -integral(p div v) -> B^T
                # K_pu = -integral(q div u) -> B (or + if constraint is div u = 0)
                
                # Terms:
                val_x = psi_a * dphi_b_dx * wt
                val_y = psi_a * dphi_b_dy * wt
                
                # Row 18+a (Continuity): integral(q * div u) -> +val
                Ke[18+a, 2*b] += val_x
                Ke[18+a, 2*b+1] += val_y
                
                # Col 18+a (Momentum): integral(grad p * v) = -integral(p div v)
                # So symmetric? Yes if we keep signs consistent.
                Ke[2*b, 18+a] += val_x
                Ke[2*b+1, 18+a] += val_y
                
    return Ke

def get_local_indices(dm, s, cell):
    points = dm.getTransitiveClosure(cell)[0]
    indices = []
    for p in points:
        dof = s.getDof(p)
        if dof > 0:
            off = s.getOffset(p)
            indices.extend(range(off, off+dof))
    return np.array(indices, dtype=np.int32)

def run():
    # 1. Mesh
    faces = [16, 16] # 16x16 Quads
    dm = PETSc.DMPlex().createBoxMesh(faces=faces, simplex=False, interpolate=True)
    dm.distribute()
    
    # 2. Section (Q2-Q1)
    s = PETSc.Section().create()
    s.setNumFields(2)
    s.setFieldComponents(0, 2)
    s.setFieldComponents(1, 1)
    
    pStart, pEnd = dm.getChart()
    s.setChart(pStart, pEnd)
    
    vStart, vEnd = dm.getDepthStratum(0)
    for p in range(vStart, vEnd):
        s.setDof(p, 3)
        s.setFieldDof(p, 0, 2)
        s.setFieldDof(p, 1, 1)
        
    eStart, eEnd = dm.getDepthStratum(1)
    for p in range(eStart, eEnd):
        s.setDof(p, 2)
        s.setFieldDof(p, 0, 2)
        
    cStart, cEnd = dm.getHeightStratum(0)
    for p in range(cStart, cEnd):
        s.setDof(p, 2)
        s.setFieldDof(p, 0, 2)
        
    s.setUp()
    dm.setSection(s)
    
    # 3. Setup Solver
    # Vectors
    x = dm.createGlobalVec()
    b = dm.createGlobalVec()
    x.set(0.0) # Initial guess (u=0)
    
    # Get previous step solution (for Picard)
    x_old = x.duplicate()
    
    # Coordinates
    coords_sec = dm.getCoordinateSection()
    coords_vec = dm.getCoordinatesLocal()
    
    # Solver
    ksp = PETSc.KSP().create()
    ksp.setType("preonly") # Direct solver
    ksp.getPC().setType("lu") 
    # ksp.getPC().setFactorSolverType("mumps") # Optional, defaults to petsc lu if seq
    ksp.setFromOptions()
    
    # Nonlinear Loop (Picard)
    nu = 0.01 # Re = 100
    tol = 1e-5
    max_iter = 20
    alpha = 0.5 # Relaxation
    
    print(f"Solving Navier-Stokes (Re={1/nu}) on {faces} grid.")
    
    for k in range(max_iter):
        J = dm.createMat() # Create fresh matrix (or zero it)
        b.set(0.0)
        
        # Local vector of x_old for element evaluation
        # We need a local vector including ghosts
        loc_x_old = dm.getLocalVec()
        dm.globalToLocal(x_old, loc_x_old)
        x_old_arr = loc_x_old.getArray()
        
        # Assembly
        for c in range(cStart, cEnd):
            # Coords
            c_coords = dm.getVecClosure(coords_sec, coords_vec, c).reshape(-1, 2)
            
            # Old solution at cell nodes
            # indices in loc_x_old?
            # getVecClosure handles mapping from Section to values
            # s is the section for x.
            c_sol = dm.getVecClosure(s, loc_x_old, c)
            # Size 22. 
            # We need only Velocity components for u_old_dofs.
            # Q2 velocity nodes are first 18 values?
            # getVecClosure ordering depends on point ordering in closure.
            # Order: V1, V2, V3, V4, E1, E2, E3, E4, C
            # Vertices have 3 dofs (u,v,p).
            # Edges have 2 (u,v).
            # Cell has 2 (u,v).
            
            # We need to extract (u,v) for the 9 nodes.
            # It's messy to parse flat closure.
            # But the order is consistent.
            # Point 0 (V1): u,v,p
            # ...
            # Point 4 (E1): u,v
            
            # Map:
            # V1: 0,1. (2 is p)
            # V2: 3,4. (5 is p)
            # V3: 6,7. (8 is p)
            # V4: 9,10. (11 is p)
            # E1: 12,13
            # E2: 14,15
            # E3: 16,17
            # E4: 18,19
            # C:  20,21
            
            # This order assumes closure points are [V1, V2, V3, V4, E1, E2, E3, E4, C]
            # Standard DMPlex closure usually lists vertices then edges then faces.
            # Assuming consistent box mesh ordering.
            
            u_old_vals = np.zeros((9, 2))
            
            # Helpers indices in closure array
            # V: 0, 3, 6, 9
            # E: 12, 14, 16, 18
            # C: 20
            
            # Careful: check point ordering
            # Closure: 4 verts, 4 edges, 1 cell.
            # Verts: 3 dofs each. Offsets 0, 3, 6, 9.
            # Edges: 2 dofs each. Offsets 12, 14, 16, 18.
            # Cell: 2 dofs. Offset 20.
            
            # Extract
            u_old_vals[0] = c_sol[0:2]
            u_old_vals[1] = c_sol[3:5]
            u_old_vals[2] = c_sol[6:8]
            u_old_vals[3] = c_sol[9:11]
            u_old_vals[4] = c_sol[12:14]
            u_old_vals[5] = c_sol[14:16]
            u_old_vals[6] = c_sol[16:18]
            u_old_vals[7] = c_sol[18:20]
            u_old_vals[8] = c_sol[20:22]
            
            Ke = element_ns_quad(c_coords, u_old_vals, nu=nu)
            
            indices = get_local_indices(dm, s, c)
            J.setValuesLocal(indices, indices, Ke, PETSc.InsertMode.ADD_VALUES)
            
        J.assemblyBegin()
        J.assemblyEnd()
        
        # BCs
        label = dm.getLabel("marker")
        if not label:
            label = dm.getLabel("Face Sets")
            
        local_rows = []
        local_vals = []
        
        def add_constraint(p, u_val, v_val):
            dof = s.getDof(p)
            if dof > 0:
                off = s.getOffset(p)
                # Check field 0 (velocity)
                if s.getFieldDof(p, 0) > 0:
                    # Offset of field 0 is just 'off' since it's first? Yes.
                    local_rows.extend([off, off+1])
                    local_vals.extend([u_val, v_val])
        
        # Boundary points loop
        # Check all possible marker IDs (usually 1 for all boundary in some versions, or 1,2,3,4)
        # If we found points on 1 but not others, maybe 1 is "Boundary".
        
        # Collect all boundary points from likely labels
        b_points = set()
        for i in [1, 2, 3, 4]:
            is_pts = label.getStratumIS(i)
            if is_pts:
                b_points.update(is_pts.getIndices())
                
        print(f"Total boundary points found: {len(b_points)}")
        
        # Coordinate access for BC check
        # We need to map point index to coordinates.
        # This is slow in python loop but fine for 16x16.
        coords_arr = coords_vec.getArray().reshape(-1, 2)
        # Note: coords_vec indices might not match point indices directly if section is involved?
        # Coordinates are stored in a Section too.
        # dm.getCoordinatesLocal returns a Vec.
        # To get coords of point p: dm.getVecClosure(coord_section, coord_vec, p)
        
        for p in b_points:
            # Get coords
            # Usually closure of vertex p is just the vertex coords.
            # Closure of edge p is 2 vertices?
            # We want coordinate of the point itself (if it has dofs).
            
            # If p has DOFs in 's', we apply BC.
            # Coordinates are usually defined on vertices (depth 0).
            # If p is edge (depth 1), does it have coordinates?
            # BoxMesh interpolate=True: yes, edges are points.
            # But standard coordinate section is P1 (vertices only).
            # So edges don't have explicit coordinates in the coord vec usually.
            # We can get coordinates from closure (vertices of edge) and average.
            
            # For Q2, we have DOFs on edges and cells.
            # BCs apply to Vertices and Edges.
            
            dof = s.getDof(p)
            if dof > 0:
                # Get coords
                # closure = dm.getVecClosure(coords_sec, coords_vec, p)
                # closure size?
                # If p is vertex: size 2.
                # If p is edge: size 4 (2 verts). Average them.
                
                c_vals = dm.getVecClosure(coords_sec, coords_vec, p).reshape(-1, 2)
                centroid = np.mean(c_vals, axis=0)
                cx, cy = centroid
                
                # Check BC
                tol = 1e-5
                # Top: y=1. u=1, v=0
                if abs(cy - 1.0) < tol:
                    add_constraint(p, 1.0, 0.0)
                # Walls: y=0, x=0, x=1. u=0, v=0
                elif abs(cy) < tol or abs(cx) < tol or abs(cx - 1.0) < tol:
                    add_constraint(p, 0.0, 0.0)
                    
        print(f"Applied BCs to {len(local_rows)} dofs.")
        
        # Pin Pressure at Vertex 0
        vStart, vEnd = dm.getDepthStratum(0)
        p0 = vStart
        dof = s.getDof(p0)
        if dof > 0:
            off = s.getOffset(p0)
            # Check field 1 (pressure)
            if s.getFieldDof(p0, 1) > 0:
                p_off = off + s.getFieldDof(p0, 0) # u_dof is 2.
                # Actually getFieldOffset is safer
                p_off = s.getFieldOffset(p0, 1)
                
                local_rows.append(p_off)
                local_vals.append(0.0)
                print(f"Pinned pressure at row {p_off}")

        # Apply BCs
        print(f"DEBUG: rows={local_rows[:5]}, vals={local_vals[:5]}")
        b.setValuesLocal(local_rows, local_vals, PETSc.InsertMode.INSERT_VALUES)
        b.assemble()
        
        print(f"RHS norm: {b.norm()}")
        
        rows_is = PETSc.IS().createGeneral(local_rows)
        J.zeroRowsLocal(rows_is, diag=1.0, x=None, b=None)
        
        # Solve
        ksp.setOperators(J)
        x.set(0.0) # Reset guess? Or keep?
        ksp.solve(b, x)
        print(f"KSP reason: {ksp.getConvergedReason()}")
        
        # Convergence check
        diff_vec = x.duplicate()
        diff_vec.waxpy(-1.0, x_old, x) # diff = x - x_old
        norm = diff_vec.norm()
        x_norm = x.norm()
        
        print(f"Iter {k+1}: Norm diff = {norm:.2e}")
        
        if norm < tol * x_norm:
            print("Converged!")
            break
            
        # Relaxation: x_old = alpha * x + (1-alpha) * x_old
        # x already holds new solution.
        # We want x_old to become the new guess.
        # x_new_guess = alpha * x_new + (1-alpha) * x_old
        # x_old.axpby(alpha, 1.0-alpha, x) # y = alpha*x + beta*y
        x_old.axpby(alpha, 1.0-alpha, x)
        
        dm.restoreLocalVec(loc_x_old)
        J.destroy()
        
    # Save
    viewer = PETSc.Viewer().createVTK("ns_sol.vtu", "w")
    dm.view(viewer)
    x.view(viewer)
    viewer.destroy()

if __name__ == "__main__":
    run()
