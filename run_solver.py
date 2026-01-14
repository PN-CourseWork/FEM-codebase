"""
Main solver driver using Hydra for configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from FEM.datastructures import Mesh2d, LEFT, RIGHT, BOTTOM, TOP, BOUNDARY_TOL
from FEM.boundary import get_boundary_nodes
from FEM.navier_stokes import (
    StreamFunctionSolver,
    IMEXStepper,
    VorticityBoundaryHandler,
    compute_velocity
)

def create_mesh(cfg: DictConfig) -> Mesh2d:
    if cfg.mesh.type == "unit_square":
        return Mesh2d(
            x0=0.0, y0=0.0, 
            L1=cfg.mesh.L, L2=cfg.mesh.L, 
            noelms1=cfg.mesh.N, noelms2=cfg.mesh.N
        )
    elif cfg.mesh.type == "file":
        path = hydra.utils.to_absolute_path(cfg.mesh.path)
        return Mesh2d.from_meshio(path)
    else:
        raise ValueError(f"Unknown mesh type: {cfg.mesh.type}")

def setup_cylinder_bcs(mesh: Mesh2d, cfg: DictConfig):
    """
    Setup BCs for Flow Past Cylinder.
    - Inlet: Parabolic u(y), v=0.
    - Walls: u=0, v=0.
    - Cylinder: u=0, v=0.
    - Outlet: Natural.
    """
    psi = np.zeros(mesh.nonodes)
    omega = np.zeros(mesh.nonodes)
    
    # Tag constants (from config or conventions)
    INLET = cfg.problem.BC.inlet_tag
    OUTLET = cfg.problem.BC.outlet_tag
    WALLS = cfg.problem.BC.walls_tag
    CYLINDER = cfg.problem.BC.cylinder_tag
    
    # Helper to find nodes by tag
    def get_nodes(tag):
        indices = np.where(mesh.boundary_sides == tag)[0]
        edges = mesh.boundary_edges[indices]
        from FEM.datastructures import EDGE_VERTICES
        nodes = set()
        for e_idx in range(len(edges)):
            elem = edges[e_idx, 0] - 1
            local_edge = edges[e_idx, 1] - 1
            v1, v2 = EDGE_VERTICES[local_edge]
            nodes.add(mesh.EToV[elem, v1] - 1)
            nodes.add(mesh.EToV[elem, v2] - 1)
        return np.array(list(nodes), dtype=np.int64)

    inlet_nodes = get_nodes(INLET)
    wall_nodes = get_nodes(WALLS)
    cyl_nodes = get_nodes(CYLINDER)
    
    # Parameters
    H = 0.41
    U_mean = cfg.problem.u_mean
    U_max = 1.5 * U_mean
    
    # 1. Inlet Psi Profile
    # u(y) = 4*Um * y*(H-y)/H^2
    # psi(y) = integral u dy = 4*Um/H^2 * (H*y^2/2 - y^3/3)
    ys = mesh.VY[inlet_nodes]
    psi[inlet_nodes] = (4 * U_max / H**2) * (H * ys**2 / 2 - ys**3 / 3)
    
    # 2. Inlet Omega Profile
    # omega = -du/dy = -4*Um/H^2 * (H - 2y)
    omega[inlet_nodes] = -(4 * U_max / H**2) * (H - 2 * ys)
    
    # 3. Wall Psi
    # Bottom (y=0) -> psi=0. Top (y=H) -> psi=Q
    # Q = psi(H) = 4*Um/H^2 * (H^3/6) = 2/3 * Um * H
    Q = (2/3) * U_max * H
    
    is_top = mesh.VY[wall_nodes] > H/2
    psi[wall_nodes[~is_top]] = 0.0
    psi[wall_nodes[is_top]] = Q
    
    # 4. Cylinder Psi
    # Approx: psi = psi_inlet(0.2)
    # y_c = 0.2
    psi_cyl = (4 * U_max / H**2) * (H * 0.2**2 / 2 - 0.2**3 / 3)
    psi[cyl_nodes] = psi_cyl
    
    # Dirichlet Sets
    psi_dir_nodes = np.unique(np.concatenate([inlet_nodes, wall_nodes, cyl_nodes])) + 1
    # For Omega: Inlet is fixed. Walls/Cyl are updated via Thom (so essentially Dirichlet with changing values).
    omega_dir_nodes = np.unique(np.concatenate([inlet_nodes, wall_nodes, cyl_nodes])) + 1
    
    return psi, omega, psi_dir_nodes, omega_dir_nodes, (inlet_nodes, wall_nodes, cyl_nodes)


def setup_boundary_conditions(mesh: Mesh2d, cfg: DictConfig):
    if cfg.problem.name == "flow_past_cylinder":
        return setup_cylinder_bcs(mesh, cfg)

    """
    Parse BC config and return:
    - psi_initial: Initial psi field (with Dirichlet BCs set)
    - omega_initial: Initial omega field (with Dirichlet BCs set)
    - psi_dirichlet_nodes: Indices of nodes where Psi is fixed
    - omega_dirichlet_nodes: Indices of nodes where Omega is fixed (e.g. Inlet)
    """
    psi = np.zeros(mesh.nonodes)
    omega = np.zeros(mesh.nonodes)
    
    # Identify nodes for each side
    # For unit square, we use boundary_sides
    # boundary_edges is [elem, edge_idx], boundary_sides is side_id
    
    # Helper to get nodes for a specific side tag
    def get_nodes_for_side(side_tag):
        # Find edges with this tag
        indices = np.where(mesh.boundary_sides == side_tag)[0]
        edges = mesh.boundary_edges[indices] # [elem, local_edge]
        
        nodes = set()
        from FEM.datastructures import EDGE_VERTICES
        for e_idx in range(len(edges)):
            elem = edges[e_idx, 0] - 1
            local_edge = edges[e_idx, 1] - 1
            v1_local, v2_local = EDGE_VERTICES[local_edge]
            # Get global nodes from EToV
            nodes.add(mesh.EToV[elem, v1_local] - 1) # 0-based
            nodes.add(mesh.EToV[elem, v2_local] - 1)
        return np.array(list(nodes), dtype=np.int64)

    # Map config names to tags
    # For unit square:
    side_map = {
        "left": LEFT,
        "right": RIGHT,
        "bottom": BOTTOM,
        "top": TOP
    }
    
    psi_fixed_nodes = set()
    omega_fixed_nodes = set() # Usually only inlet
    
    # Process BCs
    # Config structure: problem.BC.top = {type: ..., u: ..., psi: ...}
    bc_conf = cfg.problem.BC
    
    for side_name, side_tag in side_map.items():
        if side_name in bc_conf:
            bc = bc_conf[side_name]
            nodes = get_nodes_for_side(side_tag)
            
            # Psi BC
            if "psi" in bc:
                psi[nodes] = bc.psi
                psi_fixed_nodes.update(nodes + 1) # 1-based for solver
            
            # Omega BC
            # For walls, omega is determined by flow (Thom's formula), so NOT fixed in linear solve
            # For Inlet, omega might be fixed (if u is prescribed profile)
            # Current config for LDC: u=1, v=0 -> Omega = -du/dy? No, lid is sliding wall.
            # On lid, u=1, v=0. dpsi/dy = 1. psi is const (0).
            # This implies Neumann on psi? No.
            # LDC setup: Psi=0 on all walls. u=1 on top.
            # This means Psi is Dirichlet. Omega is unknown (calculated via Thom).
            
            pass

    # For LDC specifically:
    # All boundary nodes are Dirichlet for Psi
    all_bnodes = get_boundary_nodes(mesh)
    psi_dirichlet_nodes = all_bnodes
    
    # Omega Dirichlet nodes: usually None for LDC (all handle by Thom's)
    # But wait, Thom's formula IS a Dirichlet BC for the next step.
    # So effectively all boundary nodes are Dirichlet for Omega too in the linear solve step.
    omega_dirichlet_nodes = all_bnodes
    
    # Pre-compute top nodes for moving lid correction
    top_nodes = get_nodes_for_side(TOP)
    
    # Return 5 values to match signature
    return psi, omega, psi_dirichlet_nodes, omega_dirichlet_nodes, top_nodes


def calculate_divergence(u, v, mesh: Mesh2d):
    """Compute L2 norm of divergence for nodal velocity field."""
    v1, v2, v3 = mesh._v1, mesh._v2, mesh._v3
    delta = mesh.delta
    # Basis coeffs
    b1, b2, b3 = mesh.abc[:, 0, 1], mesh.abc[:, 1, 1], mesh.abc[:, 2, 1]
    c1, c2, c3 = mesh.abc[:, 0, 2], mesh.abc[:, 1, 2], mesh.abc[:, 2, 2]
    
    inv_2delta = 1.0 / (2 * delta)
    div_elem = (u[v1]*b1 + u[v2]*b2 + u[v3]*b3 + v[v1]*c1 + v[v2]*c2 + v[v3]*c3) * inv_2delta
    
    return np.sqrt(np.sum(div_elem**2 * np.abs(delta)))

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Running solver for problem: {cfg.problem.name}")
    print(f"Output directory: {Path.cwd()}")
    
    # Save resolved config explicitly
    with open("config_resolved.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    print("Saved config_resolved.yaml")

    # 1. Mesh
    mesh = create_mesh(cfg)
    print(f"Mesh: {mesh.nonodes} nodes, {mesh.noelms} elements")

    # 2. Setup BCs
    psi, omega, psi_dir_nodes, omega_dir_nodes, extra_nodes = setup_boundary_conditions(mesh, cfg)
    
    # 3. Solver Components
    solver_type = cfg.solver.linear_solver
    
    psi_solver = StreamFunctionSolver(mesh, psi_dir_nodes, solver=solver_type)
    stepper = IMEXStepper(mesh, cfg.solver.dt, 1.0/cfg.solver.Re, omega_dir_nodes, solver=solver_type)
    
    vort_handler = VorticityBoundaryHandler(mesh)
    
    # Define BC updater callback
    def bc_updater(omega_field, psi_field):
        if cfg.problem.name == "lid_driven_cavity":
            # 1. Apply generic (stationary) update to all
            vort_handler.apply(omega_field, psi_field)
            
            # 2. Correct Top Wall for moving lid term
            # Use pre-computed top_nodes
            top_nodes = extra_nodes
            u_lid = cfg.problem.u_lid
            
            for n in top_nodes:
                if n in vort_handler.map:
                    _, d2 = vort_handler.map[n]
                    h = np.sqrt(d2)
                    omega_field[n] -= 2 * u_lid / h
                    
        elif cfg.problem.name == "flow_past_cylinder":
            inlet_nodes, wall_nodes, cyl_nodes = extra_nodes
            # Update walls and cylinder using Thom's (stationary)
            # Pass 1-based indices to handler
            vort_handler.apply(omega_field, psi_field, target_nodes=wall_nodes+1)
            vort_handler.apply(omega_field, psi_field, target_nodes=cyl_nodes+1)
            # Inlet is fixed (Dirichlet), assume constant profile.

    # Initialize Omega BCs
    bc_updater(omega, psi)

    # 4. Time Loop
    u = np.zeros(mesh.nonodes)
    v = np.zeros(mesh.nonodes)
    
    # Residuals (steady-state increments)
    res_u = []
    res_v = []
    res_omega = []
    res_div = []
    
    import time
    start_time = time.time()
    
    print("Starting time stepping...")
    for step in range(cfg.solver.n_steps):
        u_old, v_old, omega_old = u.copy(), v.copy(), omega.copy()
        
        omega, psi, u, v = stepper.step(omega, psi, u, v, psi_solver, bc_updater)
        
        # Compute steady-state increments
        diff_u = np.max(np.abs(u - u_old))
        diff_v = np.max(np.abs(v - v_old))
        diff_omega = np.max(np.abs(omega - omega_old))
        
        # Compute Divergence (Physical Residual for Incompressibility)
        div_l2 = calculate_divergence(u, v, mesh)
        
        res_u.append(diff_u)
        res_v.append(diff_v)
        res_omega.append(diff_omega)
        res_div.append(div_l2)
        
        if (step + 1) % cfg.solver.print_every == 0:
            # Note: div_l2 is the divergence of the PROJECTED nodal velocity. 
            # It measures discretization error and will not reach machine precision (unlike algebraic residuals).
            print(f"Step {step+1:5d}: max|u|={np.max(np.abs(u)):.4f}, Δu={diff_u:.2e}, Δv={diff_v:.2e}, Δω={diff_omega:.2e}, div={div_l2:.2e}")
            
        if diff_u < cfg.solver.steady_tol and step > 100:
            print(f"Converged at step {step+1}")
            break
            
    print(f"Solved in {time.time() - start_time:.2f}s")
    
    # 5. Plot Residuals
    plt.figure(figsize=(10, 6))
    plt.semilogy(res_u, label=r'Increment $||\mathbf{u}^{n+1} - \mathbf{u}^n||_\infty$')
    plt.semilogy(res_v, label=r'Increment $||\mathbf{v}^{n+1} - \mathbf{v}^n||_\infty$')
    plt.semilogy(res_omega, label=r'Increment $||\omega^{n+1} - \omega^n||_\infty$')
    plt.semilogy(res_div, label=r'Divergence $||\nabla \cdot \mathbf{u}||_{L_2}$', color='black', lw=1.5)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlabel("Time Step ($n$)")
    plt.ylabel("Residual / Increment")
    plt.title(f"Convergence History: {cfg.problem.name}")
    plt.savefig("residual_history.png", dpi=200)
    print("Saved residual_history.png")
    
    # 6. Save VTK
    import meshio
    points = np.column_stack([mesh.VX, mesh.VY, np.zeros(mesh.nonodes)])
    cells = [("triangle", mesh.EToV - 1)]
    meshio.write("solution_final.vtk", meshio.Mesh(
        points, cells, 
        point_data={
            "u": u, 
            "v": v, 
            "psi": psi, 
            "omega": omega,
            "velocity": np.column_stack([u, v, np.zeros_like(u)])
        }
    ))
    print("Saved solution_final.vtk")

    # 7. Visualization with PyVista
    try:
        import pyvista as pv
        pv.set_plot_theme("paraview")
        pv.global_theme.allow_empty_mesh = True
        
        # Load the saved VTK
        viz_mesh = pv.read("solution_final.vtk")
        
        # Compute speed
        vectors = np.column_stack([u, v, np.zeros_like(u)])
        viz_mesh.point_data["velocity"] = vectors
        viz_mesh.point_data["speed"] = np.linalg.norm(vectors, axis=1)
        
        def save_scalar_plot(name, scalar, title, cmap="viridis"):
            plotter = pv.Plotter(off_screen=True)
            plotter.add_text(f"{title}\n{cfg.problem.name} (Re={cfg.solver.Re})", font_size=10)
            
            # Smart clim for divergent fields (u, v, omega)
            clim = None
            if name in ["u", "v", "omega"]:
                val = viz_mesh.point_data[scalar]
                abs_max = np.max(np.abs(val))
                clim = [-abs_max, abs_max]
                if cmap == "viridis": cmap = "RdBu_r"
            
            plotter.add_mesh(viz_mesh, scalars=scalar, cmap=cmap, clim=clim, show_edges=False)
            plotter.view_xy()
            plotter.camera.zoom(1.2)
            filename = f"plot_{name}.png"
            plotter.screenshot(filename)
            print(f"Saved {filename}")
            plotter.close()

        save_scalar_plot("speed", "speed", "Velocity Magnitude")
        save_scalar_plot("u", "u", "Velocity U-Component")
        save_scalar_plot("v", "v", "Velocity V-Component")
        save_scalar_plot("psi", "psi", "Stream Function", cmap="RdBu_r")
        save_scalar_plot("omega", "omega", "Vorticity", cmap="RdBu_r")
        
    except ImportError:
        print("PyVista not installed, skipping visualization.")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()

