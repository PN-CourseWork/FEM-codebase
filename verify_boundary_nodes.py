
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from FEM.datastructures import Mesh2d, LEFT, RIGHT, BOTTOM, TOP
from run_solver import setup_boundary_conditions, create_mesh

@hydra.main(version_base=None, config_path="conf", config_name="config")
def verify(cfg: DictConfig):
    # Force problem to cylinder
    cfg.problem.name = "flow_past_cylinder"

    print(f"Loading mesh: {cfg.mesh.path}")
    mesh = create_mesh(cfg)
    
    # Get BCs
    # This calls setup_boundary_conditions -> setup_cylinder_bcs
    psi, omega, psi_dir, omega_dir, extra_nodes = setup_boundary_conditions(mesh, cfg)
    
    inlet_nodes, wall_nodes, cyl_nodes = extra_nodes
    
    print(f"Inlet nodes: {len(inlet_nodes)}")
    print(f"Wall nodes: {len(wall_nodes)}")
    print(f"Cylinder nodes: {len(cyl_nodes)}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.scatter(mesh.VX, mesh.VY, s=1, c='gray', alpha=0.1, label='Interior')
    
    plt.scatter(mesh.VX[inlet_nodes], mesh.VY[inlet_nodes], s=10, c='blue', label='Inlet')
    plt.scatter(mesh.VX[wall_nodes], mesh.VY[wall_nodes], s=10, c='black', label='Walls')
    plt.scatter(mesh.VX[cyl_nodes], mesh.VY[cyl_nodes], s=20, c='red', label='Cylinder')
    
    plt.legend()
    plt.title("Boundary Node Identification")
    plt.savefig("verify_nodes.png")
    print("Saved verify_nodes.png")
    
    # Check Cylinder Radius
    cx, cy = 0.2, 0.2
    r_nodes = np.sqrt((mesh.VX[cyl_nodes] - cx)**2 + (mesh.VY[cyl_nodes] - cy)**2)
    print(f"Cylinder Node Radii: min={r_nodes.min():.4f}, max={r_nodes.max():.4f}")
    
    if np.any(np.abs(r_nodes - 0.05) > 1e-3):
        print("WARNING: Some cylinder nodes are off-geometry!")

if __name__ == "__main__":
    verify()
