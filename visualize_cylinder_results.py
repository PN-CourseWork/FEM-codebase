"""
Visualize the cylinder flow results using PyVista with ParaView theme.
"""

import pyvista as pv
import numpy as np
from pathlib import Path

def visualize():
    vtk_file = "results_cylinder.vtk"
    if not Path(vtk_file).exists():
        print(f"Error: {vtk_file} not found.")
        return

    # Load the mesh
    mesh = pv.read(vtk_file)
    
    # Compute velocity magnitude
    vel = mesh.point_data["velocity"]
    mag = np.linalg.norm(vel, axis=1)
    mesh.point_data["speed"] = mag

    # Set ParaView theme
    pv.set_plot_theme("paraview")
    pv.global_theme.allow_empty_mesh = True

    # Create plotter
    plotter = pv.Plotter(off_screen=True, shape=(2, 1))
    
    # --- Plot 1: Velocity Magnitude + Streamlines ---
    plotter.subplot(0, 0)
    plotter.add_text("Velocity Magnitude and Streamlines", font_size=12)
    plotter.add_mesh(mesh, scalars="speed", cmap="viridis", show_edges=False)
    
    # Streamlines
    try:
        streamlines = mesh.streamlines(
            vectors="velocity",
            source_center=(0.2, 0.2, 0),
            source_radius=0.3,
            n_points=30,
        )
        if streamlines.n_points > 0:
            plotter.add_mesh(streamlines.tube(radius=0.0015), color="white")
    except:
        print("Streamlines could not be generated.")

    plotter.view_xy()
    plotter.camera.zoom(1.2)

    # --- Plot 2: Vorticity ---
    plotter.subplot(1, 0)
    plotter.add_text("Vorticity", font_size=12)
    vort = mesh.point_data["omega"]
    # Dynamic clipping for better contrast
    vort_lim = np.percentile(np.abs(vort), 98)
    plotter.add_mesh(
        mesh, 
        scalars="omega", 
        cmap="RdBu_r", 
        clim=[-vort_lim, vort_lim],
        show_edges=False
    )

    output_png = "figures/results/cylinder_re5_converged.png"
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(output_png)
    print(f"Saved visualization to {output_png}")

if __name__ == "__main__":
    visualize()
