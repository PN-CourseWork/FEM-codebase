"""Visualize Stokes Lid-Driven Cavity solution using PyVista."""

import pyvista as pv
import numpy as np

def visualize_stokes_solution(filename="stokes_sol.vtu", save_png=True):
    """Create comprehensive visualization of the Stokes solution."""

    # Load the mesh
    mesh = pv.read(filename)

    # Get field names (they have auto-generated prefixes)
    vel_key = [k for k in mesh.point_data.keys() if 'velocity' in k.lower()][0]
    pres_key = [k for k in mesh.point_data.keys() if 'pressure' in k.lower()][0]

    vel = mesh.point_data[vel_key]
    pres = mesh.point_data[pres_key]

    # Compute velocity magnitude
    vel_mag = np.linalg.norm(vel[:, :2], axis=1)
    mesh.point_data['velocity_magnitude'] = vel_mag
    mesh.point_data['pressure'] = pres

    # Rename velocity for easier glyph handling
    mesh.point_data['velocity'] = vel

    print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    print(f"Velocity magnitude: [{vel_mag.min():.4f}, {vel_mag.max():.4f}]")
    print(f"Pressure: [{pres.min():.4f}, {pres.max():.4f}]")

    # Create a 2x2 subplot figure
    pv.set_plot_theme('document')
    plotter = pv.Plotter(shape=(2, 2), window_size=[1200, 1000], off_screen=save_png)

    # --- Plot 1: Velocity magnitude with arrows ---
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars='velocity_magnitude', cmap='viridis',
                     show_edges=True, edge_color='gray', line_width=0.5,
                     scalar_bar_args={'title': '|u|'})

    # Add velocity arrows (subsample for clarity)
    arrows = mesh.glyph(orient='velocity', scale='velocity_magnitude',
                        factor=0.08, geom=pv.Arrow())
    plotter.add_mesh(arrows, color='white')
    plotter.add_title("Velocity Magnitude + Vectors")
    plotter.view_xy()

    # --- Plot 2: Pressure field ---
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars='pressure', cmap='RdBu_r',
                     show_edges=True, edge_color='gray', line_width=0.5,
                     scalar_bar_args={'title': 'p'})
    plotter.add_title("Pressure Field")
    plotter.view_xy()

    # --- Plot 3: U-velocity component ---
    plotter.subplot(1, 0)
    mesh.point_data['u_velocity'] = vel[:, 0]
    plotter.add_mesh(mesh, scalars='u_velocity', cmap='coolwarm',
                     show_edges=True, edge_color='gray', line_width=0.5,
                     scalar_bar_args={'title': 'u'})
    plotter.add_title("Horizontal Velocity (u)")
    plotter.view_xy()

    # --- Plot 4: V-velocity component ---
    plotter.subplot(1, 1)
    mesh.point_data['v_velocity'] = vel[:, 1]
    plotter.add_mesh(mesh, scalars='v_velocity', cmap='coolwarm',
                     show_edges=True, edge_color='gray', line_width=0.5,
                     scalar_bar_args={'title': 'v'})
    plotter.add_title("Vertical Velocity (v)")
    plotter.view_xy()

    if save_png:
        plotter.screenshot('stokes_ldc_visualization.png')
        print("Saved: stokes_ldc_visualization.png")
    else:
        plotter.show()

    plotter.close()

    # Create a separate streamline plot
    create_streamline_plot(mesh, vel, save_png)

    return mesh


def create_streamline_plot(mesh, vel, save_png=True):
    """Create streamline visualization."""

    # For streamlines, we need a 3D mesh with 3D velocity
    # The mesh is already 2D embedded in 3D, velocity has 3 components

    plotter = pv.Plotter(window_size=[800, 700], off_screen=save_png)

    # Add mesh colored by velocity magnitude
    vel_mag = np.linalg.norm(vel[:, :2], axis=1)
    mesh.point_data['velocity_magnitude'] = vel_mag

    plotter.add_mesh(mesh, scalars='velocity_magnitude', cmap='viridis',
                     show_edges=True, edge_color='lightgray', line_width=0.3,
                     opacity=0.7, scalar_bar_args={'title': '|u|'})

    # Create seed points for streamlines
    # Use a grid of points inside the domain
    n_seeds = 8
    x_seeds = np.linspace(0.1, 0.9, n_seeds)
    y_seeds = np.linspace(0.1, 0.9, n_seeds)
    xx, yy = np.meshgrid(x_seeds, y_seeds)
    seed_points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(n_seeds**2)])
    seeds = pv.PolyData(seed_points)

    # Generate streamlines
    try:
        # Set velocity as active vectors
        mesh.point_data['vectors'] = vel
        mesh.set_active_vectors('vectors')

        streamlines = mesh.streamlines_from_source(
            seeds,
            vectors='vectors',
            integration_direction='both',
            max_time=2.0,
            initial_step_length=0.01,
            max_step_length=0.05,
        )

        if streamlines.n_points > 0:
            plotter.add_mesh(streamlines, color='black', line_width=2)
            print(f"Generated {streamlines.n_lines} streamlines")
    except Exception as e:
        print(f"Streamline generation failed: {e}")
        # Fall back to just arrows
        arrows = mesh.glyph(orient='velocity', scale='velocity_magnitude',
                           factor=0.06, geom=pv.Arrow())
        plotter.add_mesh(arrows, color='black')

    plotter.add_title("Lid-Driven Cavity - Stokes Flow (Q2-Q1)")
    plotter.view_xy()

    if save_png:
        plotter.screenshot('stokes_ldc_streamlines.png')
        print("Saved: stokes_ldc_streamlines.png")
    else:
        plotter.show()

    plotter.close()


if __name__ == "__main__":
    visualize_stokes_solution()
