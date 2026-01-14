"""
Visualize a mesh file using pyvista.
"""

import argparse
from pathlib import Path

import meshio
import pyvista as pv
import numpy as np


def visualize_mesh(
    mesh_path: Path,
    output_path: Path | None = None,
    show_edges: bool = True,
):
    """
    Visualize mesh from file and save a screenshot.

    Parameters
    ----------
    mesh_path : Path
        Path to the .msh or .vtk mesh file.
    output_path : Path, optional
        Path to save the screenshot.
    show_edges : bool
        Whether to show element edges.
    """
    # Read mesh using meshio
    mesh = meshio.read(mesh_path)

    # Convert to PyVista UnstructuredGrid
    # PyVista expects cells as a flat array [n_nodes, id1, id2, ..., n_nodes, id1, id2, ...]
    cells = mesh.cells_dict["triangle"]
    num_cells = cells.shape[0]
    cell_type = np.full(num_cells, 3, dtype=np.int32) # 3 for triangle nodes
    cells_pv = np.column_stack([cell_type, cells]).flatten()

    # PyVista cell types: 5 is triangle
    cell_types = np.full(num_cells, 5, dtype=np.uint8)

    pv_mesh = pv.UnstructuredGrid(cells_pv, cell_types, mesh.points)

    # Create Plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        pv_mesh,
        show_edges=show_edges,
        color="lightblue",
        edge_color="black",
        line_width=0.5,
    )
    
    # Set view to XY plane
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(output_path)
        print(f"Saved mesh visualization to {output_path}")
    else:
        plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize FEM mesh with PyVista")
    parser.add_argument("input", type=Path, help="Input mesh file (.msh, .vtk)")
    parser.add_argument("--output", type=Path, help="Output image file (.png, .pdf)")
    parser.add_argument("--no-edges", action="store_false", dest="show_edges", help="Hide element edges")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} does not exist.")
        return

    try:
        visualize_mesh(args.input, args.output, args.show_edges)
    except Exception as e:
        print(f"Failed to visualize mesh: {e}")
        print("Note: PyVista might require a virtual framebuffer (e.g., xvfb) on headless systems.")


if __name__ == "__main__":
    main()
