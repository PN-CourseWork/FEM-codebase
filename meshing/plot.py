#!/usr/bin/env python3
"""
Mesh plotting utility for Gmsh .msh files using PyVista.

Usage:
  python plot.py --meshfile fine.msh

Supports:
- Quadrilateral meshes
- Triangle meshes
- Interactive visualization
"""

import argparse
import meshio
import numpy as np
import pyvista as pv


def plot_mesh(meshfile):
    mesh = meshio.read(meshfile)

    points = mesh.points[:, :3]

    # Determine cell type
    if "quad" in mesh.cells_dict:
        cells = mesh.cells_dict["quad"]
        cell_type = pv.CellType.QUAD
        nverts = 4
    elif "triangle" in mesh.cells_dict:
        cells = mesh.cells_dict["triangle"]
        cell_type = pv.CellType.TRIANGLE
        nverts = 3
    else:
        raise RuntimeError("No supported 2D cells found (quad or triangle)")

    # Build PyVista cell array format
    n_cells = cells.shape[0]
    cell_sizes = np.full((n_cells, 1), nverts)
    pv_cells = np.hstack([cell_sizes, cells]).astype(np.int64).ravel()

    celltypes = np.full(n_cells, cell_type, dtype=np.uint8)

    grid = pv.UnstructuredGrid(pv_cells, celltypes, points)

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        show_edges=True,
        color="white",
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title=meshfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a Gmsh .msh file with PyVista")
    parser.add_argument("--meshfile", type=str, required=True, help="Path to .msh file")

    args = parser.parse_args()
    plot_mesh(args.meshfile)
