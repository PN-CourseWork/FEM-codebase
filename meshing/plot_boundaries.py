#!/usr/bin/env python3
import meshio
import numpy as np
import pyvista as pv


def plot_boundaries(meshfile):
    mesh = meshio.read(meshfile)
    points = mesh.points[:, :3]

    plotter = pv.Plotter()

    # -------------------------------------------------
    # Plot domain (2D cells)
    # -------------------------------------------------
    if "quad" in mesh.cells_dict:
        quads = mesh.cells_dict["quad"]
        n = quads.shape[0]
        cells = np.hstack([np.full((n, 1), 4), quads]).astype(int).ravel()
        grid = pv.UnstructuredGrid(
            cells,
            np.full(n, pv.CellType.QUAD),
            points,
        )
        plotter.add_mesh(
            grid,
            color="lightgray",
            show_edges=True,
            opacity=0.4,
        )

    # -------------------------------------------------
    # Collect ALL line cells (IMPORTANT)
    # -------------------------------------------------
    line_blocks = [block.data for block in mesh.cells if block.type == "line"]
    if not line_blocks:
        raise RuntimeError("No line cells found")

    line_cells = np.vstack(line_blocks)

    # -------------------------------------------------
    # Boundary colors
    # -------------------------------------------------
    colors = {
        "FreeSurface": "red",
        "Bottom": "blue",
        "Inlet": "green",
        "Outlet": "orange",
    }

    # -------------------------------------------------
    # Plot boundaries (CORRECT indexing)
    # -------------------------------------------------
    for name, groups in mesh.cell_sets_dict.items():
        if name not in colors:
            continue
        if "line" not in groups:
            continue

        idx = np.asarray(groups["line"], dtype=int)
        lines = line_cells[idx]

        n = lines.shape[0]
        cells = np.hstack([np.full((n, 1), 2), lines]).astype(int).ravel()

        grid = pv.UnstructuredGrid(
            cells,
            np.full(n, pv.CellType.LINE),
            points,
        )

        plotter.add_mesh(
            grid,
            color=colors[name],
            line_width=6,
            label=name,
        )

    plotter.add_legend()
    plotter.view_xy()
    plotter.show()


if __name__ == "__main__":
    plot_boundaries("fine.msh")
