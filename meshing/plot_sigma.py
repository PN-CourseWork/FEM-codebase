#!/usr/bin/env python3
"""
Plot structured sigma-grid meshes written in VTK format.

Works with:
  submerged_bar_sigma.py

Boundary markers (cell_data['marker']):
  0 = Domain
  1 = FreeSurface
  2 = Bottom
  3 = Inlet
  4 = Outlet
"""

import argparse
import pyvista as pv


MARKER_COLORS = {
    1: "red",     # FreeSurface
    2: "blue",    # Bottom
    3: "green",   # Inlet
    4: "orange",  # Outlet
}

MARKER_LABELS = {
    1: "Free surface boundary",
    2: "Bottom boundary",
    3: "Body boundary",
    4: "Body boundary",
}


def plot_mesh(meshfile):
    grid = pv.read(meshfile)

    if "marker" not in grid.cell_data:
        raise RuntimeError("No 'marker' cell_data found in mesh")

    plotter = pv.Plotter()

    # -----------------------------
    # Domain
    # -----------------------------
    domain = grid.threshold(value=(0, 0), scalars="marker")
    plotter.add_mesh(
        domain,
        color="lightgray",
        show_edges=True,
        opacity=0.5,
        label="Domain",
    )

    # -----------------------------
    # Boundaries
    # -----------------------------
    for marker, color in MARKER_COLORS.items():
        boundary = grid.threshold(
            value=(marker, marker),
            scalars="marker",
        )

        if boundary.n_cells == 0:
            continue

        plotter.add_mesh(
            boundary,
            color=color,
            line_width=6,
            label=MARKER_LABELS.get(marker, f"marker {marker}"),
        )

    plotter.view_xy()
    plotter.reset_camera()
    plotter.add_axes()
    plotter.add_legend()
    plotter.show(title=meshfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot sigma-grid VTK mesh with boundary markers"
    )
    parser.add_argument(
        "--meshfile",
        type=str,
        required=True,
        help="Path to *_sigma.vtk file",
    )

    args = parser.parse_args()
    plot_mesh(args.meshfile)
