#!/usr/bin/env python3
"""
Structured sigma-type quadrilateral mesh generator (VTK).

This script supports TWO domains:
  - Rectangular domain
  - Beji & Battjes (1994) submerged bar (Fig. 10)

CHOOSE DOMAIN BY EDITING ONE VARIABLE BELOW.

Boundary markers (cell_data['marker']):
  0 = Domain
  1 = FreeSurface
  2 = Bottom
  3 = Inlet
  4 = Outlet
"""

import numpy as np
import meshio
import argparse

# =================================================
# USER CONFIGURATION (EDIT THIS)
# =================================================

DOMAIN_TYPE = "submerged_bar"
#DOMAIN_TYPE = "rectangle"

# Physical dimensions (Fig. 10)
L = 29.0     # domain length [m]
H = 0.40     # still water depth [m]

# Output file prefixes
RECT_PREFIX = "rectangle"
BAR_PREFIX  = "sigma"

# =================================================
# Bathymetry: Beji & Battjes (1994), Fig. 10
# =================================================
def bottom_y_submerged_bar(x):
    """
    Bottom y-coordinate.
    Free surface is at y = H.
    """
    if x < 6.0:
        h = 0.40
    elif x < 12.0:
        h = 0.40 - (0.30 / 6.0) * (x - 6.0)
    elif x < 14.0:
        h = 0.10
    elif x < 17.0:
        h = 0.10 + (0.30 / 3.0) * (x - 14.0)
    else:
        h = 0.40

    return H - h


# =================================================
# Structured sigma-grid builder
# =================================================
def build_mesh(Nx, Ny, filename, bottom_y_func=None):

    x = np.linspace(0.0, L, Nx + 1)
    sigma = np.linspace(0.0, 1.0, Ny + 1)

    points = []
    pid = {}
    k = 0

    # --------------------
    # Points (sigma grid)
    # --------------------
    for i, xi in enumerate(x):
        yb = 0.0 if bottom_y_func is None else bottom_y_func(xi)
        for j, sj in enumerate(sigma):
            y = (1.0 - sj) * yb + sj * H
            points.append([xi, y, 0.0])
            pid[(i, j)] = k
            k += 1

    points = np.asarray(points)

    # --------------------
    # Quad cells (domain)
    # --------------------
    quads = []
    for i in range(Nx):
        for j in range(Ny):
            quads.append([
                pid[(i, j)],
                pid[(i + 1, j)],
                pid[(i + 1, j + 1)],
                pid[(i, j + 1)],
            ])
    quads = np.asarray(quads)

    # --------------------
    # Boundary line cells
    # --------------------
    free_surface = []
    bottom = []
    inlet = []
    outlet = []

    for i in range(Nx):
        free_surface.append([pid[(i, Ny)], pid[(i + 1, Ny)]])
        bottom.append([pid[(i, 0)], pid[(i + 1, 0)]])

    for j in range(Ny):
        inlet.append([pid[(0, j)], pid[(0, j + 1)]])
        outlet.append([pid[(Nx, j)], pid[(Nx, j + 1)]])

    free_surface = np.asarray(free_surface)
    bottom = np.asarray(bottom)
    inlet = np.asarray(inlet)
    outlet = np.asarray(outlet)

    # --------------------
    # Cell markers
    # --------------------
    quad_markers = np.zeros(len(quads), dtype=int)

    line_cells = np.vstack([free_surface, bottom, inlet, outlet])
    line_markers = np.concatenate([
        np.full(len(free_surface), 1),  # FreeSurface
        np.full(len(bottom), 2),        # Bottom
        np.full(len(inlet), 3),         # Inlet
        np.full(len(outlet), 4),        # Outlet
    ])

    # --------------------
    # Mesh assembly
    # --------------------
    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("quad", quads),
            ("line", line_cells),
        ],
        cell_data={
            "marker": [
                quad_markers,
                line_markers,
            ]
        },
    )

    meshio.write(filename, mesh, file_format="vtk")
    print(f"✔ wrote {filename}")


# =================================================
# Main
# =================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Structured sigma-grid mesh generator (VTK)"
    )
    parser.add_argument("--Nx", type=int, required=True)
    parser.add_argument("--Ny", type=int, required=True)

    args = parser.parse_args()

    if DOMAIN_TYPE == "rectangle":
        bottom_func = None
        prefix = RECT_PREFIX
        print("▶ Generating RECTANGULAR domain")

    elif DOMAIN_TYPE == "submerged_bar":
        bottom_func = bottom_y_submerged_bar
        prefix = BAR_PREFIX
        print("▶ Generating SUBMERGED BAR domain")

    else:
        raise ValueError(f"Unknown DOMAIN_TYPE = {DOMAIN_TYPE}")

    meshes = {
        f"coarse_{prefix}.vtk": (args.Nx, args.Ny),
        f"medium_{prefix}.vtk": (2 * args.Nx, 2 * args.Ny),
        f"fine_{prefix}.vtk": (4 * args.Nx, 4 * args.Ny),
    }

    for name, (nx, ny) in meshes.items():
        build_mesh(nx, ny, name, bottom_y_func=bottom_func)
