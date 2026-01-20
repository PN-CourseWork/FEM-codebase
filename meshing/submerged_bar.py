#!/usr/bin/env python3
"""
Generate a QUAD-DOMINANT UNSTRUCTURED mesh for a 2D free-surface flow domain
with piecewise-linear bathymetry, made as STRUCTURED AS POSSIBLE using:

- Transfinite curves (vertical + horizontal)
- Uniform vertical coordinate y ∈ [0, H]
- Straight free surface at y = H
- Quad recombination

Physical domain (solver-friendly coordinates):
  Ω = { (x,y) | x1 <= x <= x2,  0 <= y <= H(x) }

where H(x) = h(x) + H_ref.

Boundaries:
  - FreeSurface : y = H_ref (flat)
  - Bottom      : y = 0
  - Inlet       : x = x1
  - Outlet      : x = x2

Usage:
  uv run meshing/free_surface_domain.py --Nx 40 --Ny 20
"""

import gmsh
import argparse


def build_mesh(nx, ny, outfile, name_prefix="free_surface"):
    gmsh.initialize()
    gmsh.model.add(f"{name_prefix}_{outfile}")

    # -------------------------------------------------
    # Geometry parameters
    # -------------------------------------------------
    L = 10.0
    H = 2.0      # reference water depth (TOP = y=H, BOTTOM = y=0)

    # Bathymetry defined as depth below free surface
    x_bathy = [0.0, 3.0, 5.0, 6.0, 7.5, 10.0]
    h_bathy = [2.0, 2.0, 1.2, 1.2, 2.0, 2.0]  # depth h(x)

    # -------------------------------------------------
    # Geometry construction (SHIFTED COORDINATES)
    # -------------------------------------------------
    # Free surface: y = H (perfectly flat)
    top_pts = [gmsh.model.occ.addPoint(x, H, 0.0) for x in x_bathy]

    # Bottom: y = H - h(x)
    bot_pts = [gmsh.model.occ.addPoint(x, H - h, 0.0)
               for x, h in zip(x_bathy, h_bathy)]

    top_curves = []
    bot_curves = []

    for i in range(len(top_pts) - 1):
        top_curves.append(gmsh.model.occ.addLine(top_pts[i], top_pts[i + 1]))
        bot_curves.append(gmsh.model.occ.addLine(bot_pts[i], bot_pts[i + 1]))

    inlet = gmsh.model.occ.addLine(bot_pts[0], top_pts[0])
    outlet = gmsh.model.occ.addLine(top_pts[-1], bot_pts[-1])

    loop = gmsh.model.occ.addCurveLoop([
        *top_curves,
        outlet,
        *reversed(bot_curves),
        inlet,
    ])

    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    # -------------------------------------------------
    # Physical groups
    # -------------------------------------------------
    gmsh.model.addPhysicalGroup(2, [surface], name="Domain")
    gmsh.model.addPhysicalGroup(1, top_curves, name="FreeSurface")
    gmsh.model.addPhysicalGroup(1, bot_curves, name="Bottom")
    gmsh.model.addPhysicalGroup(1, [inlet], name="Inlet")
    gmsh.model.addPhysicalGroup(1, [outlet], name="Outlet")

    # -------------------------------------------------
    # Transfinite CURVES (MAXIMUM STRUCTURE WITHOUT BLOCKS)
    # -------------------------------------------------

    # Vertical alignment (Ny layers everywhere)
    gmsh.model.mesh.setTransfiniteCurve(inlet, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(outlet, ny + 1)

    # Horizontal alignment (same nodes on top & bottom)
    for c_top, c_bot, (x0, x1) in zip(top_curves, bot_curves,
                                      zip(x_bathy[:-1], x_bathy[1:])):
        seg_len = x1 - x0
        n_seg = max(2, int(round(nx * seg_len / L)) + 1)
        gmsh.model.mesh.setTransfiniteCurve(c_top, n_seg)
        gmsh.model.mesh.setTransfiniteCurve(c_bot, n_seg)

    # Ask Gmsh to respect curve directions strongly
    gmsh.option.setNumber("Mesh.TransfiniteTri", 1)

    # -------------------------------------------------
    # Quad-dominant unstructured meshing
    # -------------------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm", 8)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)

    # -------------------------------------------------
    # Generate mesh
    # -------------------------------------------------
    gmsh.model.mesh.generate(2)
    gmsh.write(outfile)
    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Highly structured quad-dominant free-surface mesh"
    )
    parser.add_argument("--Nx", type=int, required=True,
                        help="Approximate number of cells in x")
    parser.add_argument("--Ny", type=int, required=True,
                        help="Number of vertical layers")

    args = parser.parse_args()

    resolutions = {
        "coarse.msh": (args.Nx, args.Ny),
        "medium.msh": (2 * args.Nx, args.Ny),
        "fine.msh": (4 * args.Nx, args.Ny),
    }

    for fname, (nx, ny) in resolutions.items():
        print(f"Generating {fname} with Nx={nx}, Ny={ny}")
        build_mesh(nx, ny, fname)
