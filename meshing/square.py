#!/usr/bin/env python3
"""
Base utility for generating a structured quadrilateral square/rectangular mesh
using the Gmsh Python API.

This file is meant to be EXTENDED by more complex geometries
(e.g. submerged_bar.py).

Features:
- Pure quadrilateral mesh
- Transfinite structured grid
- Boundary tagging via Physical Groups
- Simple rectangular domain

Typical usage (standalone):
  python square_mesh.py --Nx 40 --Ny 40 --Lx 1.0 --Ly 1.0 --out square.msh
"""

import gmsh
import argparse


def generate_square_mesh(Nx, Ny, Lx, Ly, outfile, name="square"):
    gmsh.initialize()
    gmsh.model.add(name)

    # -----------------------------
    # Geometry
    # -----------------------------
    surface = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, Ly)
    gmsh.model.occ.synchronize()

    # -----------------------------
    # Boundary detection
    # -----------------------------
    boundaries = gmsh.model.getBoundary([(2, surface)], oriented=False)

    left = []
    right = []
    bottom = []
    top = []

    tol = 1e-10
    for dim, tag in boundaries:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)

        if abs(xmin) < tol and abs(xmax) < tol:
            left.append(tag)
        elif abs(xmin - Lx) < tol and abs(xmax - Lx) < tol:
            right.append(tag)
        elif abs(ymin) < tol and abs(ymax) < tol:
            bottom.append(tag)
        elif abs(ymin - Ly) < tol and abs(ymax - Ly) < tol:
            top.append(tag)

    # -----------------------------
    # Physical groups
    # -----------------------------
    gmsh.model.addPhysicalGroup(2, [surface], name="Domain")
    gmsh.model.addPhysicalGroup(1, left, name="Left")
    gmsh.model.addPhysicalGroup(1, right, name="Right")
    gmsh.model.addPhysicalGroup(1, bottom, name="Bottom")
    gmsh.model.addPhysicalGroup(1, top, name="Top")

    # -----------------------------
    # Structured quad meshing
    # -----------------------------
    curves = gmsh.model.getEntities(1)
    for _, ctag in curves:
        if ctag in left or ctag in right:
            gmsh.model.mesh.setTransfiniteCurve(ctag, Ny + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(ctag, Nx + 1)

    gmsh.model.mesh.setTransfiniteSurface(surface)
    gmsh.model.mesh.setRecombine(2, surface)

    gmsh.option.setNumber("Mesh.RecombineAll", 1)

    # -----------------------------
    # Generate and write
    # -----------------------------
    gmsh.model.mesh.generate(2)
    gmsh.write(outfile)
    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a structured square/rectangular quad mesh")
    parser.add_argument("--Nx", type=int, required=True, help="Number of cells in x")
    parser.add_argument("--Ny", type=int, required=True, help="Number of cells in y")
    parser.add_argument("--Lx", type=float, default=1.0, help="Domain length in x")
    parser.add_argument("--Ly", type=float, default=1.0, help="Domain length in y")
    parser.add_argument("--out", type=str, default="square.msh", help="Output mesh file")

    args = parser.parse_args()

    generate_square_mesh(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=args.Lx,
        Ly=args.Ly,
        outfile=args.out,
        name="square_mesh",
    )
