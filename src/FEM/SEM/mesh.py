"""Spectral Element Mesh for 2D quad elements.

Loads quad meshes from gmsh via meshio, generates high-order LGL nodes,
and builds C⁰-continuous local-to-global DOF mapping.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import meshio

from .spectral import legendre_gauss_lobatto_nodes, SpectralElement2D


@dataclass
class SEMMesh2D:
    """Spectral Element Mesh for 2D quads.

    Parameters
    ----------
    filepath : path to gmsh mesh file (.msh)
    polynomial_order : p (uses p+1 LGL nodes per direction)
    """

    filepath: str | Path
    polynomial_order: int

    # Computed attributes
    VX: NDArray[np.float64] = field(init=False, repr=False)
    VY: NDArray[np.float64] = field(init=False, repr=False)
    quads: NDArray[np.int64] = field(init=False, repr=False)
    loc2glb: NDArray[np.int64] = field(init=False, repr=False)
    jacobians: NDArray[np.float64] = field(init=False, repr=False)
    noelms: int = field(init=False)
    nonodes: int = field(init=False)
    nloc: int = field(init=False)
    ref_elem: SpectralElement2D = field(init=False, repr=False)
    boundary_nodes: dict[str, NDArray[np.int64]] = field(init=False, repr=False)

    def __post_init__(self):
        p = self.polynomial_order
        self.nloc = (p + 1) ** 2

        # Load mesh
        mesh = meshio.read(self.filepath)
        vertices = mesh.points[:, :2].astype(np.float64)
        self.quads = next(c.data for c in mesh.cells if c.type == "quad").astype(np.int64)
        self.noelms = len(self.quads)

        # Reference element and LGL nodes
        self.ref_elem = SpectralElement2D(Nx=p, Ny=p, domain=((-1, 1), (-1, 1)))
        xi_1d = legendre_gauss_lobatto_nodes(p + 1)
        Xi, Eta = np.meshgrid(xi_1d, xi_1d, indexing="ij")
        xi_ref, eta_ref = Xi.ravel(), Eta.ravel()

        # Generate physical coordinates and Jacobians for each element
        all_x = np.empty((self.noelms, self.nloc))
        all_y = np.empty((self.noelms, self.nloc))
        self.jacobians = np.empty(self.noelms)

        for e in range(self.noelms):
            corners = vertices[self.quads[e]]  # [SW, SE, NE, NW]
            # Bilinear mapping
            N = np.array([
                0.25 * (1 - xi_ref) * (1 - eta_ref),  # SW
                0.25 * (1 + xi_ref) * (1 - eta_ref),  # SE
                0.25 * (1 + xi_ref) * (1 + eta_ref),  # NE
                0.25 * (1 - xi_ref) * (1 + eta_ref),  # NW
            ])
            all_x[e] = N.T @ corners[:, 0]
            all_y[e] = N.T @ corners[:, 1]
            # Jacobian at center
            dx_dxi = 0.25 * (corners[1, 0] - corners[0, 0] + corners[2, 0] - corners[3, 0])
            dy_deta = 0.25 * (corners[3, 1] - corners[0, 1] + corners[2, 1] - corners[1, 1])
            self.jacobians[e] = abs(dx_dxi * dy_deta)

        # Build C⁰ local-to-global mapping
        self.loc2glb, self.nonodes = self._build_c0_mapping(p)

        # Assemble unique global coordinates
        self.VX = np.zeros(self.nonodes)
        self.VY = np.zeros(self.nonodes)
        for e in range(self.noelms):
            self.VX[self.loc2glb[e]] = all_x[e]
            self.VY[self.loc2glb[e]] = all_y[e]

        # Identify boundary nodes
        tol = 1e-10
        x_min, x_max = self.VX.min(), self.VX.max()
        y_min, y_max = self.VY.min(), self.VY.max()
        self.boundary_nodes = {
            "left": np.where(np.abs(self.VX - x_min) < tol)[0],
            "right": np.where(np.abs(self.VX - x_max) < tol)[0],
            "bottom": np.where(np.abs(self.VY - y_min) < tol)[0],
            "top": np.where(np.abs(self.VY - y_max) < tol)[0],
        }
        self.boundary_nodes["all"] = np.unique(np.concatenate(list(self.boundary_nodes.values())))

    def _build_c0_mapping(self, p: int) -> tuple[NDArray[np.int64], int]:
        """Build local-to-global DOF map ensuring C⁰ continuity."""
        n = p + 1  # nodes per direction
        loc2glb = -np.ones((self.noelms, self.nloc), dtype=np.int64)
        next_dof = 0

        # Local index: (i, j) -> i * n + j  (i=xi-index, j=eta-index)
        def idx(i, j): return i * n + j

        # Corner indices: SW, SE, NE, NW
        corners = [idx(0, 0), idx(p, 0), idx(p, p), idx(0, p)]

        # 1) Corners - shared via vertex connectivity
        vertex_to_dof = {}
        for e in range(self.noelms):
            for c, loc in enumerate(corners):
                v = self.quads[e, c]
                if v not in vertex_to_dof:
                    vertex_to_dof[v] = next_dof
                    next_dof += 1
                loc2glb[e, loc] = vertex_to_dof[v]

        # 2) Edges - shared between adjacent elements
        if p > 1:
            # Edge local indices (interior only, oriented counterclockwise)
            edges = [
                [idx(i, 0) for i in range(1, p)],      # bottom: j=0
                [idx(p, j) for j in range(1, p)],      # right: i=p
                [idx(i, p) for i in range(p-1, 0, -1)], # top: j=p (reversed)
                [idx(0, j) for j in range(p-1, 0, -1)], # left: i=0 (reversed)
            ]

            edge_to_dof = {}
            for e in range(self.noelms):
                for edge_idx in range(4):
                    v0, v1 = self.quads[e, edge_idx], self.quads[e, (edge_idx + 1) % 4]
                    key = (min(v0, v1), max(v0, v1))
                    locs = edges[edge_idx]

                    if key not in edge_to_dof:
                        edge_to_dof[key] = list(range(next_dof, next_dof + len(locs)))
                        next_dof += len(locs)
                        loc2glb[e, locs] = edge_to_dof[key]
                    else:
                        # Shared edge - reverse orientation
                        loc2glb[e, locs] = edge_to_dof[key][::-1]

        # 3) Interior - unique per element
        if p > 1:
            interior = [idx(i, j) for i in range(1, p) for j in range(1, p)]
            for e in range(self.noelms):
                loc2glb[e, interior] = list(range(next_dof, next_dof + len(interior)))
                next_dof += len(interior)

        return loc2glb, next_dof

    def get_boundary_nodes(self, side: str = "all") -> NDArray[np.int64]:
        """Get global indices of boundary nodes."""
        return self.boundary_nodes[side]


def create_unit_square_mesh(nx: int, ny: int, filepath: str | Path) -> None:
    """Create uniform quad mesh on [0,1]² using gmsh."""
    import gmsh

    gmsh.initialize()
    gmsh.model.add("unit_square")

    # Geometry
    gmsh.model.geo.addPoint(0, 0, 0, 1, 1)
    gmsh.model.geo.addPoint(1, 0, 0, 1, 2)
    gmsh.model.geo.addPoint(1, 1, 0, 1, 3)
    gmsh.model.geo.addPoint(0, 1, 0, 1, 4)
    for i in range(4):
        gmsh.model.geo.addLine(i + 1, (i % 4) + 2 if i < 3 else 1, i + 1)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()

    # Transfinite for structured quads
    gmsh.model.mesh.setTransfiniteCurve(1, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(3, nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(2, ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(4, ny + 1)
    gmsh.model.mesh.setTransfiniteSurface(1)
    gmsh.model.mesh.setRecombine(2, 1)

    gmsh.model.mesh.generate(2)
    gmsh.write(str(filepath))
    gmsh.finalize()


def load_gmsh_quads(filepath: str | Path) -> tuple[NDArray, NDArray, dict]:
    """Load quad mesh from gmsh file. Returns (vertices, quads, physical_groups)."""
    mesh = meshio.read(filepath)
    vertices = mesh.points[:, :2].astype(np.float64)
    quads = next(c.data for c in mesh.cells if c.type == "quad").astype(np.int64)
    return vertices, quads, dict(mesh.point_sets) if mesh.point_sets else {}
