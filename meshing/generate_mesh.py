"""
Mesh generation script using pygmsh/gmsh.

Generates triangular meshes for FEM simulations and saves them in various formats.
"""

import argparse
from pathlib import Path

import gmsh
import meshio
import numpy as np


def generate_unit_square(
    mesh_size: float = 0.05,
    output_path: Path | None = None,
) -> meshio.Mesh:
    """
    Generate a triangular mesh for the unit square [0,1]^2.

    Parameters
    ----------
    mesh_size : float
        Target element size (smaller = finer mesh).
    output_path : Path, optional
        If provided, save mesh to this path.

    Returns
    -------
    meshio.Mesh
        The generated mesh.
    """
    gmsh.initialize()
    gmsh.model.add("unit_square")

    # Create geometry
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, mesh_size)

    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Add physical groups for boundary identification
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1], tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], tag=2, name="right")
    gmsh.model.addPhysicalGroup(1, [l3], tag=3, name="top")
    gmsh.model.addPhysicalGroup(1, [l4], tag=4, name="left")
    gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Extract mesh data
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :2]  # x, y only

    # Get triangles
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangles = elem_node_tags[0].reshape(-1, 3) - 1  # 0-based indexing

    # Reindex nodes (gmsh node tags may not be contiguous)
    node_map = {tag: i for i, tag in enumerate(node_tags)}
    points = np.zeros((len(node_tags), 2))
    for tag, i in node_map.items():
        points[i] = coords[int(tag) - 1]

    # Remap triangles
    triangles_remapped = np.array(
        [[node_map[node_tags[n]] for n in tri] for tri in triangles]
    )

    gmsh.finalize()

    # Create meshio mesh
    mesh = meshio.Mesh(
        points=np.column_stack([points, np.zeros(len(points))]),  # Add z=0
        cells=[("triangle", triangles_remapped)],
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.write(output_path)
        print(f"Saved mesh to {output_path}")
        print(f"  Nodes: {len(points)}")
        print(f"  Elements: {len(triangles_remapped)}")

    return mesh


def generate_rectangle(
    x0: float,
    y0: float,
    L1: float,
    L2: float,
    mesh_size: float = 0.05,
    output_path: Path | None = None,
) -> meshio.Mesh:
    """
    Generate a triangular mesh for a rectangle [x0, x0+L1] x [y0, y0+L2].

    Parameters
    ----------
    x0, y0 : float
        Bottom-left corner coordinates.
    L1, L2 : float
        Width and height of rectangle.
    mesh_size : float
        Target element size.
    output_path : Path, optional
        If provided, save mesh to this path.

    Returns
    -------
    meshio.Mesh
        The generated mesh.
    """
    gmsh.initialize()
    gmsh.model.add("rectangle")

    # Create geometry
    p1 = gmsh.model.geo.addPoint(x0, y0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(x0 + L1, y0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(x0 + L1, y0 + L2, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(x0, y0 + L2, 0, mesh_size)

    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([loop])

    # Add physical groups
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1], tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], tag=2, name="right")
    gmsh.model.addPhysicalGroup(1, [l3], tag=3, name="top")
    gmsh.model.addPhysicalGroup(1, [l4], tag=4, name="left")
    gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="domain")

    # Generate mesh
    gmsh.model.mesh.generate(2)

    # Extract mesh data
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :2]

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    triangles = elem_node_tags[0].reshape(-1, 3) - 1

    # Reindex nodes
    node_map = {tag: i for i, tag in enumerate(node_tags)}
    points = np.zeros((len(node_tags), 2))
    for tag, i in node_map.items():
        points[i] = coords[int(tag) - 1]

    triangles_remapped = np.array(
        [[node_map[node_tags[n]] for n in tri] for tri in triangles]
    )

    gmsh.finalize()

    mesh = meshio.Mesh(
        points=np.column_stack([points, np.zeros(len(points))]),
        cells=[("triangle", triangles_remapped)],
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.write(output_path)
        print(f"Saved mesh to {output_path}")
        print(f"  Nodes: {len(points)}")
        print(f"  Elements: {len(triangles_remapped)}")

    return mesh


def generate_lid_driven_cavity_mesh(
    n_elements: int = 32,
    output_dir: Path | None = None,
) -> meshio.Mesh:
    """
    Generate mesh for lid-driven cavity (unit square with appropriate resolution).

    Parameters
    ----------
    n_elements : int
        Approximate number of elements along each side.
    output_dir : Path, optional
        Directory to save mesh files.

    Returns
    -------
    meshio.Mesh
        The generated mesh.
    """
    mesh_size = 1.0 / n_elements

    output_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_path = output_dir / f"lid_driven_cavity_{n_elements}x{n_elements}.msh"

    return generate_unit_square(mesh_size=mesh_size, output_path=output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate FEM meshes using gmsh")
    parser.add_argument(
        "--type",
        choices=["unit_square", "rectangle", "lid_cavity"],
        default="unit_square",
        help="Type of mesh to generate",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=0.05,
        help="Target element size",
    )
    parser.add_argument(
        "--n-elements",
        type=int,
        default=32,
        help="Approximate elements per side (for lid_cavity)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for mesh files",
    )
    parser.add_argument(
        "--x0", type=float, default=0.0, help="x-coordinate of bottom-left corner"
    )
    parser.add_argument(
        "--y0", type=float, default=0.0, help="y-coordinate of bottom-left corner"
    )
    parser.add_argument("--L1", type=float, default=1.0, help="Width of rectangle")
    parser.add_argument("--L2", type=float, default=1.0, help="Height of rectangle")

    args = parser.parse_args()

    if args.type == "unit_square":
        output_path = args.output_dir / f"unit_square_h{args.mesh_size:.3f}.msh"
        generate_unit_square(mesh_size=args.mesh_size, output_path=output_path)
    elif args.type == "rectangle":
        output_path = args.output_dir / f"rectangle_h{args.mesh_size:.3f}.msh"
        generate_rectangle(
            x0=args.x0,
            y0=args.y0,
            L1=args.L1,
            L2=args.L2,
            mesh_size=args.mesh_size,
            output_path=output_path,
        )
    elif args.type == "lid_cavity":
        generate_lid_driven_cavity_mesh(
            n_elements=args.n_elements,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
