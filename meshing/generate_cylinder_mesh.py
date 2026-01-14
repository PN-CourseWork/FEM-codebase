"""
Mesh generation script for Schäfer and Turek 'Flow past a Cylinder' benchmark.
Ref: Schäfer, M., Turek, S., et al. (1996). Benchmark Computations of Laminar Flow Around a Cylinder.

Domain: [0, 2.2] x [0, 0.41]
Cylinder: Center (0.2, 0.2), Diameter 0.1
"""

import argparse
from pathlib import Path
import tempfile

import gmsh
import meshio
import pygmsh


def generate_flow_past_cylinder(
    mesh_size_background: float = 0.05,
    mesh_size_cylinder: float = 0.005,
    output_path: Path | None = None,
) -> meshio.Mesh:
    """
    Generate mesh for flow past cylinder benchmark using pygmsh.

    Parameters
    ----------
    mesh_size_background : float
        Far-field element size (used at outlet).
    mesh_size_cylinder : float
        Element size near the cylinder surface and inlet.
    output_path : Path, optional
        Path to save the mesh.

    Returns
    -------
    meshio.Mesh
        Generated mesh.
    """
    # Channel parameters
    L = 2.2
    H = 0.41
    c = [0.2, 0.2, 0.0]
    r = 0.05

    # Initialize empty geometry using the built-in kernel
    with pygmsh.geo.Geometry() as geometry:
        # Add circle (cylinder)
        circle = geometry.add_circle(c, r, mesh_size=mesh_size_cylinder)

        # Add points with finer resolution on left side (Inlet) and coarser on right (Outlet)
        p1 = geometry.add_point((0, 0, 0), mesh_size=mesh_size_cylinder)      # Bottom-Left
        p2 = geometry.add_point((L, 0, 0), mesh_size=mesh_size_background)    # Bottom-Right
        p3 = geometry.add_point((L, H, 0), mesh_size=mesh_size_background)    # Top-Right
        p4 = geometry.add_point((0, H, 0), mesh_size=mesh_size_cylinder)      # Top-Left

        points = [p1, p2, p3, p4]

        # Add lines between all points creating the rectangle
        # Indices: -1 (p4->p1), 0 (p1->p2), 1 (p2->p3), 2 (p3->p4)
        channel_lines = [
            geometry.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
        ]

        # Create a line loop and plane surface for meshing
        channel_loop = geometry.add_curve_loop(channel_lines)
        plane_surface = geometry.add_plane_surface(channel_loop, holes=[circle.curve_loop])

        # Synchronize model before adding physical entities
        geometry.synchronize()

        # Add physical groups
        # Note: In pygmsh/gmsh, physical groups are added to the model
        geometry.add_physical([plane_surface], "fluid")           # Volume
        geometry.add_physical([channel_lines[0]], "inlet")        # Left (p4->p1)
        geometry.add_physical([channel_lines[2]], "outlet")       # Right (p2->p3)
        geometry.add_physical([channel_lines[1], channel_lines[3]], "walls") # Bottom (p1->p2) & Top (p3->p4)
        geometry.add_physical(circle.curve_loop.curves, "cylinder") # Obstacle

        # Generate mesh
        # We let pygmsh generate it, which returns a meshio.Mesh
        # However, to ensure clean physical tag handling as per tutorial recommendation,
        # we can write to a temp file using gmsh API and read it back.
        
        # generate_mesh() calls gmsh.model.mesh.generate() internally
        # We can also just call it here explicitly if we want to use gmsh.write
        geometry.generate_mesh(dim=2)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output_path))
            print(f"Saved mesh to {output_path}")
            
            # Read back to return
            mesh = meshio.read(output_path)
        else:
            # Use temp file to handle physical tags correctly via IO
            with tempfile.NamedTemporaryFile(suffix=".msh") as tmp:
                gmsh.write(tmp.name)
                mesh = meshio.read(tmp.name)

    return mesh


def main():
    parser = argparse.ArgumentParser(description="Generate Flow Past Cylinder Mesh")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "cylinder_mesh.msh")
    parser.add_argument("--h-back", type=float, default=0.05, help="Background mesh size (Outlet)")
    parser.add_argument("--h-cyl", type=float, default=0.01, help="Cylinder/Inlet mesh size")
    
    args = parser.parse_args()
    
    mesh = generate_flow_past_cylinder(
        mesh_size_background=args.h_back,
        mesh_size_cylinder=args.h_cyl,
        output_path=args.output
    )
    
    print(f"  Nodes: {len(mesh.points)}")
    print(f"  Elements: {len(mesh.cells_dict.get('triangle', []))}")

if __name__ == "__main__":
    main()
