"""
Exercise 2.6: Boundary Edge Data Structure

Demonstrates boundary edge identification and retrieval.
"""

from FEM.datastructures import Mesh2d, LEFT, RIGHT, BOTTOM, TOP
from FEM.boundary import get_boundary_edges

print("Exercise 2.6: Boundary Edge Data Structure")
print("=" * 50)

# Test case: Unit square mesh
mesh = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=4, noelms2=3)

# Get all boundary edges
beds_all = get_boundary_edges(mesh)

print(f"\na) Total boundary edges: {len(beds_all)}")

# Print beds array
print("\nb) Boundary edge table (beds):")
print(f"   {'Edge':<6} {'Element':<10} {'Local r':<10}")
print("   " + "-" * 30)
for p, (n, r) in enumerate(beds_all):
    print(f"   {p + 1:<6} {n:<10} {r:<10}")

# Get edges by side
beds_left = get_boundary_edges(mesh, LEFT)
beds_right = get_boundary_edges(mesh, RIGHT)
beds_top = get_boundary_edges(mesh, TOP)
beds_bottom = get_boundary_edges(mesh, BOTTOM)

print("\nc) Edges per side:")
print(f"   Left:   {len(beds_left)} edges (expected: 3)")
print(f"   Right:  {len(beds_right)} edges (expected: 3)")
print(f"   Top:    {len(beds_top)} edges (expected: 4)")
print(f"   Bottom: {len(beds_bottom)} edges (expected: 4)")
