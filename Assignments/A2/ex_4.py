"""
Exercise 2.4: Dirichlet Boundary Conditions

Demonstrates application of Dirichlet BCs to the FEM system.
"""

import numpy as np

from FEM.datastructures import Mesh2d
from FEM.assembly import assembly_2d
from FEM.boundary import get_boundary_nodes, dirbc_2d

print("Exercise 2.4: Dirichlet Boundary Conditions")
print("=" * 50)

# Case 1: Unit square, q(x,y) = 0, f(x,y) = 1
print("\nCase 1: Unit square, q=0, f=1 on boundary")
print("-" * 50)

mesh1 = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=4, noelms2=3)
print(f"EoTOv: {mesh1.EToV}")

qt1 = np.zeros(mesh1.nonodes)
A1, b1 = assembly_2d(mesh1, qt1)

bnodes1 = get_boundary_nodes(mesh1)
f1 = np.ones(len(bnodes1))
A1, b1 = dirbc_2d(bnodes1, f1, A1, b1)
print(f"b for case 1: {b1}")

# A(0:12, 0:12)
dense_A1 = A1.todense()
print(f"Part of A in case 1: {dense_A1[0:12,0:12]}")

print(f"\n  Mesh: 4x3 elements, {mesh1.nonodes} nodes")
print(f"  Boundary nodes: {len(bnodes1)}")

# Case 2: q(x,y) = -6x + 2y - 2, f(x,y) = x^3 - x^2*y + y^2 - 1
print("\n\nCase 2: q = -6x+2y-2, f = x^3 - x^2*y + y^2 - 1")
print("-" * 50)

mesh2 = Mesh2d(x0=-2.5, y0=-4.8, L1=7.6, L2=5.9, noelms1=4, noelms2=3)
qt2 = -6 * mesh2.VX + 2 * mesh2.VY - 2
A2, b2 = assembly_2d(mesh2, qt2)

bnodes2 = get_boundary_nodes(mesh2)
idx = bnodes2 - 1  # Convert to 0-based indices
f2 = mesh2.VX[idx] ** 3 - mesh2.VX[idx] ** 2 * mesh2.VY[idx] + mesh2.VY[idx] ** 2 - 1
A2, b2 = dirbc_2d(bnodes2, f2, A2, b2)

print(f"\n  Mesh: 4x3 elements, {mesh2.nonodes} nodes")
print(f"  Boundary nodes: {len(bnodes2)}")
print(f"b after bc mod. : {b2}")
