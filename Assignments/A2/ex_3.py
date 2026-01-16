"""
Exercise 2.3: 2D Assembly

Demonstrates global stiffness matrix and load vector assembly.
"""

import numpy as np

from FEM.datastructures import Mesh2d
from FEM.assembly import assembly_2d

print("Exercise 2.3: 2D Assembly")
print("=" * 50)

# Case 1: Unit square with q(x,y) = 0
print("\nCase 1: Unit square [0,1]x[0,1], q(x,y) = 0")
print("-" * 50)

mesh1 = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=4, noelms2=3)
qt1 = np.zeros(mesh1.nonodes)
A1, b1 = assembly_2d(mesh1, qt1)

print("\nStiffness matrix A:")
print(f"full A: {A1.todia()}")
print("\nLoad vector b: all zeros (q=0)")
print(f"b1: {b1}")

# Case 2: With q(x,y) = -6x + 2y - 2
print("\n\nCase 2: Domain [-2.5,5.1]x[-4.8,1.1], q(x,y) = -6x + 2y - 2")
print("-" * 50)

mesh2 = Mesh2d(x0=-2.5, y0=-4.8, L1=7.6, L2=5.9, noelms1=4, noelms2=3)
qt2 = -6 * mesh2.VX + 2 * mesh2.VY - 2
A2, b2 = assembly_2d(mesh2, qt2)

print(f"b2: {b2}")
