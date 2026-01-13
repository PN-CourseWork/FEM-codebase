"""
Exercise 2.1: 2D Mesh Generation

Demonstrates mesh generation for triangular elements on rectangular domains.
Run pytest tests/test_a2.py::TestEx1MeshGeneration for validation.
"""

import numpy as np
from pathlib import Path

from FEM.datastructures import Mesh2d


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "A2"


def main():
    print("Exercise 2.1: 2D Mesh Generation")
    print("=" * 50)

    # Case 1: Unit square
    print("\nCase 1: Unit square [0,1]x[0,1], 4x3 elements")
    print("-" * 50)

    mesh1 = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=4, noelms2=3)
    data1 = np.load(DATA_DIR / "ex1_case1.npz")

    print(f"  Nodes: {mesh1.nonodes}")
    print(f"  Elements: {mesh1.noelms}")
    print(f"  VX range: [{mesh1.VX.min():.2f}, {mesh1.VX.max():.2f}]")
    print(f"  VY range: [{mesh1.VY.min():.2f}, {mesh1.VY.max():.2f}]")

    vx_match = np.allclose(mesh1.VX, data1["VX"], rtol=1e-3)
    vy_match = np.allclose(mesh1.VY, data1["VY"], rtol=1e-3)
    etov_match = np.array_equal(mesh1.EToV, data1["EToV"])

    print(f"  VX matches expected: {'PASSED' if vx_match else 'FAILED'}")
    print(f"  VY matches expected: {'PASSED' if vy_match else 'FAILED'}")
    print(f"  EToV matches expected: {'PASSED' if etov_match else 'FAILED'}")
    # Case 2: Offset domain
    print("\nCase 2: Domain [-2.5,5.1]x[-4.8,1.1], 4x3 elements")
    print("-" * 50)

    mesh2 = Mesh2d(x0=-2.5, y0=-4.8, L1=7.6, L2=5.9, noelms1=4, noelms2=3)
    data2 = np.load(DATA_DIR / "ex1_case2.npz")

    print(f"  Nodes: {mesh2.nonodes}")
    print(f"  Elements: {mesh2.noelms}")
    print(f"  VX range: [{mesh2.VX.min():.2f}, {mesh2.VX.max():.2f}]")
    print(f"  VY range: [{mesh2.VY.min():.2f}, {mesh2.VY.max():.2f}]")

    vx_match = np.allclose(mesh2.VX, data2["VX"], rtol=1e-3)
    vy_match = np.allclose(mesh2.VY, data2["VY"], rtol=1e-3)
    etov_match = np.array_equal(mesh2.EToV, data2["EToV"])

    print(f"  VX matches expected: {'PASSED' if vx_match else 'FAILED'}")
    print(f"  VY matches expected: {'PASSED' if vy_match else 'FAILED'}")
    print(f"  EToV matches expected: {'PASSED' if etov_match else 'FAILED'}")

    print("\nAll mesh generation tests completed!")

    # print VX, VY and EToV 
    #print(f"VX: {mesh2.VX}")
    #print(f"VY: {mesh2.VY}")
    #print(f"EToV: {mesh2.EToV}")


if __name__ == "__main__":
    main()
