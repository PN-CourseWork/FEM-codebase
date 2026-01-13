"""
Exercise 2.2: Basis Functions and Outer Normals

Demonstrates basis function computation for triangular elements.
Run pytest tests/test_a2.py::TestEx2BasisFunctions for validation.
"""

import numpy as np
from pathlib import Path

from FEM.datastructures import Mesh2d, outernormal


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "A2"


def main():
    print("Exercise 2.2: Basis Functions and Outer Normals")
    print("=" * 50)

    # Test with element n=9
    n = 9
    x0, y0 = -2.5, -4.8
    L1, L2 = 7.6, 5.9

    # Basis functions are now computed automatically in Mesh2d.__post_init__
    mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=4, noelms2=3)

    # Load expected data
    data = np.load(DATA_DIR / "ex2_element9.npz")
    expected_delta = float(data["delta"])

    # Exercise 2.2a - Test basfun
    print("\nPart a) Basis Functions (element n=9)")
    print("-" * 50)

    idx = n - 1
    computed_delta = mesh.delta[idx]
    delta_match = np.isclose(computed_delta, expected_delta, rtol=1e-3)

    print(f"  delta = {computed_delta:.4f}")
    print(f"  Expected delta = {expected_delta:.4f}")
    print(f"  Delta matches expected: {'PASSED' if delta_match else 'FAILED'}")

    print("\n  abc matrix:")
    print("           a_i        b_i        c_i")
    print("          " + "-" * 35)
    for i, row in enumerate(mesh.abc[idx], 1):
        print(f"    N_{i}   {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}")

    # Exercise 2.2b - Test outernormal
    print("\n\nPart b) Outer Normals (element n=9)")
    print("-" * 50)
    print("\n    Face      n1         n2")
    print("    " + "-" * 28)
    for k in [1, 2, 3]:
        n1, n2 = outernormal(n, k, mesh.VX, mesh.VY, mesh.EToV)
        print(f"      {k}     {n1:8.4f}   {n2:8.4f}")

    print("\n  Expected (from PDF):")
    print("    Face 1: n = (0, -1)")
    print("    Face 2: n = (0.7192, 0.6948)")
    print("    Face 3: n = (-1, 0)")

    print("\nAll basis function tests completed!")


if __name__ == "__main__":
    main()
