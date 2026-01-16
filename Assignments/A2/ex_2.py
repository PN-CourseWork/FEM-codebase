"""
Exercise 2.2: Basis Functions and Outer Normals

Demonstrates basis function computation for triangular elements.
Validates against reference solutions from Week 2.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from FEM.datastructures import Mesh2d, outernormal

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# Load validation data
VALIDATION_FILE = Path(__file__).parent / "validation_data.parquet"
df = pd.read_parquet(VALIDATION_FILE)


def get(name: str) -> np.ndarray:
    """Get validation data by name as numpy array."""
    return np.array(json.loads(df.loc[name, 'data']))


print("Exercise 2.2: Basis Functions and Outer Normals")
print("=" * 60)

# Test with element n=9 (as specified in exercise)
n = 9
x0, y0 = -2.5, -4.8
L1, L2 = 7.6, 5.9

mesh = Mesh2d(x0=x0, y0=y0, L1=L1, L2=L2, noelms1=4, noelms2=3)

# Expected values from Week 2 solutions
expected_delta = get('ex22_delta')[0]
expected_abc = get('ex22_abc')
expected_n1 = get('ex22_face1_n')
expected_n2 = get('ex22_face2_n')
expected_n3 = get('ex22_face3_n')

# ============================================================
# Exercise 2.2a - Basis Functions
# ============================================================
print(f"\nPart a) Basis Functions (element n={n})")
print("-" * 60)

idx = n - 1  # 0-indexed
computed_delta = mesh.delta[idx]
computed_abc = mesh.abc[idx]

print(f"\ndelta = {computed_delta:.4f}")

print("\nabc =")
print(computed_abc)

# Validate
delta_match = np.isclose(computed_delta, expected_delta, atol=1e-4)
abc_match = np.allclose(computed_abc, expected_abc, atol=1e-4)

print(f"\nValidation: delta={delta_match}, abc={abc_match}")

# ============================================================
# Exercise 2.2b - Outer Normals
# ============================================================
print(f"\n\nPart b) Outer Normals (element n={n})")
print("-" * 60)

# Compute normals for all 3 faces
# Note: outernormal takes 0-indexed element number
n1_computed = outernormal(idx, 1, mesh.VX, mesh.VY, mesh.EToV)
n2_computed = outernormal(idx, 2, mesh.VX, mesh.VY, mesh.EToV)
n3_computed = outernormal(idx, 3, mesh.VX, mesh.VY, mesh.EToV)

print(f"\nFace 1 normal: [{n1_computed[0]:.4f}, {n1_computed[1]:.4f}]")
print(f"Face 2 normal: [{n2_computed[0]:.4f}, {n2_computed[1]:.4f}]")
print(f"Face 3 normal: [{n3_computed[0]:.4f}, {n3_computed[1]:.4f}]")

# Validate normals
n1_match = np.allclose(n1_computed, expected_n1, atol=1e-4)
n2_match = np.allclose(n2_computed, expected_n2, atol=1e-4)
n3_match = np.allclose(n3_computed, expected_n3, atol=1e-4)

print(f"\nValidation: n1={n1_match}, n2={n2_match}, n3={n3_match}")

# Summary
all_passed = delta_match and abc_match and n1_match and n2_match and n3_match
print(f"\n{'='*60}\nAll tests passed: {all_passed}")
