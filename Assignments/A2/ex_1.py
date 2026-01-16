"""
Exercise 2.1: 2D Mesh Generation

Demonstrates mesh generation for triangular elements on rectangular domains.
Validates against reference solutions from Week 2.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from FEM.datastructures import Mesh2d

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# Load validation data
VALIDATION_FILE = Path(__file__).parent / "validation_data.parquet"
df = pd.read_parquet(VALIDATION_FILE)


def get(name: str) -> np.ndarray:
    """Get validation data by name as numpy array."""
    return np.array(json.loads(df.loc[name, 'data']))


print("Exercise 2.1: 2D Mesh Generation")
print("=" * 60)

# ============================================================
# Case 1: Unit square [0,1]x[0,1], 4x3 elements
# ============================================================
print("\nCASE 1: Unit square [0,1]x[0,1], 4x3 elements")
print("=" * 60)

mesh1 = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=4, noelms2=3)

# Expected values from Week 2 solutions
expected_vx_1 = get('ex21_case1_x')
expected_vy_1 = get('ex21_case1_y')
expected_elmtab_1 = get('ex21_case1_elmtab')  # 1-indexed

# Print VX
print("\nx =")
print(mesh1.VX)

# Print VY
print("\ny =")
print(mesh1.VY)

# Print EToV (convert to 1-indexed for comparison with MATLAB output)
#TODO: Note in the report that we obviously 0 index instead since we use python! 
etov_1indexed = mesh1.EToV + 1 if mesh1.EToV.min() == 0 else mesh1.EToV
print("\nelmtab =")
print(etov_1indexed)

# Validate
vx_match_1 = np.allclose(mesh1.VX, expected_vx_1, atol=1e-4)
vy_match_1 = np.allclose(mesh1.VY, expected_vy_1, atol=1e-4)
etov_match_1 = np.array_equal(etov_1indexed, expected_elmtab_1)

print(f"\nValidation: VX={vx_match_1}, VY={vy_match_1}, EToV={etov_match_1}")

# ============================================================
# Case 2: # ============================================================
print("\n\nCASE 2: Domain [-2.5,5.1]x[-4.8,1.1], 4x3 elements")
print("=" * 60)

mesh2 = Mesh2d(x0=-2.5, y0=-4.8, L1=7.6, L2=5.9, noelms1=4, noelms2=3)

# Expected values from Week 2 solutions
expected_vx_2 = get('ex21_case2_x')
expected_vy_2 = get('ex21_case2_y')

# Print VX
print("\nx =")
print(mesh2.VX)

# Print VY
print("\ny =")
print(mesh2.VY)

# Validate
vx_match_2 = np.allclose(mesh2.VX, expected_vx_2, atol=1e-4)
vy_match_2 = np.allclose(mesh2.VY, expected_vy_2, atol=1e-4)

# EToV should be same structure as Case 1 (same noelms1, noelms2)
etov_2_1indexed = mesh2.EToV + 1 if mesh2.EToV.min() == 0 else mesh2.EToV
etov_match_2 = np.array_equal(etov_2_1indexed, expected_elmtab_1)

print(f"\nValidation: VX={vx_match_2}, VY={vy_match_2}, EToV={etov_match_2}")

# Summary
all_passed = vx_match_1 and vy_match_1 and etov_match_1 and vx_match_2 and vy_match_2 and etov_match_2
print(f"\n{'='*60}\nAll tests passed: {all_passed}")
