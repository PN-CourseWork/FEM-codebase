"""
Exercise 2.3: 2D Assembly
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from FEM.datastructures import Mesh2d
from FEM.assembly import assembly_2d

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# Load validation data
# Resolve path relative to this script: Assignments/A2/ex_3.py -> ../../data/A2/validation_data.parquet
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_FILE = PROJECT_ROOT / "data/A2/validation_data.parquet"
df = pd.read_parquet(VALIDATION_FILE)


def get(name: str) -> np.ndarray:
    """Get validation data by name as numpy array."""
    return np.array(json.loads(df.loc[name, 'data']))

#TODO: is this right??? 
def extract_band_matrix(A, expected_d):
    """Extract band matrix B from sparse matrix A matching MATLAB's spdiags format.
    """
    A_dia = A.todia()
    # Get indices matching expected diagonal offsets
    d_list = list(A_dia.offsets)
    B = np.zeros((A.shape[0], len(expected_d)))

    for k, offset in enumerate(expected_d):
        if offset in d_list:
            idx = d_list.index(offset)
            B[:, k] = A_dia.data[idx, :]

    return B


print("Exercise 2.3: 2D Assembly")
print("=" * 60)

# ============================================================
# Case 1:  q(x,y) = 0
# ============================================================
print("-" * 60)
print("\nCASE 1:")
print("-" * 60)

mesh1 = Mesh2d(x0=0, y0=0, L1=1, L2=1, noelms1=4, noelms2=3)
qt1 = np.zeros(mesh1.nonodes)
A1, b1 = assembly_2d(mesh1, qt1)

# Expected values
expected_B1 = get('ex23_case1_B')
expected_d1 = get('ex23_case1_d')

# Extract band structure
B1 = extract_band_matrix(A1, expected_d1)

print("\nB (band matrix) =")
print(B1)

print("\nA  =")
print(A1.todense())

print("\nb =")
print(b1)

# Validate
B1_match = np.allclose(B1, expected_B1, atol=1e-4)
b1_match = np.allclose(b1, np.zeros_like(b1), atol=1e-10)  # b should be zero for q=0

print(f"\nValidation: B={B1_match}, b_zero={b1_match}")

# ============================================================
# Case 2:q(x,y) = -6x + 2y - 2
# ============================================================
print("\n\nCASE 2:")
print("-" * 60)

mesh2 = Mesh2d(x0=-2.5, y0=-4.8, L1=7.6, L2=5.9, noelms1=4, noelms2=3)
qt2 = -6 * mesh2.VX + 2 * mesh2.VY - 2
A2, b2 = assembly_2d(mesh2, qt2)

# Expected values
expected_B2 = get('ex23_case2_B')
expected_d2 = get('ex23_case2_d')
expected_b2 = get('ex23_case2_b')

# Extract band structure
B2 = extract_band_matrix(A2, expected_d2)

print("\nB (band matrix) =")
print(B2)

print("\nA  =")
print(A2.todense())

print("\nb =")
print(b2)

# Validate
B2_match = np.allclose(B2, expected_B2, atol=1e-4)
b2_match = np.allclose(b2, expected_b2, atol=1e-4)

print(f"\nValidation: B={B2_match}, b={b2_match}")

