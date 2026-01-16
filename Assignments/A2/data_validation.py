"""
Print validation data from Week 2 solutions for verification.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

np.set_printoptions(precision=4, suppress=True, linewidth=120)

data_file = Path('data/A2/validation_data.parquet')
df = pd.read_parquet(data_file)


def get(name: str) -> np.ndarray:
    """Get validation data by name as numpy array."""
    return np.array(json.loads(df.loc[name, 'data']))


print("=" * 70)
print("WEEK 2 VALIDATION DATA")
print("=" * 70)

# ============================================================
# EXERCISE 2.1
# ============================================================
print("\n" + "=" * 70)
print("EXERCISE 2.1 - Mesh Data")
print("=" * 70)

print("\nCASE 1: Unit domain [0,1] x [0,1], 4x3 elements")
print("-" * 50)
print(f"x (20 nodes):\n{get('ex21_case1_x')}")
print(f"\ny (20 nodes):\n{get('ex21_case1_y')}")
print(f"\nelmtab (24 elements, 1-indexed):\n{get('ex21_case1_elmtab')}")

print("\nCASE 2: Domain [-2.5, 5.1] x [-4.8, 1.1]")
print("-" * 50)
print(f"x (20 nodes):\n{get('ex21_case2_x')}")
print(f"\ny (20 nodes):\n{get('ex21_case2_y')}")

# ============================================================
# EXERCISE 2.2
# ============================================================
print("\n" + "=" * 70)
print("EXERCISE 2.2 - Element Properties")
print("=" * 70)

print(f"\ndelta = {get('ex22_delta')[0]:.4f}")
print(f"\nabc matrix (3x3):\n{get('ex22_abc')}")
print(f"\nFace 1 normal: {get('ex22_face1_n')}")
print(f"Face 2 normal: {get('ex22_face2_n')}")
print(f"Face 3 normal: {get('ex22_face3_n')}")

# ============================================================
# EXERCISE 2.3
# ============================================================
print("\n" + "=" * 70)
print("EXERCISE 2.3 - Assembly (before BC application)")
print("=" * 70)

print("\nCASE 1: Unit domain")
print("-" * 50)
print(f"B matrix (20x5) - sparse band structure:\n{get('ex23_case1_B')}")
print(f"\nd (diagonals): {get('ex23_case1_d')}")

print("\nCASE 2: Scaled domain [-2.5, 5.1] x [-4.8, 1.1]")
print("-" * 50)
print(f"B matrix (20x5):\n{get('ex23_case2_B')}")
print(f"\nd (diagonals): {get('ex23_case2_d')}")
print(f"\nb vector (RHS):\n{get('ex23_case2_b')}")

# ============================================================
# EXERCISE 2.4
# ============================================================
print("\n" + "=" * 70)
print("EXERCISE 2.4 - After Dirichlet BC application")
print("=" * 70)

print("\nCASE 1: Unit domain")
print("-" * 50)
print(f"B matrix (20x5) after BC:\n{get('ex24_case1_B')}")
print(f"\nd (diagonals): {get('ex24_case1_d')}")
print(f"\nb vector (RHS):\n{get('ex24_case1_b')}")

print("\nCASE 2: Scaled domain")
print("-" * 50)
print(f"B matrix (20x5) after BC:\n{get('ex24_case2_B')}")
print(f"\nd (diagonals): {get('ex24_case2_d')}")
print(f"\nb vector (RHS):\n{get('ex24_case2_b')}")

print("\n" + "=" * 70)
print("END OF VALIDATION DATA")
print("=" * 70)
