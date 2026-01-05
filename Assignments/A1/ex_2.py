"""
Exercise 1.2 - 1D Finite Element Method for Boundary Value Problem
02623 The Finite Element Method for PDEs

Solves: u'' - u = 0, 0 <= x <= L
With:   u(0) = c, u(L) = d

Authors: Philip Korsager Nickel (s214960), Aske Funch Schrøder Nielsen (s224409)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


plt.style.use("src/fem.mplstyle")

# Figures saved to figures/A1/ (run from repo root)
FIGURE_DIR = Path("figures/A1")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)



# Problem parameters
print("=" * 60)
print("Exercise 1.2 - 1D FEM Solver")
print("=" * 60)

# ========================================
# Part a) Non-uniform mesh
# ========================================
print("\n--- Part a) Non-uniform mesh ---")
x_test = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.9, 1.4, 1.5, 1.8, 1.9, 2.0])



# ========================================
# Part b) Uniform mesh
# ========================================

# ========================================
# Part c) Validation plots
# ========================================


# ========================================
# Part d) Convergence study
# ========================================

