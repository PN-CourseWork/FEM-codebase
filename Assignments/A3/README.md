# Assignment 3: Spectral Element Method

This assignment demonstrates the implementation and convergence properties of the
Spectral Element Method (SEM) for solving the 2D Poisson equation on quad meshes.

## Problem

Solve the Poisson equation with Dirichlet boundary conditions:

```
-∇²u = f   on Ω = [0,1]²
 u = g     on ∂Ω
```

Using manufactured solution: `u = sin(πx)sin(πy)` with homogeneous BCs.

## Implementation

The SEM implementation uses:
- **Mesh generation**: gmsh for creating structured quad meshes
- **Basis functions**: Legendre-Gauss-Lobatto (LGL) nodal basis
- **Quadrature**: LGL quadrature (diagonal mass matrix)
- **C⁰ continuity**: Shared DOFs at element interfaces

Key modules:
- `FEM.spectral.mesh` - Mesh loading and DOF mapping
- `FEM.spectral.assembly` - Global matrix assembly (COO→CSR)
- `FEM.spectral.solvers` - Poisson solver and error computation

## Running the Convergence Study

```bash
uv run python assignments/A3/sem_convergence_study.py
```

## Results

### h-Convergence

For polynomial order p, the SEM achieves O(h^{p+2}) convergence for smooth solutions:

| p | Expected Rate | Observed Rate |
|---|---------------|---------------|
| 2 | O(h⁴)        | 4.00          |
| 3 | O(h⁵)        | 5.00          |
| 4 | O(h⁶)        | 6.00          |
| 5 | O(h⁷)        | 7.00          |
| 6 | O(h⁸)        | 8.00          |

### p-Convergence (Spectral Accuracy)

For smooth solutions, increasing polynomial order yields exponential convergence.
On a 2×2 mesh:

| p | L² Error   | L∞ Error   |
|---|------------|------------|
| 2 | 5.69e-03   | 1.63e-02   |
| 4 | 1.20e-05   | 2.43e-05   |
| 6 | 1.91e-08   | 4.22e-08   |
| 8 | 2.29e-11   | 5.02e-11   |

## Generated Figures

Figures are saved to `figures/A3/` in the project root:

- `h_convergence.pdf` - L² and L∞ error vs mesh size for different p
- `p_convergence.pdf` - Error vs polynomial order (spectral accuracy)
- `error_vs_dof.pdf` - Efficiency plot: error vs degrees of freedom

## Key Observations

1. **Optimal h-convergence**: The method achieves the theoretical O(h^{p+2}) rate
   for the L² error on smooth problems.

2. **Spectral accuracy**: For fixed mesh, increasing p gives exponential error
   decay until machine precision is reached (~10^{-13}).

3. **Efficiency**: Higher p is more efficient than h-refinement for smooth
   solutions (fewer DOFs for same accuracy).

4. **Machine precision plateau**: At high p or fine meshes, errors plateau at
   ~10^{-13} due to floating-point limits.
