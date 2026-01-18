# Spectral Element Method Implementation Plan

## Target: FNPF Laplace Problem

Solve ∇²ϕ = 0 with mixed boundary conditions for water wave simulations.

## Stack

- **Mesh generation:** gmsh (quad meshes)
- **Mesh I/O:** meshio
- **Spectral basis:** Reuse from 02689 (`polynomial.py`, `basis.py`, `operators.py`)
- **Linear algebra:** scipy.sparse

## Milestones

### M1: 1D Spectral Foundations ✅
- LGL nodes, quadrature weights, Vandermonde, differentiation
- **Files:** `polynomial.py`, `basis.py` (from 02689)

### M2: Single Element Operators ✅
- 2D tensor-product differentiation O(N³)
- Local mass/stiffness matrices
- **File:** `operators.py` (from 02689)

### M3: Quad Mesh with gmsh/meshio
- Load quad mesh via meshio
- Map gmsh connectivity to SEM node ordering
- Identify boundary nodes/edges by physical groups
- **File:** `mesh.py`

### M4: Global Assembly
- Assemble global stiffness from local element contributions
- C⁰ continuity at element interfaces
- Apply Dirichlet/Neumann BCs
- **File:** `assembly.py`

### M5: Linear SEM Solver
- Solve -∇²u = f on multi-element mesh
- Validate h-convergence O(h^(P+1)) and p-convergence (exponential)
- **File:** `solvers.py`

### M6: Picard Iteration
- Solve nonlinear: -∇·(k(u)∇u) = f
- **File:** `nonlinear.py`

### M7: Anderson Acceleration
- Accelerate Picard convergence
- **File:** `nonlinear.py`

### M8: Nonlinear Multigrid (FAS) — Future
- Accelerate outer Picard loop (requires M6-M7 first)
- **File:** `multigrid.py`

## Work Split

| Milestone | Person A | Person B |
|-----------|----------|----------|
| M3: Mesh (gmsh/meshio) | | ✅ |
| M4: Assembly | Shared | Shared |
| M5: Solver | ✅ | |
| M6: Picard | ✅ | |
| M7: Anderson | ✅ | |
| M8: FAS (future) | | ✅ |

**Person A:** Solvers (M5 → M6 → M7)
**Person B:** Infrastructure (M3 → M4 → M8)

## Success Criteria

- [ ] **M3:** Load gmsh quad mesh, visualize with correct node ordering
- [ ] **M4:** Assembly produces SPD matrix
- [ ] **M5:** h-convergence O(h^(P+1)), p-convergence exponential
- [ ] **M6:** Picard converges for k(u) = 1 + u²
- [ ] **M7:** Anderson reduces iterations ~2-3x vs plain Picard
