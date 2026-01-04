"""Lid-driven cavity solver framework.

This module provides solvers for comparing finite volume and spectral methods.

Solver Hierarchy:
-----------------
LidDrivenCavitySolver (abstract base - defines problem)
├── FVSolver (finite volume with SIMPLE algorithm)
└── SpectralSolver (unified spectral with optional multigrid)
    - n_levels=1: Single-grid mode
    - n_levels>1: FSG multigrid mode
"""

from .base import LidDrivenCavitySolver
from .datastructures import (
    # Base classes (shared by all solvers)
    Parameters,
    Metrics,
    Fields,
    TimeSeries,
    # FV-specific
    FVParameters,
    FVSolverFields,
    # Spectral-specific
    SpectralParameters,
    SpectralSolverFields,
)
from solvers.fv.solver import FVSolver
from solvers.spectral import SpectralSolver

# Legacy aliases for backward compatibility
SGSolver = SpectralSolver
FSGSolver = SpectralSolver


__all__ = [
    # Base solver
    "LidDrivenCavitySolver",
    # Shared data structures
    "Parameters",
    "Metrics",
    "Fields",
    "TimeSeries",
    # FV solver
    "FVSolver",
    "FVParameters",
    "FVSolverFields",
    # Spectral solver (unified)
    "SpectralSolver",
    "SpectralParameters",
    "SpectralSolverFields",
    # Legacy aliases
    "SGSolver",
    "FSGSolver",
]
