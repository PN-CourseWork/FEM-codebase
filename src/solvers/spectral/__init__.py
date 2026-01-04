"""Spectral solver package."""

from solvers.spectral.solver import SpectralSolver

# Legacy aliases for backward compatibility
SGSolver = SpectralSolver
FSGSolver = SpectralSolver

__all__ = ["SpectralSolver", "SGSolver", "FSGSolver"]
