import numpy as np
from numba import njit


@njit
def diffusion(h: float, coeff: float = 1.0) -> np.ndarray:
    """
    Element stiffness matrix for diffusion: -coeff * u''

    Weak form: coeff * ∫ u' v' dx
    """
    Ke = np.empty((2, 2))
    c = coeff / h
    Ke[0, 0] = c
    Ke[0, 1] = -c
    Ke[1, 0] = -c
    Ke[1, 1] = c
    return Ke


@njit
def mass(h: float, coeff: float = 1.0) -> np.ndarray:
    """
    Element mass matrix for reaction: coeff * u

    Weak form: coeff * ∫ u v dx
    """
    Ke = np.empty((2, 2))
    c = coeff * h / 6.0
    Ke[0, 0] = 2.0 * c
    Ke[0, 1] = c
    Ke[1, 0] = c
    Ke[1, 1] = 2.0 * c
    return Ke


@njit
def advection(h: float, psi: float) -> np.ndarray:
    """
    Element matrix for advection: psi * u'

    Weak form: psi * ∫ u' v dx
    """
    Ke = np.empty((2, 2))
    c = psi / 2.0
    Ke[0, 0] = -c
    Ke[0, 1] = c
    Ke[1, 0] = -c
    Ke[1, 1] = c
    return Ke
