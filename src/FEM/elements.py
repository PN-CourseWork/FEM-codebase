import numpy as np


def element_diffusion(h: float, coeff: float = 1.0) -> np.ndarray:
    """
    Element stiffness matrix for diffusion term: -coeff * u''

    Weak form contribution: coeff * ∫ u' v' dx

    Parameters
    ----------
    h : float
        Element size
    coeff : float
        Diffusion coefficient (default 1.0)

    Returns
    -------
    Ke : ndarray (2, 2)
        Element stiffness matrix
    """
    return coeff / h * np.array([[1, -1], [-1, 1]])


def element_mass(h: float, coeff: float = 1.0) -> np.ndarray:
    """
    Element mass matrix for reaction term: coeff * u

    Weak form contribution: coeff * ∫ u v dx

    Parameters
    ----------
    h : float
        Element size
    coeff : float
        Reaction coefficient (default 1.0)

    Returns
    -------
    Me : ndarray (2, 2)
        Element mass matrix
    """
    return coeff * h / 6 * np.array([[2, 1], [1, 2]])


def element_advection(psi: float) -> np.ndarray:
    """
    Element matrix for advection term: psi * u'

    Weak form contribution: psi * ∫ u' v dx

    Parameters
    ----------
    psi : float
        Advection velocity

    Returns
    -------
    Ce : ndarray (2, 2)
        Element advection matrix
    """
    return psi / 2 * np.array([[-1, 1], [-1, 1]])


def element_load(h: float) -> np.ndarray:
    """
    Element load vector for constant source f=1.

    Weak form contribution: ∫ f v dx = ∫ v dx

    Parameters
    ----------
    h : float
        Element size

    Returns
    -------
    fe : ndarray (2,)
        Element load vector
    """
    return h / 2 * np.array([1, 1])


def element_advection_diffusion(h: float, eps: float, psi: float) -> np.ndarray:
    """
    Combined element matrix for advection-diffusion: -eps * u'' + psi * u'

    Parameters
    ----------
    h : float
        Element size
    eps : float
        Diffusion coefficient
    psi : float
        Advection velocity

    Returns
    -------
    Ke : ndarray (2, 2)
        Element matrix
    """
    return element_diffusion(h, eps) + element_advection(psi)


def element_diffusion_reaction(h: float) -> np.ndarray:
    """
    Combined element matrix for diffusion-reaction: u'' - u = 0

    Weak form: ∫ u' v' dx + ∫ u v dx = 0

    Parameters
    ----------
    h : float
        Element size

    Returns
    -------
    Ke : ndarray (2, 2)
        Element matrix
    """
    return element_diffusion(h) + element_mass(h)
