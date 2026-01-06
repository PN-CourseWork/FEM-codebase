import numpy as np


def uniform_mesh(L: float, M: int) -> tuple[np.ndarray, float]:
    """
    Create a uniform 1D mesh on [0, L] with M nodes.

    Parameters
    ----------
    L : float
        Domain length
    M : int
        Number of nodes

    Returns
    -------
    x : ndarray
        Node coordinates (length M)
    h : float
        Element size
    """
    x = np.linspace(0.0, L, M)
    h = L / (M - 1)
    return x, h
