"""
Project on norm1 nonnegative orthant
max_{u>=0,||u||=1} u^T Av
"""

import numpy as np


def projectNonnegOrthnorm1(Av):
    """
    Project on norm1 nonnegative orthant
    
    Parameters
    ----------
    Av : ndarray
        Vector to project
    
    Returns
    -------
    u : ndarray
        Projected vector with ||u||=1 and u>=0
    """
    u = np.maximum(0, -Av)
    
    if np.linalg.norm(u) <= 1e-9:
        b = np.argmax(-Av)
        u[b] = 1
    else:
        u = u / np.linalg.norm(u)
    
    return u