"""
Unified interface for cone projections
"""

import numpy as np
from utils.optimize_cone_generators import optimize_cone_generators
from projections.projectNonnegOrthnorm1 import projectNonnegOrthnorm1
from projections.projectPSDnorm1 import projectPSDnorm1


def update_cone(Av, coneP, G=None, g=None):
    """
    Project vector onto cone
    
    Parameters
    ----------
    Av : ndarray
        Vector to project
    coneP : str
        Cone type: 'generator', 'nonnegort', or 'semidefin'
    G : ndarray, optional
        Generator matrix (required if coneP='generator')
    g : ndarray, optional
        Not used currently
    
    Returns
    -------
    u : ndarray
        Projected unit vector
    """
    
    if coneP == 'generator':
        if G is None:
            raise ValueError("Generator matrix G required for 'generator' cone")
        
        u, _ = optimize_cone_generators(G, Av)
        return u
    
    elif coneP == 'nonnegort':
        u = projectNonnegOrthnorm1(Av)
        return u
    
    elif coneP == 'semidefin':
        # CRITICAL: projectPSDnorm1 expects a MATRIX, not a vector!
        # We need to reshape the vector Av to a matrix Q
        
        # Infer matrix dimension from vector length
        n_squared = len(Av)
        n = int(np.sqrt(n_squared))
        
        if n * n != n_squared:
            raise ValueError(f"len(Av)={n_squared} is not a perfect square for PSD cone")
        
        # Reshape vector to n×n matrix (column-major order like MATLAB)
        Q = Av.reshape((n, n), order='F')
        
        # Project onto PSD cone (Q is now a matrix)
        Qp = projectPSDnorm1(Q)
        
        # Reshape back to vector (column-major order)
        u = Qp.flatten(order='F')
        
        return u
    
    else:
        raise ValueError(f"Unknown cone type: {coneP}")