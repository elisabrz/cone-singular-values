"""
update_cone.py - Unified interface for cone projections

This module provides a unified interface for updating vectors within different cones.
For generator cones, it calls optimize_cone_generators with the appropriate sign.
"""

import numpy as np
from utils.optimize_cone_generators import optimize_cone_generators
from projections.projectNonnegOrthnorm1 import projectNonnegOrthnorm1
from projections.projectPSDnorm1 import projectPSDnorm1


def update_cone(Av, coneP, G=None, g=None):
    """
    Update cone projection - unified interface
    
    For angle MAXIMIZATION problems (like Schur vs R+):
    - We want to MAXIMIZE <u, Av>
    - But optimize_cone_generators does MINIMIZATION
    - Solution: Pass -Av to get max instead of min
    
    Parameters
    ----------
    Av : ndarray
        Vector to project
    coneP : str
        Type of cone: 'generator', 'nonnegort', 'semidefin'
    G : ndarray, optional
        Generator matrix (required if coneP='generator')
    g : ndarray, optional
        Additional parameter (not used currently)
    
    Returns
    -------
    u : ndarray
        Projected vector
    """
    
    if coneP == 'generator':
        if G is None:
            raise ValueError("Generator matrix G is required for 'generator' cone")
        
        u, info = optimize_cone_generators(G, Av)  
        return u
    
    elif coneP == 'nonnegort':
        # Nonnegative orthant with norm 1
        u = projectNonnegOrthnorm1(Av)
        return u
    
    elif coneP == 'semidefin':
        # PSD cone with norm 1
        u = projectPSDnorm1(Av)
        return u
    
    else:
        raise ValueError(f"Unknown cone type: {coneP}")