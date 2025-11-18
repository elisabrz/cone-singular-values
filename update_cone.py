"""
Solve min_{u in P, ||u||=1} <u, Av>

coneP is the type of cone:
1) 'generator': P = { u | u = G*x, x >= 0}
              In that case, G provides the generators
2) 'nonnegort': P = { x | x >= 0}
3) 'semidefin': P = { x | mat(x) is PSD}
4) 'facetsrep': P = { x | G*x >= g}


Pour le cas de la Figure 2, on utilise 'generator' avec G la matrice de Schur et 'nonnegort' pour l'autre cône (orthant positif).
"""

import numpy as np
from optimize_cone_generators import optimize_cone_generators
from optimize_cone_facets import optimize_cone_facets
from projectNonnegOrthnorm1 import projectNonnegOrthnorm1
from projectPSDnorm1 import projectPSDnorm1
from vec import vec


def update_cone(Av, coneP, G=None, g=None):
    """
    Update cone projection
    
    Parameters
    ----------
    Av : ndarray
        Vector to project
    coneP : str
        Type of cone ('generator', 'facetsrep', 'nonnegort', 'semidefin')
    G : ndarray, optional
        Generator matrix (for 'generator' and 'facetsrep')
    g : ndarray, optional
        Offset vector (for 'facetsrep')
    
    Returns
    -------
    u : ndarray
        Projected vector
    """
    if coneP == 'generator':
        # Solve max u^T*Av such that ||u||=1, u=G*x, x>= 0
        u = optimize_cone_generators(G, Av)
    elif coneP == 'facetsrep':
        # Solve max u^T*Av such that ||u||=1, G*u >= 0
        u = optimize_cone_facets(G, g, Av)
    elif coneP == 'nonnegort':
        u = projectNonnegOrthnorm1(Av)
    elif coneP == 'semidefin':
        n = len(Av)
        u = vec(projectPSDnorm1(np.reshape(Av, (int(np.sqrt(n)), int(np.sqrt(n))))))
    
    return u