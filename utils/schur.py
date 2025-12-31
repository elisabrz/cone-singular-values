import numpy as np

def schur(n):
    """
    Generate the Schur cone generators for R^n.
    
    Parameters
    ----------
    n : int
        Dimension
    Returns
    -------
    G : ndarray
        Schur cone generator matrix of shape (n, n-1)
    """
    G = np.zeros((n, n-1))
    for i in range(n-1):
        G[i, i] = 1
        G[i+1, i] = -1
    G = G / np.linalg.norm(G, axis=0, keepdims=True)
    return G