import numpy as np

def nonneg(n):
    """
    Generate the nonnegative orthant cone generators for R^n.
    
    Parameters
    ----------
    n : int
        Dimension
    Returns
    -------
    G : ndarray
        Nonnegative orthant generator matrix of shape (n, n) - identity matrix
    """
    G = np.eye(n)
    return G