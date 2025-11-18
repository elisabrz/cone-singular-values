# Project the matrix Q onto the PSD cone 
# 
# This requires an eigendecomposition and then setting the negative
# eigenvalues to zero, 
# or all eigenvalues in the interval [epsilon,delta] if specified. 


import numpy as np

def projectPSDnorm1(Q):
    n = Q.shape[0]
    if Q.size == 0:
        Qp = Q
        return Qp
    # Symmetrize Q
    Q = 0.5 * (Q + Q.T)
    # Check for NaN or Inf entries
    if np.isnan(Q).max() == 1 or np.isinf(Q).max() == 1:
        raise ValueError('Input matrix has infinite or NaN entries')
    # Eigendecomposition
    e_vals, V = np.linalg.eigh(Q)
    # Project eigenvalues onto nonnegative orthant and normalize
    e = projectNonnegOrthnorm1(e_vals)
    Qp = V @ np.diag(e) @ V.T
    return Qp

