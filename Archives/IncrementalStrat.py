"""
Incremental Submatrix Strategy for E-AO

This module implements an incremental approach to solving the singular value 
problem under conic constraints. It works by first solving on smaller submatrices 
and gradually increasing their size until covering the entire matrix.
"""

import numpy as np
import sys
import os

# Add algorithms directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'algorithms'))

from AlternatingOptimization import AlternatingOptimization


def extend_vector(v, new_dim, extension_method='random', preserve_structure=True):
    """
    Extend a vector to a higher dimension and renormalize.
    
    Parameters:
    -----------
    v : ndarray
        Original vector
    new_dim : int
        New dimension (must be >= len(v))
    extension_method : str
        Method for initializing new components: 'random', 'zero', 'small', 'copy_last'
    preserve_structure : bool
        If True, scale existing components to maintain relative magnitudes
    
    Returns:
    --------
    v_extended : ndarray
        Extended and normalized vector
    """
    if new_dim < len(v):
        raise ValueError("New dimension must be >= current dimension")
    
    if new_dim == len(v):
        return v.copy()
    
    # Create extended vector
    v_extended = np.zeros(new_dim)
    
    if preserve_structure:
        # Keep the structure of the original vector before normalization
        v_extended[:len(v)] = v
    else:
        v_extended[:len(v)] = v
    
    # Initialize new components
    num_new = new_dim - len(v)
    if extension_method == 'random':
        # Use much smaller random values to avoid disrupting the solution
        v_extended[len(v):] = np.random.randn(num_new) * 0.01
    elif extension_method == 'random_uniform':
        v_extended[len(v):] = np.random.uniform(0, 0.01, size=num_new)
    elif extension_method == 'zero':
        v_extended[len(v):] = 0
    elif extension_method == 'small':
        v_extended[len(v):] = 1e-6
    elif extension_method == 'copy_last':
        # Copy the last few components (useful for structured problems)
        if len(v) > 0:
            last_val = v[-1]
            v_extended[len(v):] = last_val * 0.1
    
    # Renormalize
    norm = np.linalg.norm(v_extended)
    if norm > 1e-10:
        v_extended = v_extended / norm
    else:
        # If norm is too small, reinitialize randomly
        v_extended = np.random.randn(new_dim)
        v_extended = v_extended / np.linalg.norm(v_extended)
    
    return v_extended


def get_submatrix_indices(m, n, t, growth_scheme, initial_size=10, step_size=None):
    """
    Determine the submatrix indices at step t according to the growth scheme.
    
    Parameters:
    -----------
    m, n : int
        Dimensions of the full matrix
    t : int
        Current step
    growth_scheme : str
        Growth strategy: 'rows', 'cols', 'both', 'square_first'
    initial_size : int
        Initial submatrix size
    step_size : int or None
        Step size for growth (if None, computed automatically)
    
    Returns:
    --------
    rows_end, cols_end : int
        End indices for rows and columns (exclusive)
    """
    # Adaptive step size: grow more aggressively at the beginning
    if step_size is None:
        # Use smaller steps for more gradual growth
        step_size = max(5, min(m, n) // 20)
    
    if growth_scheme == 'rows':
        # Gradually increase rows, keep all columns
        rows_end = min(initial_size + t * step_size, m)
        cols_end = n
        
    elif growth_scheme == 'cols':
        # Gradually increase columns, keep all rows
        rows_end = m
        cols_end = min(initial_size + t * step_size, n)
        
    elif growth_scheme == 'both':
        # Increase both dimensions proportionally
        ratio = m / n if n > 0 else 1.0
        if ratio > 1:
            # More rows than columns
            rows_end = min(initial_size + int(t * step_size * ratio), m)
            cols_end = min(initial_size + t * step_size, n)
        else:
            # More columns than rows
            rows_end = min(initial_size + t * step_size, m)
            cols_end = min(initial_size + int(t * step_size / ratio), n)
        
    elif growth_scheme == 'square_first':
        # First solve square submatrices, then extend to rectangular
        min_dim = min(m, n)
        square_dim = min(initial_size + t * step_size, min_dim)
        
        if square_dim < min_dim:
            # Still in square phase
            rows_end = square_dim
            cols_end = square_dim
        else:
            # Square phase done, now extend to full dimensions
            extra_steps = t - (min_dim - initial_size) // step_size
            if m > n:
                rows_end = min(min_dim + extra_steps * step_size, m)
                cols_end = n
            else:
                rows_end = m
                cols_end = min(min_dim + extra_steps * step_size, n)
    else:
        raise ValueError(f"Unknown growth scheme: {growth_scheme}")
    
    return rows_end, cols_end


def IncrementalStrategy(A, options, growth_scheme='square_first', 
                       initial_size=10, extension_method='small', 
                       display=False, final_run=True, max_intermediate_iter=100):
    """
    Incremental Submatrix Strategy for E-AO.
    
    Parameters:
    -----------
    A : ndarray
        Matrix (m x n)
    options : dict
        Options for AlternatingOptimization (must include cone specifications)
    growth_scheme : str
        Growth strategy: 'rows', 'cols', 'both', 'square_first'
    initial_size : int
        Initial submatrix size
    extension_method : str
        Method for extending vectors: 'random', 'zero', 'small', 'copy_last'
    display : bool
        Whether to print progress information
    final_run : bool
        Whether to run a final E-AO on the complete matrix (default: True)
    max_intermediate_iter : int
        Maximum iterations for intermediate subproblems (to save time)
    
    Returns:
    --------
    u, v : ndarray
        Approximate solution for the full matrix A
    e : list
        Evolution of objective values (final value if final_run=True)
    """
    m, n = A.shape
    
    # Ensure initial size is valid
    initial_size = min(initial_size, min(m, n))
    
    # Initialize vectors for the initial submatrix
    u_current = np.random.randn(initial_size)
    u_current = u_current / np.linalg.norm(u_current)
    v_current = np.random.randn(initial_size)
    v_current = v_current / np.linalg.norm(v_current)
    
    t = 0
    all_objectives = []
    
    while True:
        # Get current submatrix dimensions
        rows_end, cols_end = get_submatrix_indices(m, n, t, growth_scheme, initial_size)
        
        # Extract submatrix
        A_sub = A[:rows_end, :cols_end]
        
        if display:
            print(f"Step {t}: Submatrix size {rows_end} x {cols_end}")
        
        # Prepare options for current submatrix
        options_sub = options.copy()
        
        # Limit iterations for intermediate steps to save time
        if rows_end < m or cols_end < n:
            options_sub['maxiter'] = max_intermediate_iter
        
        # Handle cone generators if they exist
        if 'G' in options and options['G'] is not None:
            G_full = options['G']
            # Adjust G to match submatrix dimensions
            if G_full.shape[0] >= rows_end:
                options_sub['G'] = G_full[:rows_end, :min(G_full.shape[1], cols_end)]
            else:
                options_sub['G'] = G_full
        
        if 'H' in options and options['H'] is not None:
            H_full = options['H']
            # Adjust H to match submatrix dimensions
            if H_full.shape[0] >= cols_end:
                options_sub['H'] = H_full[:cols_end, :min(H_full.shape[1], cols_end)]
            else:
                options_sub['H'] = H_full
        
        # Set initial vectors for E-AO
        options_sub['u0'] = u_current if len(u_current) == rows_end else extend_vector(u_current, rows_end, extension_method)
        options_sub['v0'] = v_current if len(v_current) == cols_end else extend_vector(v_current, cols_end, extension_method)
        
        # Run E-AO on submatrix
        u_opt, v_opt, e = AlternatingOptimization(A_sub, options_sub)
        
        all_objectives.extend(e)
        
        if display:
            print(f"  Final objective: {e[-1]:.6e}")
        
        # Check if we've reached the full matrix
        if rows_end == m and cols_end == n:
            if display:
                print("Reached full matrix size")
            
            # Optionally run a final E-AO on the full matrix with the warm start
            if final_run:
                if display:
                    print("Running final E-AO on full matrix...")
                
                options_final = options.copy()
                options_final['u0'] = u_opt
                options_final['v0'] = v_opt
                
                u_opt, v_opt, e_final = AlternatingOptimization(A, options_final)
                all_objectives.extend(e_final)
                
                if display:
                    print(f"  Final objective after full run: {e_final[-1]:.6e}")
            
            break
        
        # Extend vectors for next iteration
        next_rows, next_cols = get_submatrix_indices(m, n, t + 1, growth_scheme, initial_size)
        u_current = extend_vector(u_opt, next_rows, extension_method)
        v_current = extend_vector(v_opt, next_cols, extension_method)
        
        t += 1
        
        # Safety check to avoid infinite loops
        if t > 100:
            #print("Warning: Maximum iterations reached")
            break
    
    return u_opt, v_opt, all_objectives


def compare_strategies(A, options, strategies=['square_first', 'rows', 'cols', 'both'],
                       initial_size=10, display=True):
    """
    Compare different growth strategies.
    
    Parameters:
    -----------
    A : ndarray
        Matrix to analyze
    options : dict
        Options for AlternatingOptimization
    strategies : list
        List of growth strategies to compare
    initial_size : int
        Initial submatrix size
    display : bool
        Whether to print results
    
    Returns:
    --------
    results : dict
        Dictionary with results for each strategy
    """
    import time
    
    results = {}
    
    for strategy in strategies:
        if display:
            print(f"\n{'='*60}")
            print(f"Testing strategy: {strategy}")
            print(f"{'='*60}")
        
        start_time = time.time()
        u, v, e = IncrementalStrategy(A, options, growth_scheme=strategy, 
                                      initial_size=initial_size, display=display)
        elapsed_time = time.time() - start_time
        
        final_objective = e[-1]
        
        results[strategy] = {
            'u': u,
            'v': v,
            'objective': final_objective,
            'time': elapsed_time,
            'iterations': len(e)
        }
        
        if display:
            print(f"Final objective: {final_objective:.6e}")
            print(f"Time: {elapsed_time:.3f}s")
            print(f"Total iterations: {len(e)}")
    
    return results