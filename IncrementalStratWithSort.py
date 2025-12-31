"""
Incremental Submatrix Strategy for E-AO with Graph Sorting

This module implements an incremental approach to solving the singular value 
problem under conic constraints. It works by first solving on smaller submatrices 
and gradually increasing their size until covering the entire matrix.

Includes optional preprocessing that sorts the bipartite graph by vertex degrees
to concentrate high-degree vertices in the top-left corner.
"""

import numpy as np
import sys
import os

# Add algorithms directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'algorithms'))

from AlternatingOptimization import AlternatingOptimization


def sort_bipartite_graph(Adj, return_permutations=False):
    """
    Sort bipartite graph adjacency matrix by vertex degrees.
    High-degree vertices are placed in the top-left corner.
    
    Parameters:
    -----------
    Adj : ndarray (m x n)
        Bipartite graph adjacency matrix (0s and 1s)
    return_permutations : bool
        If True, also return the permutation indices used
    
    Returns:
    --------
    Adj_sorted : ndarray (m x n)
        Sorted adjacency matrix
    row_perm : ndarray (optional)
        Row permutation indices (original -> sorted)
    col_perm : ndarray (optional)
        Column permutation indices (original -> sorted)
    """
    m, n = Adj.shape
    
    # Calculate degrees (number of edges for each vertex)
    row_degrees = np.sum(Adj, axis=1)  # Degree of each row vertex
    col_degrees = np.sum(Adj, axis=0)  # Degree of each column vertex
    
    # Sort indices by degree (descending order - highest degree first)
    row_perm = np.argsort(-row_degrees)  # Negative for descending
    col_perm = np.argsort(-col_degrees)
    
    # Apply permutations to get sorted matrix
    Adj_sorted = Adj[row_perm, :]
    Adj_sorted = Adj_sorted[:, col_perm]
    
    if return_permutations:
        return Adj_sorted, row_perm, col_perm
    else:
        return Adj_sorted


def unsort_vectors(u, v, row_perm, col_perm):
    """
    Reverse the sorting permutation to get vectors in original order.
    
    Parameters:
    -----------
    u : ndarray
        Vector corresponding to sorted rows
    v : ndarray
        Vector corresponding to sorted columns
    row_perm : ndarray
        Row permutation used for sorting
    col_perm : ndarray
        Column permutation used for sorting
    
    Returns:
    --------
    u_orig : ndarray
        Vector in original row order
    v_orig : ndarray
        Vector in original column order
    """
    # Check dimension compatibility
    if len(u) != len(row_perm):
        raise ValueError(f"Dimension mismatch: u has length {len(u)} but row_perm has length {len(row_perm)}")
    if len(v) != len(col_perm):
        raise ValueError(f"Dimension mismatch: v has length {len(v)} but col_perm has length {len(col_perm)}")
    
    # Create inverse permutations
    row_inv = np.argsort(row_perm)
    col_inv = np.argsort(col_perm)
    
    # Apply inverse permutations
    u_orig = u[row_inv]
    v_orig = v[col_inv]
    
    return u_orig, v_orig


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
        Method for initializing new components: 'random', 'random_uniform', 
        'zero', 'small', 'copy_last'
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
        v_extended[:len(v)] = v
    else:
        v_extended[:len(v)] = v
    
    # Initialize new components
    num_new = new_dim - len(v)
    
    if extension_method == 'random':
        # Adaptive scaling: smaller perturbation for large extensions
        scale = 0.01 / np.sqrt(1 + num_new / len(v))
        v_extended[len(v):] = np.random.randn(num_new) * scale
    elif extension_method == 'random_uniform':
        scale = 0.01 / np.sqrt(1 + num_new / len(v))
        v_extended[len(v):] = np.random.uniform(-scale, scale, num_new)
    elif extension_method == 'zero':
        v_extended[len(v):] = 0
    elif extension_method == 'small':
        v_extended[len(v):] = 1e-6
    elif extension_method == 'copy_last':
        if len(v) > 0:
            # Use weighted average of last few components
            last_vals = v[-min(3, len(v)):]
            avg_val = np.mean(np.abs(last_vals))
            v_extended[len(v):] = np.random.randn(num_new) * avg_val * 0.1
    
    # Renormalize
    norm = np.linalg.norm(v_extended)
    if norm > 1e-10:
        v_extended = v_extended / norm
    else:
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
    if step_size is None:
        step_size = max(5, min(m, n) // 20)
    
    if growth_scheme == 'rows':
        rows_end = min(initial_size + t * step_size, m)
        cols_end = n
        
    elif growth_scheme == 'cols':
        rows_end = m
        cols_end = min(initial_size + t * step_size, n)
        
    elif growth_scheme == 'both':
        ratio = m / n if n > 0 else 1.0
        if ratio > 1:
            rows_end = min(initial_size + int(t * step_size * ratio), m)
            cols_end = min(initial_size + t * step_size, n)
        else:
            rows_end = min(initial_size + t * step_size, m)
            cols_end = min(initial_size + int(t * step_size / ratio), n)
        
    elif growth_scheme == 'square_first':
        min_dim = min(m, n)
        square_dim = min(initial_size + t * step_size, min_dim)
        
        if square_dim < min_dim:
            rows_end = square_dim
            cols_end = square_dim
        else:
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


"""
Incremental Submatrix Strategy for E-AO with Graph Sorting

This module implements an incremental approach to solving the singular value 
problem under conic constraints. It works by first solving on smaller submatrices 
and gradually increasing their size until covering the entire matrix.

Includes optional preprocessing that sorts the bipartite graph by vertex degrees
to concentrate high-degree vertices in the top-left corner.
"""

import numpy as np
import sys
import os

# Add algorithms directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'algorithms'))

from AlternatingOptimization import AlternatingOptimization


def sort_bipartite_graph(Adj, return_permutations=False):
    """
    Sort bipartite graph adjacency matrix by vertex degrees.
    High-degree vertices are placed in the top-left corner.
    
    Parameters:
    -----------
    Adj : ndarray (m x n)
        Bipartite graph adjacency matrix (0s and 1s)
    return_permutations : bool
        If True, also return the permutation indices used
    
    Returns:
    --------
    Adj_sorted : ndarray (m x n)
        Sorted adjacency matrix
    row_perm : ndarray (optional)
        Row permutation indices (original -> sorted)
    col_perm : ndarray (optional)
        Column permutation indices (original -> sorted)
    """
    m, n = Adj.shape
    
    # Calculate degrees (number of edges for each vertex)
    row_degrees = np.sum(Adj, axis=1)  # Degree of each row vertex
    col_degrees = np.sum(Adj, axis=0)  # Degree of each column vertex
    
    # Sort indices by degree (descending order - highest degree first)
    row_perm = np.argsort(-row_degrees)  # Negative for descending
    col_perm = np.argsort(-col_degrees)
    
    # Apply permutations to get sorted matrix
    Adj_sorted = Adj[row_perm, :]
    Adj_sorted = Adj_sorted[:, col_perm]
    
    if return_permutations:
        return Adj_sorted, row_perm, col_perm
    else:
        return Adj_sorted


def unsort_vectors(u, v, row_perm, col_perm):
    """
    Reverse the sorting permutation to get vectors in original order.
    
    Parameters:
    -----------
    u : ndarray
        Vector corresponding to sorted rows
    v : ndarray
        Vector corresponding to sorted columns
    row_perm : ndarray
        Row permutation used for sorting
    col_perm : ndarray
        Column permutation used for sorting
    
    Returns:
    --------
    u_orig : ndarray
        Vector in original row order
    v_orig : ndarray
        Vector in original column order
    """
    # Check dimension compatibility
    if len(u) != len(row_perm):
        raise ValueError(f"Dimension mismatch: u has length {len(u)} but row_perm has length {len(row_perm)}")
    if len(v) != len(col_perm):
        raise ValueError(f"Dimension mismatch: v has length {len(v)} but col_perm has length {len(col_perm)}")
    
    # Create inverse permutations
    row_inv = np.argsort(row_perm)
    col_inv = np.argsort(col_perm)
    
    # Apply inverse permutations
    u_orig = u[row_inv]
    v_orig = v[col_inv]
    
    return u_orig, v_orig


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
        Method for initializing new components: 'random', 'random_uniform', 
        'zero', 'small', 'copy_last'
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
        v_extended[:len(v)] = v
    else:
        v_extended[:len(v)] = v
    
    # Initialize new components
    num_new = new_dim - len(v)
    if extension_method == 'random':
        v_extended[len(v):] = np.random.randn(num_new) * 0.01
    elif extension_method == 'random_uniform':
        v_extended[len(v):] = np.random.uniform(0, 0.01, num_new)
    elif extension_method == 'zero':
        v_extended[len(v):] = 0
    elif extension_method == 'small':
        v_extended[len(v):] = 1e-6
    elif extension_method == 'copy_last':
        if len(v) > 0:
            last_val = v[-1]
            v_extended[len(v):] = last_val * 0.1
    
    # Renormalize
    norm = np.linalg.norm(v_extended)
    if norm > 1e-10:
        v_extended = v_extended / norm
    else:
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
    initial_size : int or tuple
        Initial submatrix size. Can be:
        - int: square initial submatrix
        - tuple (m0, n0): rectangular initial submatrix
    step_size : int or None
        Step size for growth (if None, computed automatically)
    
    Returns:
    --------
    rows_end, cols_end : int
        End indices for rows and columns (exclusive)
    """
    # Handle initial_size as int or tuple
    if isinstance(initial_size, tuple):
        initial_m, initial_n = initial_size
    else:
        initial_m = initial_size
        initial_n = initial_size
    
    if step_size is None:
        step_size = max(5, min(m, n) // 20)
    
    if growth_scheme == 'rows':
        rows_end = min(initial_m + t * step_size, m)
        cols_end = n
        
    elif growth_scheme == 'cols':
        rows_end = m
        cols_end = min(initial_n + t * step_size, n)
        
    elif growth_scheme == 'both':
        ratio = m / n if n > 0 else 1.0
        if ratio > 1:
            rows_end = min(initial_m + int(t * step_size * ratio), m)
            cols_end = min(initial_n + t * step_size, n)
        else:
            rows_end = min(initial_m + t * step_size, m)
            cols_end = min(initial_n + int(t * step_size / ratio), n)
        
    elif growth_scheme == 'square_first':
        min_dim = min(m, n)
        min_initial = min(initial_m, initial_n)
        square_dim = min(min_initial + t * step_size, min_dim)
        
        if square_dim < min_dim:
            rows_end = square_dim
            cols_end = square_dim
        else:
            extra_steps = t - (min_dim - min_initial) // step_size
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
                       initial_size=10, extension_method='random', 
                       display=False, final_run=True, max_intermediate_iter=100,
                       use_sorting=False, Adj_original=None):
    """
    Incremental Submatrix Strategy for E-AO with optional graph sorting.
    
    Parameters:
    -----------
    A : ndarray
        Matrix (m x n) - can be the transformed matrix for optimization
    options : dict
        Options for AlternatingOptimization (must include cone specifications)
    growth_scheme : str
        Growth strategy: 'rows', 'cols', 'both', 'square_first'
    initial_size : int or tuple
        Initial submatrix size. Can be:
        - int: square initial submatrix of size initial_size x initial_size
        - tuple (m0, n0): rectangular initial submatrix of size m0 x n0
    extension_method : str
        Method for extending vectors: 'random', 'random_uniform', 'zero', 'small', 'copy_last'
    display : bool
        Whether to print progress information
    final_run : bool
        Whether to run a final E-AO on the complete matrix (default: True)
    max_intermediate_iter : int
        Maximum iterations for intermediate subproblems (to save time)
    use_sorting : bool
        Whether to apply degree-based sorting preprocessing
    Adj_original : ndarray or None
        Original adjacency matrix (required if use_sorting=True)
    
    Returns:
    --------
    u, v : ndarray
        Approximate solution for the full matrix A (in original order if sorted)
    e : list
        Evolution of objective values (final value if final_run=True)
    final_dims : tuple
        Final dimensions (m, n) reached by the incremental strategy
    submatrix_info : list of dict
        Information about each submatrix step: {'dims': (m,n), 'iterations': int}
    """
    # Apply sorting if requested
    if use_sorting:
        if Adj_original is None:
            raise ValueError("Adj_original must be provided when use_sorting=True")
        
        if display:
            print("Sorting graph by vertex degrees...")
        
        Adj_sorted, row_perm, col_perm = sort_bipartite_graph(Adj_original, 
                                                               return_permutations=True)
        
        # Transform sorted adjacency matrix
        n_sorted = max(Adj_sorted.shape)
        A_sorted = -Adj_sorted + (1 - Adj_sorted) * n_sorted
        A_work = A_sorted
        
        if display:
            # Handle both int and tuple initial_size for display
            if isinstance(initial_size, tuple):
                init_m, init_n = initial_size
                orig_top_left = np.sum(Adj_original[:init_m, :init_n])
                sorted_top_left = np.sum(Adj_sorted[:init_m, :init_n])
                print(f"  Edges in top-left {init_m}x{init_n}:")
            else:
                orig_top_left = np.sum(Adj_original[:initial_size, :initial_size])
                sorted_top_left = np.sum(Adj_sorted[:initial_size, :initial_size])
                print(f"  Edges in top-left {initial_size}x{initial_size}:")
            print(f"    Original: {orig_top_left}")
            print(f"    Sorted:   {sorted_top_left} (+{sorted_top_left - orig_top_left})")
    else:
        A_work = A
        row_perm = None
        col_perm = None
    
    m, n = A_work.shape
    
    # Handle initial_size as either int or tuple
    if isinstance(initial_size, tuple):
        initial_m, initial_n = initial_size
        # Ensure initial sizes are valid
        initial_m = min(initial_m, m)
        initial_n = min(initial_n, n)
    else:
        # Ensure initial size is valid
        initial_size = min(initial_size, min(m, n))
        initial_m = initial_size
        initial_n = initial_size
    
    # For square_first on highly rectangular matrices, adjust initial size
    # to ensure we start with a square that fits within both dimensions
    if growth_scheme == 'square_first' and initial_m != initial_n:
        # Start with a square using the smaller of the two initial dimensions
        square_initial = min(initial_m, initial_n)
        if display:
            print(f"Adjusting initial_size for square_first: ({initial_m}, {initial_n}) -> {square_initial}x{square_initial}")
        initial_m = square_initial
        initial_n = square_initial
    
    # Initialize vectors for the initial submatrix
    u_current = np.random.randn(initial_m)
    u_current = u_current / np.linalg.norm(u_current)
    v_current = np.random.randn(initial_n)
    v_current = v_current / np.linalg.norm(v_current)
    
    t = 0
    all_objectives = []
    submatrix_info = []  # Track info for each submatrix step
    
    while True:
        # Get current submatrix dimensions
        # Pass the actual initial dimensions to get_submatrix_indices
        rows_end, cols_end = get_submatrix_indices(m, n, t, growth_scheme, 
                                                    initial_size=(initial_m, initial_n))
        
        # Extract submatrix
        A_sub = A_work[:rows_end, :cols_end]
        
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
            if G_full.shape[0] >= rows_end:
                options_sub['G'] = G_full[:rows_end, :min(G_full.shape[1], cols_end)]
            else:
                options_sub['G'] = G_full
        
        if 'H' in options and options['H'] is not None:
            H_full = options['H']
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
        
        # Record submatrix information
        submatrix_info.append({
            'dims': (rows_end, cols_end),
            'iterations': len(e),
            'objective': e[-1]
        })
        
        if display:
            print(f"  Final objective: {e[-1]:.6e}, E-AO iterations: {len(e)}")
        
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
                
                u_opt, v_opt, e_final = AlternatingOptimization(A_work, options_final)
                all_objectives.extend(e_final)
                
                # Record final run
                submatrix_info.append({
                    'dims': (m, n),
                    'iterations': len(e_final),
                    'objective': e_final[-1],
                    'is_final_run': True
                })
                
                if display:
                    print(f"  Final objective after full run: {e_final[-1]:.6e}, E-AO iterations: {len(e_final)}")
            
            break
        
        # Extend vectors for next iteration
        next_rows, next_cols = get_submatrix_indices(m, n, t + 1, growth_scheme, 
                                                      initial_size=(initial_m, initial_n))
        u_current = extend_vector(u_opt, next_rows, extension_method)
        v_current = extend_vector(v_opt, next_cols, extension_method)
        
        t += 1
        
        # Safety check to avoid infinite loops
        if t > 100:
            break
    
    # If sorting was used, reverse the permutation
    if use_sorting:
        # Only unsort if dimensions match (handles rectangular matrices correctly)
        if len(u_opt) == len(row_perm) and len(v_opt) == len(col_perm):
            u_opt, v_opt = unsort_vectors(u_opt, v_opt, row_perm, col_perm)
            if display:
                print("Vectors converted back to original order")
        else:
            if display:
                print(f"Warning: Cannot unsort - dimension mismatch")
                print(f"  u: {len(u_opt)} vs row_perm: {len(row_perm)}")
                print(f"  v: {len(v_opt)} vs col_perm: {len(col_perm)}")
                print("  Returning vectors in sorted order")
    
    # Get actual dimensions reached (in original space)
    final_dims = (len(u_opt), len(v_opt))
    
    return u_opt, v_opt, all_objectives, final_dims, submatrix_info


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