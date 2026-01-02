"""
Comparison E-AO vs Incremental Strategy
Problem: Maximum angle between PSD cone and nonnegative symmetric cone

Following Table 7 from the paper:
Tests dimensions n = 20, 30, 40, 50, 60
Compares:
- E-AO with multiple random initializations
- Incremental Strategy with adaptive growth
"""

import numpy as np
import time
import pandas as pd
from AlternatingOptimization import AlternatingOptimization
from IncrementalStratWithSort import IncrementalStrategy
from utils.pi_unit import compute_angle_from_value


def test_eao_psd_vs_nonneg(n, num_init_points=10, max_iter=500, 
                            accuracy=1e-6, verbose=True):
    """
    Test E-AO for PSD vs Nonneg
    
    Parameters
    ----------
    n : int
        Matrix dimension (problem size is n²)
    num_init_points : int
        Number of random initializations
    max_iter : int
        Maximum iterations per run
    accuracy : float
        Convergence tolerance
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        Best angle, average angle, average time
    """
    if verbose:
        print(f"  E-AO: Running {num_init_points} random initializations...")
    
    # Problem setup
    A = np.eye(n * n)  # Identity matrix of size n² × n²
    
    all_angles = []
    all_times = []
    
    for j in range(num_init_points):
        # Random initialization
        v0 = np.random.rand(n * n)
        v0 = v0 / np.linalg.norm(v0)
        
        options = {
            'cone': {'P': 'semidefin', 'Q': 'nonnegort'},
            'maxiter': max_iter,
            'accuracy': accuracy,
            'beta': 0.5,
            'display': 0,
            'v0': v0,
        }
        
        start = time.time()
        u, v, e = AlternatingOptimization(A, options)
        elapsed = time.time() - start
        
        # Convert to angle
        angle = compute_angle_from_value(e[-1])
        
        all_angles.append(angle)
        all_times.append(elapsed)
        
        if verbose and (j + 1) % max(1, num_init_points // 10) == 0:
            print(f"    Progress: {j + 1}/{num_init_points} runs completed...")
    
    # Compute statistics
    best_angle = np.max(all_angles)  # Maximum angle
    avg_angle = np.mean(all_angles)
    avg_time = np.mean(all_times)
    
    return {
        'best_angle': best_angle,
        'avg_angle': avg_angle,
        'avg_time': avg_time,
        'all_angles': all_angles,
        'all_times': all_times,
        'num_runs': num_init_points,
    }


def test_incremental_psd_vs_nonneg(n, max_iter=500, accuracy=1e-6,
                                    growth_scheme='both', verbose=True):
    """
    Test Incremental Strategy for PSD vs Nonneg
    
    Parameters
    ----------
    n : int
        Matrix dimension
    max_iter : int
        Maximum iterations
    accuracy : float
        Convergence tolerance
    growth_scheme : str
        Growth scheme to use
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        Angle, time, iterations
    """
    if verbose:
        print(f"  Incremental: Running with scheme '{growth_scheme}'...")
    
    # Problem setup
    A = np.eye(n * n)
    
    options = {
        'cone': {'P': 'semidefin', 'Q': 'nonnegort'},
        'maxiter': max_iter,
        'accuracy': accuracy,
        'beta': 0.5,
        'display': 0,
    }
    
    try:
        start = time.time()
        
        u, v, objectives_list, final_dims, submatrix_info = IncrementalStrategy(
            A=A,
            options=options,
            growth_scheme=growth_scheme,
            initial_size=10,  # Start small for n²×n² problems
            extension_method='small',
            display=False,
            final_run=True,
            max_intermediate_iter=100,
            use_sorting=False,
        )
        
        elapsed = time.time() - start
        
        # Convert to angle
        angle = compute_angle_from_value(objectives_list[-1])
        
        return {
            'angle': angle,
            'time': elapsed,
            'iterations': len(objectives_list),
            'submatrix_steps': len(submatrix_info),
            'final_dims': final_dims,
        }
        
    except Exception as e:
        if verbose:
            print(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
        return None


def run_comparison_psd_vs_nonneg(dimensions=None, num_init_points=10, 
                                  test_incremental_flag=True,
                                  known_best_values=None,
                                  verbose=True):
    """
    Run comparison across multiple dimensions
    
    Parameters
    ----------
    dimensions : list
        Dimensions to test (default: [20, 30, 40, 50, 60])
    num_init_points : int
        Number of random initializations for E-AO
    test_incremental_flag : bool
        Also test incremental strategy
    known_best_values : dict
        Known best values for reference
    verbose : bool
        Print progress
    
    Returns
    -------
    df : pd.DataFrame
        Results table
    """
    if dimensions is None:
        dimensions = [20, 30, 40, 50, 60]
    
    print("="*70)
    print("COMPARISON: E-AO vs INCREMENTAL STRATEGY")
    print("Problem: PSD cone vs Nonnegative symmetric cone")
    print("="*70)
    print(f"\nDimensions: {dimensions}")
    print(f"E-AO: {num_init_points} random initializations per dimension")
    if test_incremental_flag:
        print(f"Incremental: Single run per dimension")
    print()
    
    all_results = []
    
    for n in dimensions:
        if verbose:
            print(f"\n{'='*70}")
            print(f"n = {n} (problem size: {n*n} × {n*n})")
            print(f"{'='*70}")
        
        # Test E-AO
        eao_results = test_eao_psd_vs_nonneg(
            n,
            num_init_points=num_init_points,
            verbose=verbose
        )
        
        result = {
            'n': n,
            'problem_size': n * n,
            'eao_best_angle': eao_results['best_angle'],
            'eao_avg_angle': eao_results['avg_angle'],
            'eao_avg_time': eao_results['avg_time'],
            'eao_num_runs': eao_results['num_runs'],
        }
        
        # Add known best value if available
        if known_best_values and n in known_best_values:
            result['known_best'] = known_best_values[n]
        
        if verbose:
            print(f"\n  E-AO Results:")
            print(f"    Best angle:  {eao_results['best_angle']:.8f}π")
            print(f"    Avg angle:   {eao_results['avg_angle']:.8f}π")
            print(f"    Avg time:    {eao_results['avg_time']:.3f}s")
            
            if known_best_values and n in known_best_values:
                error = abs(eao_results['best_angle'] - known_best_values[n])
                print(f"    Known best:  {known_best_values[n]:.8f}π")
                print(f"    Error:       {error:.2e}")
        
        # Test Incremental
        if test_incremental_flag:
            inc_results = test_incremental_psd_vs_nonneg(n, verbose=verbose)
            
            if inc_results:
                result.update({
                    'inc_angle': inc_results['angle'],
                    'inc_time': inc_results['time'],
                    'inc_iterations': inc_results['iterations'],
                    'inc_submatrix_steps': inc_results['submatrix_steps'],
                })
                
                if verbose:
                    print(f"\n  Incremental Results:")
                    print(f"    Angle:      {inc_results['angle']:.8f}π")
                    print(f"    Time:       {inc_results['time']:.3f}s")
                    print(f"    Iterations: {inc_results['iterations']}")
                    print(f"    Steps:      {inc_results['submatrix_steps']}")
                    
                    # Speedup
                    if inc_results['time'] > 0:
                        speedup = eao_results['avg_time'] / inc_results['time']
                        print(f"    Speedup vs E-AO avg: {speedup:.1f}x")
                    
                    # Compare quality
                    if known_best_values and n in known_best_values:
                        inc_error = abs(inc_results['angle'] - known_best_values[n])
                        print(f"    Error:      {inc_error:.2e}")
        
        all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    # Header
    header = f"\n{'n':>5} {'Size':>8} {'Known Best':>12} {'E-AO Best':>12} {'E-AO Avg':>12} {'Time':>8}"
    if test_incremental_flag:
        header += f" {'Incr':>12} {'Time':>8} {'Speedup':>8}"
    print(header)
    print("-"*70)
    
    # Rows
    for _, row in df.iterrows():
        line = f"{row['n']:>5} {row['problem_size']:>8}"
        
        # Known best
        if 'known_best' in row and row['known_best'] is not None:
            line += f" {row['known_best']:>11.8f}π"
        else:
            line += f" {'N/A':>12}"
        
        # E-AO
        line += f" {row['eao_best_angle']:>11.8f}π"
        line += f" {row['eao_avg_angle']:>11.8f}π"
        line += f" {row['eao_avg_time']:>7.3f}s"
        
        # Incremental
        if test_incremental_flag and 'inc_angle' in row and row['inc_angle'] is not None:
            line += f" {row['inc_angle']:>11.8f}π"
            line += f" {row['inc_time']:>7.3f}s"
            
            if row['inc_time'] > 0:
                speedup = row['eao_avg_time'] / row['inc_time']
                line += f" {speedup:>7.1f}x"
            else:
                line += f" {'>1000':>8s}x"
        
        print(line)
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    print(f"\nE-AO:")
    print(f"  Average best angle: {df['eao_best_angle'].mean():.8f}π")
    print(f"  Average avg angle:  {df['eao_avg_angle'].mean():.8f}π")
    print(f"  Total time (avg):   {df['eao_avg_time'].sum():.1f}s")
    
    if test_incremental_flag and 'inc_angle' in df.columns:
        print(f"\nIncremental:")
        print(f"  Average angle:      {df['inc_angle'].mean():.8f}π")
        print(f"  Total time:         {df['inc_time'].sum():.1f}s")
        
        # Average speedup
        valid_speedups = []
        for _, row in df.iterrows():
            if 'inc_time' in row and row['inc_time'] > 0:
                valid_speedups.append(row['eao_avg_time'] / row['inc_time'])
        
        if valid_speedups:
            print(f"  Average speedup:    {np.mean(valid_speedups):.1f}x")
    
    print("\n" + "="*70)
    
    return df


if __name__ == "__main__":
    import sys
    
    # Known best values from Table 7 (if available)
    # Format: {n: best_angle_in_pi}
    known_best_values = {
        # Add known values here if you have them
        # 20: 0.XXXXXXXX,
        # 30: 0.XXXXXXXX,
        # etc.
    }
    
    # Parse arguments
    if len(sys.argv) > 1:
        # Test single dimension
        n = int(sys.argv[1])
        dimensions = [n]
        num_init_points = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    else:
        # Test default range
        dimensions = [20, 30, 40, 50, 60]
        num_init_points = 10  # Use 10000 for paper comparison
    
    # Run comparison
    df_results = run_comparison_psd_vs_nonneg(
        dimensions=dimensions,
        num_init_points=num_init_points,
        test_incremental_flag=True,
        known_best_values=known_best_values,
        verbose=True
    )
    
    # Save results
    output_file = 'comparison_psd_vs_nonneg.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")