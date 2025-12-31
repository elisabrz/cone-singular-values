"""
Optimized Comparison E-AO vs Incremental
With adaptive early stopping for E-AO
"""

import numpy as np
import time
from scipy.linalg import dft
from AlternatingOptimization import AlternatingOptimization
from IncrementalStratWithSort import IncrementalStrategy
from utils.pi_unit import compute_angle_from_value
import pandas as pd


def setup_circulant_problem(n):
    """Setup circulant PSD vs nonnegative problem"""
    N = n
    if N % 2 == 0:
        raise ValueError("N must be odd")
    
    m = (N - 1) // 2
    Aux = dft(N, scale='sqrtn')
    Aux = np.real(Aux)
    A = 2 * Aux[1:m+1, 1:m+1]
    G = np.eye(m)
    H = np.eye(m)
    
    return A, G, H, m


def test_eao_adaptive(A, m, timelimit=10, max_init_points=100, 
                     min_init_points=10, check_interval=10,
                     convergence_tol=1e-6, max_iter=500, 
                     accuracy=1e-10, verbose=True):
    """
    E-AO with ADAPTIVE early stopping
    
    Stops early if std(angles) < convergence_tol after min_init_points runs
    """
    tol = 1e-8
    all_best_values = []
    all_best_angles = []
    
    if verbose:
        print(f"  E-AO: Adaptive multi-restart (max {max_init_points}, stop if std < {convergence_tol:.2e})...")
    
    for j in range(max_init_points):
        emin = 0
        T = 0
        
        while T < timelimit:
            v0 = np.random.rand(m)
            v0 = v0 / np.linalg.norm(v0)
            
            options = {
                'cone': {'P': 'nonnegort', 'Q': 'nonnegort'},
                'maxiter': max_iter,
                'accuracy': accuracy,
                'beta': 0.5,
                'display': 0,
                'v0': v0,
            }
            
            start = time.time()
            u, v, e = AlternatingOptimization(A, options)
            T += time.time() - start
            
            if e[-1] < emin - tol:
                emin = e[-1]
        
        best_angle = compute_angle_from_value(emin)
        all_best_values.append(emin)
        all_best_angles.append(best_angle)
        
        # Check for early stopping
        if j >= min_init_points - 1 and (j + 1) % check_interval == 0:
            std_angle = np.std(all_best_angles)
            
            if std_angle < convergence_tol:
                if verbose:
                    print(f"    ✓ Early stop at {j + 1} runs (std={std_angle:.2e})")
                break
        
        if verbose and (j + 1) % 20 == 0:
            std_angle = np.std(all_best_angles)
            print(f"    Progress: {j + 1}/{max_init_points} runs (std={std_angle:.2e})")
    
    return {
        'all_values': all_best_values,
        'all_angles': all_best_angles,
        'mean_angle': np.mean(all_best_angles),
        'std_angle': np.std(all_best_angles),
        'best_angle': np.min(all_best_angles),
        'best_value': all_best_values[np.argmin(all_best_angles)],
        'num_runs': len(all_best_angles),
    }


def test_incremental(A, m, max_iter=500, accuracy=1e-10, 
                     growth_scheme='both', verbose=True):
    """Test Incremental Strategy"""
    if verbose:
        print(f"  Incremental: scheme '{growth_scheme}'...")
    
    options = {
        'cone': {'P': 'nonnegort', 'Q': 'nonnegort'},
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
            initial_size=5,
            extension_method='small',
            display=False,
            final_run=True,
            max_intermediate_iter=100,
            use_sorting=False,
        )
        
        elapsed = time.time() - start
        
        return {
            'value': objectives_list[-1],
            'angle': compute_angle_from_value(objectives_list[-1]),
            'time': elapsed,
            'iterations': len(objectives_list),
            'submatrix_steps': len(submatrix_info),
        }
        
    except Exception as e:
        if verbose:
            print(f"    Failed: {e}")
        return None


def run_comparison_optimized(dimensions=None, timelimit=10, 
                            max_init_points=100, min_init_points=10,
                            test_incremental_flag=True, verbose=True):
    """
    Optimized comparison with early stopping
    """
    if dimensions is None:
        dimensions = list(range(17, 28, 2))
    
    print("="*70)
    print("OPTIMIZED COMPARISON: E-AO vs INCREMENTAL")
    print("Problem: Circulant PSD vs Nonnegative")
    print("="*70)
    print(f"\nDimensions: {dimensions}")
    print(f"E-AO: Adaptive (max {max_init_points}, early stop if converged)")
    print(f"Incremental: Single run")
    print()
    
    all_results = []
    
    for n in dimensions:
        if verbose:
            print(f"\n{'='*70}")
            print(f"n = {n} (m = {(n-1)//2})")
            print(f"{'='*70}")
        
        A, G, H, m = setup_circulant_problem(n)
        
        # Test E-AO with adaptive stopping
        eao_start = time.time()
        eao_results = test_eao_adaptive(
            A, m,
            timelimit=timelimit,
            max_init_points=max_init_points,
            min_init_points=min_init_points,
            verbose=verbose
        )
        eao_time = time.time() - eao_start
        
        result = {
            'n': n,
            'm': m,
            'eao_mean_angle': eao_results['mean_angle'],
            'eao_std_angle': eao_results['std_angle'],
            'eao_best_angle': eao_results['best_angle'],
            'eao_num_runs': eao_results['num_runs'],
            'eao_time': eao_time,
        }
        
        if verbose:
            print(f"\n  E-AO Results ({eao_results['num_runs']} runs):")
            print(f"    Mean:  {eao_results['mean_angle']:.8f}π")
            print(f"    Std:   {eao_results['std_angle']:.8f}π")
            print(f"    Best:  {eao_results['best_angle']:.8f}π")
            print(f"    Time:  {eao_time:.3f}s")
        
        # Test Incremental
        if test_incremental_flag:
            inc_results = test_incremental(A, m, verbose=verbose)
            
            if inc_results:
                result.update({
                    'inc_angle': inc_results['angle'],
                    'inc_time': inc_results['time'],
                    'inc_iterations': inc_results['iterations'],
                })
                
                if verbose:
                    print(f"\n  Incremental Results:")
                    print(f"    Angle: {inc_results['angle']:.8f}π")
                    print(f"    Time:  {inc_results['time']:.4f}s")
                    # Avoid division by zero
                    if inc_results['time'] > 0:
                        speedup = eao_time / inc_results['time']
                        print(f"    Speedup vs E-AO: {speedup:.1f}x")
                    else:
                        print(f"    Speedup vs E-AO: >100000x (too fast to measure)")

        
        all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'n':>5} {'m':>5} {'E-AO':>12} {'Runs':>5} {'Time':>8} {'Incr':>12} {'Time':>8} {'Speedup':>8}")
    print("-"*70)
    
    for _, row in df.iterrows():
        line = f"{row['n']:>5} {row['m']:>5}"
        line += f" {row['eao_best_angle']:>11.8f}π"
        line += f" {row['eao_num_runs']:>5}"
        line += f" {row['eao_time']:>7.3f}s"
        
        if 'inc_angle' in row and row['inc_angle'] is not None:
            line += f" {row['inc_angle']:>11.8f}π"
            line += f" {row['inc_time']:>7.4f}s"  # More precision for small times
            # Avoid division by zero
            if row['inc_time'] > 0:
                speedup = row['eao_time'] / row['inc_time']
                line += f" {speedup:>7.0f}x"
            else:
                line += f" {'>100k':>7s}x"
        
        print(line)
    
    # Stats
    print("\n" + "="*70)
    if test_incremental_flag:
        # Compute average speedup, avoiding division by zero
        valid_speedups = []
        for _, row in df.iterrows():
            if 'inc_time' in row and row['inc_time'] > 0:
                valid_speedups.append(row['eao_time'] / row['inc_time'])
        
        if valid_speedups:
            print(f"Average speedup: {np.mean(valid_speedups):.0f}x")
        else:
            print(f"Average speedup: >100000x")
        
        print(f"Total time saved: {(df['eao_time'].sum() - df['inc_time'].sum()):.1f}s")
    
    print(f"Average E-AO runs before stopping: {df['eao_num_runs'].mean():.1f}")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        if n % 2 == 0:
            print("Error: n must be odd")
            sys.exit(1)
        dimensions = [n]
    else:
        dimensions = list(range(17, 28, 2))
    
    df_results = run_comparison_optimized(
        dimensions=dimensions,
        timelimit=10,
        max_init_points=100,
        min_init_points=10,
        test_incremental_flag=True,
        verbose=True
    )
    
    output_file = 'comparison_circulant_optimized.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")