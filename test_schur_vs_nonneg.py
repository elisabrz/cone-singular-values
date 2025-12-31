"""
Test Schur vs Nonneg - Following paper's exact approach

Follows the MATLAB code from the paper:
- Restart E-AO with random initializations within time limit
- Keep best result (minimum value)
- Compare with reference values
"""

import numpy as np
import time
from AlternatingOptimization import AlternatingOptimization
from utils.schur import schur
from utils.pi_unit import compute_angle_from_value
from reference_data import get_reference_data
from IncrementalStratWithSort import IncrementalStrategy


def test_schur_vs_nonneg(n, timelimit=10, max_iter=500, accuracy=1e-6, 
                          verbose=True, test_incremental=True):
    """
    Test Schur vs R+ following paper's approach
    
    Parameters
    ----------
    n : int
        Dimension
    timelimit : float
        Time limit in seconds (default: 10s like in paper)
    max_iter : int
        Maximum iterations per E-AO run
    accuracy : float
        Convergence tolerance (1e-6 for Schur vs Nonneg)
    verbose : bool
        Print progress
    test_incremental : bool
        Also test incremental strategy
    
    Returns
    -------
    results : dict
        Results dictionary
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SCHUR vs R+ (n={n})")
        print(f"{'='*70}")
    
    # Load reference
    ref = get_reference_data()
    problem_name = f'Schur_vs_R+_n{n}'
    ref_data = ref.get_problem(problem_name)
    
    exact_value = ref_data.get('exact_value') if ref_data else None
    exact_angle = ref_data.get('exact_angle') if ref_data else None
    
    if verbose and exact_value is not None:
        print(f"\nReference:")
        print(f"  Exact: value={exact_value:.8f}, angle={exact_angle:.8f}π")
    
    # Setup
    G_schur = schur(n)
    A = np.eye(n)
    
    results = {
        'n': n,
        'problem_name': problem_name,
        'exact_value': exact_value,
        'exact_angle': exact_angle,
    }
    
    # ========== E-AO (Paper's Multi-Restart Approach) ==========
    if verbose:
        print(f"\n{'-'*70}")
        print("METHOD 1: E-AO (Multi-Restart)")
        print(f"{'-'*70}")
        print(f"  Time limit: {timelimit}s")
        print(f"  Accuracy: {accuracy}")
    
    tol = 1e-8
    T = 0  # Elapsed time
    emin = 1  # Best value found so far
    num_restarts = 0
    best_u = None
    best_v = None
    best_e = None
    
    start_total = time.time()
    
    while T < timelimit:
        # Random initialization
        v0 = np.random.rand(n)
        v0 = v0 / np.linalg.norm(v0)
        
        options_eao = {
            'cone': {'P': 'generator', 'Q': 'nonnegort'},
            'G': G_schur,
            'maxiter': max_iter,
            'accuracy': accuracy,
            'beta': 0.5,
            'display': 0,
            'v0': v0,
        }
        
        # Run E-AO
        start_run = time.time()
        u, v, e = AlternatingOptimization(A, options_eao)
        T = time.time() - start_total
        
        num_restarts += 1
        
        # Check if improved
        if e[-1] < emin - tol:
            emin = e[-1]
            best_u = u
            best_v = v
            best_e = e
            
            if verbose and num_restarts <= 3:
                angle_trial = compute_angle_from_value(e[-1])
                print(f"    Restart {num_restarts}: value={e[-1]:.6f}, angle={angle_trial:.6f}π, time={T:.2f}s")
    
    # Final results
    value_eao = emin
    angle_eao = compute_angle_from_value(value_eao)
    time_eao = time.time() - start_total
    
    error_eao = None
    if exact_angle is not None:
        error_eao = abs(angle_eao - exact_angle)
    
    results['eao'] = {
        'value': value_eao,
        'angle': angle_eao,
        'error': error_eao,
        'time': time_eao,
        'iterations': len(best_e) if best_e else 0,
        'num_restarts': num_restarts,
    }
    
    if verbose:
        print(f"\n  Final result (after {num_restarts} restarts):")
        print(f"    Value:      {value_eao:.8f}")
        print(f"    Angle:      {angle_eao:.8f}π")
        if error_eao: print(f"    Error:      {error_eao:.2e}")
        print(f"    Time:       {time_eao:.3f}s")
        print(f"    Iterations: {len(best_e) if best_e else 0} (last run)")
    
    # ========== INCREMENTAL STRATEGY ==========
    if test_incremental:
        if verbose:
            print(f"\n{'-'*70}")
            print("METHOD 2: INCREMENTAL STRATEGY")
            print(f"{'-'*70}")
        
        options_inc = {
            'cone': {'P': 'generator', 'Q': 'nonnegort'},
            'G': G_schur,
            'H': None,
            'maxiter': max_iter,
            'accuracy': accuracy,
            'beta': 0.5,
            'display': 0,
        }
        
        try:
            start_time = time.time()
            
            u_inc, v_inc, objectives_list, final_dims, submatrix_info = IncrementalStrategy(
                A=A,
                options=options_inc,
                growth_scheme='both',
                initial_size=5,
                extension_method='small',
                display=False,
                final_run=True,
                max_intermediate_iter=100,
                use_sorting=False,
            )
            
            time_inc = time.time() - start_time
            
            value_inc = objectives_list[-1]
            angle_inc = compute_angle_from_value(value_inc)
            
            error_inc = None
            if exact_angle is not None:
                error_inc = abs(angle_inc - exact_angle)
            
            results['incremental'] = {
                'value': value_inc,
                'angle': angle_inc,
                'error': error_inc,
                'time': time_inc,
                'iterations': len(objectives_list),
                'submatrix_steps': len(submatrix_info),
                'final_dims': final_dims,
            }
            
            if verbose:
                print(f"    Angle:           {angle_inc:.8f}π")
                if error_inc: print(f"    Error:           {error_inc:.2e}")
                print(f"    Time:            {time_inc:.3f}s")
                print(f"    Total iters:     {len(objectives_list)}")
                print(f"    Submatrix steps: {len(submatrix_info)}")
                
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed: {e}")
            results['incremental'] = None
    
    # ========== COMPARISON ==========
    if verbose:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        
        if exact_angle:
            print(f"Reference:   {exact_angle:.8f}π")
        
        print(f"E-AO:        {results['eao']['angle']:.8f}π  time={results['eao']['time']:.3f}s")
        
        if 'incremental' in results and results['incremental']:
            i = results['incremental']
            print(f"Incremental: {i['angle']:.8f}π  time={i['time']:.3f}s")
        
        # Determine best
        if exact_angle:
            best_method = 'E-AO' if results['eao']['error'] else None
            best_error = results['eao']['error']
            
            if 'incremental' in results and results['incremental']:
                if results['incremental']['error'] < best_error:
                    best_error = results['incremental']['error']
                    best_method = 'Incremental'
            
            if best_method:
                status = "SUCCESS" if best_error < 1e-4 else "GOOD" if best_error < 1e-2 else "FAILED"
                print(f"\n{status}: Best error = {best_error:.2e} ({best_method})")
        
        print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        timelimit = float(sys.argv[2]) if len(sys.argv) > 2 else 10
        
        result = test_schur_vs_nonneg(n, timelimit=timelimit, verbose=True)
    else:
        # Test multiple dimensions
        dimensions = [5, 10, 20, 50, 100, 200]
        
        all_results = []
        for n in dimensions:
            result = test_schur_vs_nonneg(n, timelimit=10, verbose=True)
            all_results.append(result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(f"\n{'n':>5} {'Exact':>12} {'E-AO':>12} {'Error':>10} {'Inc':>12} {'Error':>10}")
        print("-"*70)
        
        for r in all_results:
            line = f"{r['n']:>5} "
            line += f"{r['exact_angle']:>11.8f}π" if r['exact_angle'] else f"{'N/A':>12}"
            
            e = r['eao']
            line += f" {e['angle']:>11.8f}π"
            line += f" {e['error']:.2e}" if e['error'] else f" {'N/A':>10}"
            
            if 'incremental' in r and r['incremental']:
                i = r['incremental']
                line += f" {i['angle']:>11.8f}π"
                line += f" {i['error']:.2e}" if i['error'] else f" {'N/A':>10}"
            else:
                line += f" {'Failed':>12} {'N/A':>10}"
            
            print(line)