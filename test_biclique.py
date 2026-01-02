"""
Biclique comparison: Track WHEN optimum is found (time + iterations)
"""

import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'algorithms'))

from AlternatingOptimization import AlternatingOptimization
from IncrementalStratWithSort import IncrementalStrategy, sort_bipartite_graph

# Configuration
timelimit = 10
Ninitpoint = 5
growth_schemes = ['square_first', 'rows', 'cols', 'both']

# Results storage
results = {
    'eao': np.zeros((4, Ninitpoint)),
    'eao_sorted': np.zeros((4, Ninitpoint)),
    # NEW: Track when best value was found
    'eao_time_to_best': np.zeros((4, Ninitpoint)),  # Time when best found
    'eao_sorted_time_to_best': np.zeros((4, Ninitpoint)),
    'eao_iters_to_best': np.zeros((4, Ninitpoint)),  # Iterations when best found
    'eao_sorted_iters_to_best': np.zeros((4, Ninitpoint)),
    'eao_total_time': np.zeros((4, Ninitpoint)),  # Total time used
    'eao_sorted_total_time': np.zeros((4, Ninitpoint)),
    'eao_total_iters': np.zeros((4, Ninitpoint)),  # Total iterations
    'eao_sorted_total_iters': np.zeros((4, Ninitpoint)),
}

for scheme in growth_schemes:
    results[f'inc_{scheme}'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_time_to_best'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted_time_to_best'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_iters_to_best'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted_iters_to_best'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_total_time'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted_total_time'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_total_iters'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted_total_iters'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_runs'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted_runs'] = np.zeros((4, Ninitpoint))

print("="*70)
print("BICLIQUE: Track WHEN optimum is found")
print("="*70)

for i in range(4):
    print(f"\n{'='*70}")
    print(f"GRAPH {i+1}/4")
    print(f"{'='*70}")
    
    # Load graph
    Adj = np.loadtxt(f"results_bicliques/Biclique_matrix_{i+1}.txt", delimiter=',', dtype=int)
    m, n = Adj.shape
    A = -Adj + (1 - Adj) * max(m, n)
    
    initial_m = max(5, m // 20)
    initial_n = max(5, n // 20)
    initial_size = (initial_m, initial_n)
    
    print(f"Size: {m}x{n}, Initial: {initial_m}x{initial_n}")
    
    options = {
        'cone': {'P': 'nonnegort', 'Q': 'nonnegort'},
        'accuracy': 1e-10,
        'maxiter': 500,
        'beta': 0.5,
        'display': 0
    }
    tol = 1e-8
    
    # =========================================================================
    # E-AO STANDARD - Track when best found
    # =========================================================================
    print("\nE-AO standard:", end=' ')
    for j in range(Ninitpoint):
        emin = 0
        total_iters = 0
        
        # Track WHEN best was found
        time_to_best = 0
        iters_to_best = 0
        
        start_total = time.time()
        T = 0
        
        while T < timelimit and timelimit - T >= 0.1:
            options['v0'] = np.random.rand(n) / np.linalg.norm(np.random.rand(n))
            _, _, e = AlternatingOptimization(A, options)
            T = time.time() - start_total
            
            total_iters += len(e)
            
            # Check if this is a new best
            if e[-1] < emin - tol:
                emin = e[-1]
                # Record WHEN this best was found
                time_to_best = T
                iters_to_best = total_iters
        
        results['eao'][i, j] = emin ** 2
        results['eao_time_to_best'][i, j] = time_to_best
        results['eao_iters_to_best'][i, j] = iters_to_best
        results['eao_total_time'][i, j] = T
        results['eao_total_iters'][i, j] = total_iters
    
    mean_val = np.mean(results['eao'][i, :])
    mean_time = np.mean(results['eao_time_to_best'][i, :])
    mean_iters = np.mean(results['eao_iters_to_best'][i, :])
    print(f"{mean_val:.2f} → found at {mean_time:.8f}s / {mean_iters:.0f}it")
    
    # =========================================================================
    # E-AO SORTED - Track when best found
    # =========================================================================
    print("E-AO sorted:  ", end=' ')
    Adj_sorted, _, _ = sort_bipartite_graph(Adj, return_permutations=True)
    A_sorted = -Adj_sorted + (1 - Adj_sorted) * max(Adj_sorted.shape)
    
    for j in range(Ninitpoint):
        emin = 0
        total_iters = 0
        time_to_best = 0
        iters_to_best = 0
        
        start_total = time.time()
        T = 0
        
        while T < timelimit and timelimit - T >= 0.1:
            options['v0'] = np.random.rand(Adj_sorted.shape[1]) / np.linalg.norm(np.random.rand(Adj_sorted.shape[1]))
            _, _, e = AlternatingOptimization(A_sorted, options)
            T = time.time() - start_total
            
            total_iters += len(e)
            
            if e[-1] < emin - tol:
                emin = e[-1]
                time_to_best = T
                iters_to_best = total_iters
        
        results['eao_sorted'][i, j] = emin ** 2
        results['eao_sorted_time_to_best'][i, j] = time_to_best
        results['eao_sorted_iters_to_best'][i, j] = iters_to_best
        results['eao_sorted_total_time'][i, j] = T
        results['eao_sorted_total_iters'][i, j] = total_iters
    
    mean_val = np.mean(results['eao_sorted'][i, :])
    mean_time = np.mean(results['eao_sorted_time_to_best'][i, :])
    mean_iters = np.mean(results['eao_sorted_iters_to_best'][i, :])
    print(f"{mean_val:.2f} → found at {mean_time:.8f}s / {mean_iters:.0f}it")
    
    # =========================================================================
    # INCREMENTAL - Track when best found
    # =========================================================================
    for scheme in growth_schemes:
        print(f"\nInc {scheme:12s}:", end=' ')
        
        # WITHOUT sorting
        for j in range(Ninitpoint):
            emin = 0
            best_info = None
            run_count = 0
            
            time_to_best = 0
            iters_to_best = 0
            total_iters = 0
            
            start_total = time.time()
            T = 0
            estimated_time_per_run = 2.0
            
            while T < timelimit:
                if timelimit - T < estimated_time_per_run * 0.3:
                    break
                
                u, v, e, dims, info = IncrementalStrategy(
                    A, options, growth_scheme=scheme, initial_size=initial_size,
                    extension_method='small', display=False, final_run=True,
                    max_intermediate_iter=100, use_sorting=False
                )
                T = time.time() - start_total
                run_count += 1
                estimated_time_per_run = T / run_count
                
                # Count iterations for this run
                run_iters = sum(step['iterations'] for step in info)
                total_iters += run_iters
                
                if e[-1] < emin - tol:
                    emin = e[-1]
                    best_info = info
                    # Record WHEN this best was found
                    time_to_best = T
                    iters_to_best = total_iters
            
            results[f'inc_{scheme}'][i, j] = emin ** 2
            results[f'inc_{scheme}_runs'][i, j] = run_count
            results[f'inc_{scheme}_time_to_best'][i, j] = time_to_best
            results[f'inc_{scheme}_iters_to_best'][i, j] = iters_to_best
            results[f'inc_{scheme}_total_time'][i, j] = T
            results[f'inc_{scheme}_total_iters'][i, j] = total_iters
        
        mean_no = np.mean(results[f'inc_{scheme}'][i, :])
        time_no = np.mean(results[f'inc_{scheme}_time_to_best'][i, :])
        iters_no = np.mean(results[f'inc_{scheme}_iters_to_best'][i, :])
        print(f"NO={mean_no:.2f}→{time_no:.8f}s/{iters_no:.0f}it", end='  ')
        
        # WITH sorting
        for j in range(Ninitpoint):
            emin = 0
            best_info = None
            run_count = 0
            
            time_to_best = 0
            iters_to_best = 0
            total_iters = 0
            
            start_total = time.time()
            T = 0
            estimated_time_per_run = 2.0
            
            while T < timelimit:
                if timelimit - T < estimated_time_per_run * 0.3:
                    break
                
                u, v, e, dims, info = IncrementalStrategy(
                    A, options, growth_scheme=scheme, initial_size=initial_size,
                    extension_method='small', display=False, final_run=True,
                    max_intermediate_iter=100, use_sorting=True, Adj_original=Adj
                )
                T = time.time() - start_total
                run_count += 1
                estimated_time_per_run = T / run_count
                
                run_iters = sum(step['iterations'] for step in info)
                total_iters += run_iters
                
                if e[-1] < emin - tol:
                    emin = e[-1]
                    best_info = info
                    time_to_best = T
                    iters_to_best = total_iters
            
            results[f'inc_{scheme}_sorted'][i, j] = emin ** 2
            results[f'inc_{scheme}_sorted_runs'][i, j] = run_count
            results[f'inc_{scheme}_sorted_time_to_best'][i, j] = time_to_best
            results[f'inc_{scheme}_sorted_iters_to_best'][i, j] = iters_to_best
            results[f'inc_{scheme}_sorted_total_time'][i, j] = T
            results[f'inc_{scheme}_sorted_total_iters'][i, j] = total_iters
        
        mean_yes = np.mean(results[f'inc_{scheme}_sorted'][i, :])
        time_yes = np.mean(results[f'inc_{scheme}_sorted_time_to_best'][i, :])
        iters_yes = np.mean(results[f'inc_{scheme}_sorted_iters_to_best'][i, :])
        print(f"YES={mean_yes:.2f}→{time_yes:.8f}s/{iters_yes:.0f}it")

# =========================================================================
# SUMMARY: TIME TO REACH OPTIMUM
# =========================================================================

print("\n" + "="*70)
print("TIME TO REACH BEST VALUE (seconds) - MEAN (±STD)")
print("="*70)

header = f"{'Graph':<8} {'E-AO':<16} {'E-AO Sort':<16}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO      "
    header += f" {scheme[:6]}_YES     "
print(header)
print("-"*150)

for i in range(4):
    mean_eao = np.mean(results['eao_time_to_best'][i, :])
    std_eao = np.std(results['eao_time_to_best'][i, :])
    mean_sorted = np.mean(results['eao_sorted_time_to_best'][i, :])
    std_sorted = np.std(results['eao_sorted_time_to_best'][i, :])
    
    row = f"{i+1:<8} {mean_eao:<6.2f}(±{std_eao:<5.2f}) {mean_sorted:<6.2f}(±{std_sorted:<5.2f})"
    
    for scheme in growth_schemes:
        mean_no = np.mean(results[f'inc_{scheme}_time_to_best'][i, :])
        std_no = np.std(results[f'inc_{scheme}_time_to_best'][i, :])
        mean_yes = np.mean(results[f'inc_{scheme}_sorted_time_to_best'][i, :])
        std_yes = np.std(results[f'inc_{scheme}_sorted_time_to_best'][i, :])
        row += f" {mean_no:<5.2f}(±{std_no:<4.2f})"
        row += f" {mean_yes:<5.2f}(±{std_yes:<4.2f})"
    print(row)

print("\n" + "="*70)
print("ITERATIONS TO REACH BEST VALUE - MEAN (±STD)")
print("="*70)

header = f"{'Graph':<8} {'E-AO':<16} {'E-AO Sort':<16}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO      "
    header += f" {scheme[:6]}_YES     "
print(header)
print("-"*150)

for i in range(4):
    mean_eao = np.mean(results['eao_iters_to_best'][i, :])
    std_eao = np.std(results['eao_iters_to_best'][i, :])
    mean_sorted = np.mean(results['eao_sorted_iters_to_best'][i, :])
    std_sorted = np.std(results['eao_sorted_iters_to_best'][i, :])
    
    row = f"{i+1:<8} {mean_eao:<6.0f}(±{std_eao:<5.0f}) {mean_sorted:<6.0f}(±{std_sorted:<5.0f})"
    
    for scheme in growth_schemes:
        mean_no = np.mean(results[f'inc_{scheme}_iters_to_best'][i, :])
        std_no = np.std(results[f'inc_{scheme}_iters_to_best'][i, :])
        mean_yes = np.mean(results[f'inc_{scheme}_sorted_iters_to_best'][i, :])
        std_yes = np.std(results[f'inc_{scheme}_sorted_iters_to_best'][i, :])
        row += f" {mean_no:<5.0f}(±{std_no:<4.0f})"
        row += f" {mean_yes:<5.0f}(±{std_yes:<4.0f})"
    print(row)

print("\n" + "="*70)
print("EFFICIENCY: % of time used productively")
print("(time_to_best / total_time)")
print("="*70)

header = f"{'Graph':<8} {'E-AO':>10} {'E-AO Sort':>12}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO"
    header += f" {scheme[:6]}_YES"
print(header)
print("-"*90)

for i in range(4):
    eff_eao = 100 * np.mean(results['eao_time_to_best'][i, :]) / np.mean(results['eao_total_time'][i, :])
    eff_sorted = 100 * np.mean(results['eao_sorted_time_to_best'][i, :]) / np.mean(results['eao_sorted_total_time'][i, :])
    
    row = f"{i+1:<8} {eff_eao:>9.1f}% {eff_sorted:>11.1f}%"
    
    for scheme in growth_schemes:
        eff_no = 100 * np.mean(results[f'inc_{scheme}_time_to_best'][i, :]) / np.mean(results[f'inc_{scheme}_total_time'][i, :])
        eff_yes = 100 * np.mean(results[f'inc_{scheme}_sorted_time_to_best'][i, :]) / np.mean(results[f'inc_{scheme}_sorted_total_time'][i, :])
        row += f" {eff_no:>9.1f}%"
        row += f" {eff_yes:>10.1f}%"
    print(row)

# Save results
np.savez('biclique_time_to_optimum.npz', **results)
print("\nResults saved to 'biclique_time_to_optimum.npz'")
print("="*70)