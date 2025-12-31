"""
Simplified comparison script: Incremental Strategy WITH and WITHOUT degree-based sorting
"""

import numpy as np
import time
import sys
import os

# Add algorithms directory to path
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
}

for scheme in growth_schemes:
    results[f'inc_{scheme}'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_iters'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_sorted_iters'] = np.zeros((4, Ninitpoint))
    results[f'inc_{scheme}_runs'] = np.zeros((4, Ninitpoint))  # Track number of runs
    results[f'inc_{scheme}_sorted_runs'] = np.zeros((4, Ninitpoint))

print("="*70)
print("BICLIQUE COMPARISON: E-AO vs Incremental (with/without sorting)")
print("="*70)

for i in range(4):
    print(f"\n{'='*70}")
    print(f"GRAPH {i+1}/4")
    print(f"{'='*70}")
    
    # Load graph
    Adj = np.loadtxt(f"Biclique_matrix_{i+1}.txt", delimiter=',', dtype=int)
    m, n = Adj.shape
    A = -Adj + (1 - Adj) * max(m, n)
    
    # Universal rectangular approach: compute initial size for both dimensions
    # Start with a fraction of each dimension
    initial_m = max(5, m // 20)
    initial_n = max(5, n // 20)
    initial_size = (initial_m, initial_n)
    
    print(f"Size: {m}x{n}, Initial submatrix: {initial_m}x{initial_n}")
    
    options = {
        'cone': {'P': 'nonnegort', 'Q': 'nonnegort'},
        'accuracy': 1e-10,
        'maxiter': 500,
        'beta': 0.5,
        'display': 0
    }
    tol = 1e-8
    
    # =========================================================================
    # E-AO STANDARD
    # =========================================================================
    print("\nE-AO standard:", end=' ')
    for j in range(Ninitpoint):
        emin, T, run_count = 0, 0, 0
        while T < timelimit and timelimit - T >= 0.1:
            start = time.time()
            options['v0'] = np.random.rand(n) / np.linalg.norm(np.random.rand(n))
            _, _, e = AlternatingOptimization(A, options)
            T += time.time() - start
            run_count += 1
            if e[-1] < emin - tol:
                emin = e[-1]
        results['eao'][i, j] = emin ** 2
    mean_eao = np.mean(results['eao'][i, :])
    std_eao = np.std(results['eao'][i, :])
    print(f"{mean_eao:.2f} (±{std_eao:.2f})")
    
    # =========================================================================
    # E-AO SORTED
    # =========================================================================
    print("E-AO sorted:  ", end=' ')
    Adj_sorted, _, _ = sort_bipartite_graph(Adj, return_permutations=True)
    A_sorted = -Adj_sorted + (1 - Adj_sorted) * max(Adj_sorted.shape)
    
    for j in range(Ninitpoint):
        emin, T, run_count = 0, 0, 0
        while T < timelimit and timelimit - T >= 0.1:
            start = time.time()
            options['v0'] = np.random.rand(Adj_sorted.shape[1]) / np.linalg.norm(np.random.rand(Adj_sorted.shape[1]))
            _, _, e = AlternatingOptimization(A_sorted, options)
            T += time.time() - start
            run_count += 1
            if e[-1] < emin - tol:
                emin = e[-1]
        results['eao_sorted'][i, j] = emin ** 2
    mean_eao_sorted = np.mean(results['eao_sorted'][i, :])
    std_eao_sorted = np.std(results['eao_sorted'][i, :])
    print(f"{mean_eao_sorted:.2f} (±{std_eao_sorted:.2f})")
    
    # =========================================================================
    # INCREMENTAL STRATEGIES
    # =========================================================================
    for scheme in growth_schemes:
        print(f"\nInc {scheme:12s}:", end=' ')
        
        # WITHOUT sorting
        for j in range(Ninitpoint):
            emin, T, best_info, run_count = 0, 0, None, 0
            # Estimate time per run (adjust after first run)
            estimated_time_per_run = 2.0  # Initial estimate
            
            while T < timelimit:
                # Check if we have enough time for another run
                if timelimit - T < estimated_time_per_run * 0.3:
                    break
                
                start = time.time()
                u, v, e, dims, info = IncrementalStrategy(
                    A, options, growth_scheme=scheme, initial_size=initial_size,
                    extension_method='small', display=False, final_run=True,
                    max_intermediate_iter=100, use_sorting=False
                )
                elapsed = time.time() - start
                T += elapsed
                run_count += 1
                
                # Update time estimate
                estimated_time_per_run = T / run_count
                
                if e[-1] < emin - tol:
                    emin, best_info = e[-1], info
            
            results[f'inc_{scheme}'][i, j] = emin ** 2
            results[f'inc_{scheme}_runs'][i, j] = run_count
            if best_info is not None:
                results[f'inc_{scheme}_iters'][i, j] = sum(step['iterations'] for step in best_info)
        
        mean_no = np.mean(results[f'inc_{scheme}'][i, :])
        std_no = np.std(results[f'inc_{scheme}'][i, :])
        avg_runs_no = np.mean(results[f'inc_{scheme}_runs'][i, :])
        print(f"NO={mean_no:.2f}(±{std_no:.2f})[{avg_runs_no:.1f}r]", end='  ')
        
        # WITH sorting
        for j in range(Ninitpoint):
            emin, T, best_info, run_count = 0, 0, None, 0
            estimated_time_per_run = 2.0
            
            while T < timelimit:
                if timelimit - T < estimated_time_per_run * 0.3:
                    break
                
                start = time.time()
                u, v, e, dims, info = IncrementalStrategy(
                    A, options, growth_scheme=scheme, initial_size=initial_size,
                    extension_method='small', display=False, final_run=True,
                    max_intermediate_iter=100, use_sorting=True, Adj_original=Adj
                )
                elapsed = time.time() - start
                T += elapsed
                run_count += 1
                
                estimated_time_per_run = T / run_count
                
                if e[-1] < emin - tol:
                    emin, best_info = e[-1], info
            
            results[f'inc_{scheme}_sorted'][i, j] = emin ** 2
            results[f'inc_{scheme}_sorted_runs'][i, j] = run_count
            if best_info is not None:
                results[f'inc_{scheme}_sorted_iters'][i, j] = sum(step['iterations'] for step in best_info)
        
        mean_yes = np.mean(results[f'inc_{scheme}_sorted'][i, :])
        std_yes = np.std(results[f'inc_{scheme}_sorted'][i, :])
        avg_runs_yes = np.mean(results[f'inc_{scheme}_sorted_runs'][i, :])
        print(f"YES={mean_yes:.2f}(±{std_yes:.2f})[{avg_runs_yes:.1f}r]")

# =========================================================================
# SUMMARY TABLES
# =========================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY - MEAN (±STD)")
print("="*70)

header = f"{'Graph':<8} {'E-AO':<16} {'E-AO Sort':<16}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO      "
    header += f" {scheme[:6]}_YES     "
print(header)
print("-"*150)

for i in range(4):
    mean_eao = np.mean(results['eao'][i, :])
    std_eao = np.std(results['eao'][i, :])
    mean_eao_sorted = np.mean(results['eao_sorted'][i, :])
    std_eao_sorted = np.std(results['eao_sorted'][i, :])
    
    row = f"{i+1:<8} {mean_eao:<6.2f}(±{std_eao:<5.2f}) {mean_eao_sorted:<6.2f}(±{std_eao_sorted:<5.2f})"
    
    for scheme in growth_schemes:
        mean_no = np.mean(results[f'inc_{scheme}'][i, :])
        std_no = np.std(results[f'inc_{scheme}'][i, :])
        mean_yes = np.mean(results[f'inc_{scheme}_sorted'][i, :])
        std_yes = np.std(results[f'inc_{scheme}_sorted'][i, :])
        row += f" {mean_no:<5.2f}(±{std_no:<4.2f})"
        row += f" {mean_yes:<5.2f}(±{std_yes:<4.2f})"
    print(row)

print("\n" + "="*70)
print("NUMBER OF RUNS - MEAN (MIN-MAX)")
print("="*70)

header = f"{'Graph':<8}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO     "
    header += f" {scheme[:6]}_YES    "
print(header)
print("-"*90)

for i in range(4):
    row = f"{i+1:<8}"
    for scheme in growth_schemes:
        mean_runs_no = np.mean(results[f'inc_{scheme}_runs'][i, :])
        min_runs_no = np.min(results[f'inc_{scheme}_runs'][i, :])
        max_runs_no = np.max(results[f'inc_{scheme}_runs'][i, :])
        mean_runs_yes = np.mean(results[f'inc_{scheme}_sorted_runs'][i, :])
        min_runs_yes = np.min(results[f'inc_{scheme}_sorted_runs'][i, :])
        max_runs_yes = np.max(results[f'inc_{scheme}_sorted_runs'][i, :])
        row += f" {mean_runs_no:.1f}({min_runs_no:.0f}-{max_runs_no:.0f})"
        row += f" {mean_runs_yes:.1f}({min_runs_yes:.0f}-{max_runs_yes:.0f})"
    print(row)

print("\n" + "="*70)
print("COMPUTATIONAL COST - MEAN (±STD) E-AO iterations")
print("="*70)

header = f"{'Graph':<8}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO      "
    header += f" {scheme[:6]}_YES     "
print(header)
print("-"*90)

for i in range(4):
    row = f"{i+1:<8}"
    for scheme in growth_schemes:
        mean_no = np.mean(results[f'inc_{scheme}_iters'][i, :])
        std_no = np.std(results[f'inc_{scheme}_iters'][i, :])
        mean_yes = np.mean(results[f'inc_{scheme}_sorted_iters'][i, :])
        std_yes = np.std(results[f'inc_{scheme}_sorted_iters'][i, :])
        row += f" {mean_no:<5.1f}(±{std_no:<4.1f})"
        row += f" {mean_yes:<5.1f}(±{std_yes:<4.1f})"
    print(row)

print("\n" + "="*70)
print("COMPUTATIONAL COST - MIN / MAX iterations")
print("="*70)

header = f"{'Graph':<8}"
for scheme in growth_schemes:
    header += f" {scheme[:6]}_NO     "
    header += f" {scheme[:6]}_YES    "
print(header)
print("-"*90)

for i in range(4):
    row = f"{i+1:<8}"
    for scheme in growth_schemes:
        min_no = np.min(results[f'inc_{scheme}_iters'][i, :])
        max_no = np.max(results[f'inc_{scheme}_iters'][i, :])
        min_yes = np.min(results[f'inc_{scheme}_sorted_iters'][i, :])
        max_yes = np.max(results[f'inc_{scheme}_sorted_iters'][i, :])
        row += f" {min_no:.0f}-{max_no:<6.0f}"
        row += f" {min_yes:.0f}-{max_yes:<6.0f}"
    print(row)

# =========================================================================
# VISUALIZATIONS
# =========================================================================

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    methods = ['E-AO', 'E-AO Sorted'] + [f'{s[:6]}_NO' for s in growth_schemes] + [f'{s[:6]}_YES' for s in growth_schemes]
    colors = ['gray', 'darkgray'] + ['lightcoral']*4 + ['darkred']*4
    
    for i in range(4):
        ax = axes[i]
        
        # Gather all results for this graph
        all_results = [
            results['eao'][i, :],
            results['eao_sorted'][i, :]
        ]
        for scheme in growth_schemes:
            all_results.append(results[f'inc_{scheme}'][i, :])
        for scheme in growth_schemes:
            all_results.append(results[f'inc_{scheme}_sorted'][i, :])
        
        positions = np.arange(len(methods))
        
        # Plot points and means with std bars
        for j, (res, color) in enumerate(zip(all_results, colors)):
            ax.scatter([positions[j]] * Ninitpoint, res, color=color, alpha=0.6, s=80)
            mean_val = np.mean(res)
            std_val = np.std(res)
            ax.plot([positions[j]-0.3, positions[j]+0.3], [mean_val, mean_val], 
                   color=color, linewidth=3)
            # Add error bars
            ax.errorbar(positions[j], mean_val, yerr=std_val, 
                       color=color, alpha=0.5, capsize=5, capthick=2)
        
        # Get graph dimensions for title
        Adj = np.loadtxt(f"Biclique_matrix_{i+1}.txt", delimiter=',', dtype=int)
        m_graph, n_graph = Adj.shape
        
        ax.set_title(f'Graph {i+1} ({m_graph}x{n_graph})', fontsize=14, fontweight='bold')
        ax.set_ylabel('Maximum Edge Biclique', fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=np.mean(results['eao'][i, :]), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('biclique_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to 'biclique_comparison.png'")
    plt.show()
    
except ImportError:
    print("\nMatplotlib not available. Skipping visualization.")
except Exception as e:
    print(f"\nVisualization error: {e}")

# Save results
np.savez('biclique_comparison_results.npz', **results)
print("Results saved to 'biclique_comparison_results.npz'")
print("="*70)