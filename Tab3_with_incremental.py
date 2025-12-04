"""
Numerical comparison for E-AO with and without incremental strategy 
for the problem of finding the maximum edge biclique in four different 
bipartite graphs.

Extension of the original Table 3 to include incremental submatrix strategy.
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
from IncrementalStrat import IncrementalStrategy

timelimit = 10
Ninitpoint = 5

# Results storage
EAO_all = np.zeros((4, Ninitpoint))
EAO_mean = np.zeros(4)

# Incremental strategy results for different growth schemes
growth_schemes = ['square_first', 'rows', 'cols', 'both']
Incremental_all = {scheme: np.zeros((4, Ninitpoint)) for scheme in growth_schemes}
Incremental_mean = {scheme: np.zeros(4) for scheme in growth_schemes}

# Option: Test incremental WITH or WITHOUT final E-AO on full matrix
# Set to False to test pure incremental strategy
# Set to True to test incremental as warm-start for E-AO
USE_FINAL_RUN = True

print("="*70)
print("BICLIQUE PROBLEM: E-AO vs Incremental Strategy")
if USE_FINAL_RUN:
    print("(Incremental strategy WITH final E-AO on full matrix)")
else:
    print("(Incremental strategy WITHOUT final E-AO - pure incremental)")
print("="*70)

for i in range(4):
    print(f"\n{'='*70}")
    print(f"GRAPH {i+1}/4")
    print(f"{'='*70}")
    
    # Load bipartite graph adjacency matrix
    Adj = np.loadtxt(f"Biclique_matrix_{i+1}.txt", delimiter=',', dtype=int)
    m, n = Adj.shape
    print(f"Graph size: {m} x {n}")
    
    # Transform adjacency matrix for optimization
    A = -Adj + (1 - Adj) * n

    # =========================================================================
    # STANDARD E-AO
    # =========================================================================
    print(f"\n--- Standard E-AO ---")
    
    options_standard = {
        'cone': {'P': 'nonnegort', 'Q': 'nonnegort'},
        'accuracy': 1e-10,
        'maxiter': 500,
        'beta': 0.5,
        'display': 0
    }
    tol = 1e-8
    
    for j in range(Ninitpoint):
        print(f"  Run {j+1}/{Ninitpoint}...", end=' ')
        emin = 0
        T = 0
        run_count = 0
        
        while T < timelimit:
            start = time.time()
            options_standard['v0'] = np.random.rand(n)
            options_standard['v0'] /= np.linalg.norm(options_standard['v0'])
            _, _, e = AlternatingOptimization(A, options_standard)
            T += time.time() - start
            run_count += 1
            
            if e[-1] < emin - tol:
                emin = e[-1]
        
        EAO_all[i, j] = emin ** 2
        print(f"Best: {EAO_all[i, j]:.2f} ({run_count} iterations)")
    
    EAO_mean[i] = np.mean(EAO_all[i, :])
    print(f"  Average: {EAO_mean[i]:.2f}")

    # =========================================================================
    # INCREMENTAL STRATEGY E-AO
    # =========================================================================
    
    # Determine appropriate initial size based on graph dimensions
    min_dim = min(m, n)
    # Start smaller: 5% of smallest dimension (but at least 5)
    initial_size = max(5, min_dim // 20)
    
    for scheme in growth_schemes:
        print(f"\n--- Incremental Strategy: {scheme} ---")
        print(f"  Initial size: {initial_size}")
        
        options_incremental = {
            'cone': {'P': 'nonnegort', 'Q': 'nonnegort'},
            'accuracy': 1e-10,
            'maxiter': 500,
            'beta': 0.5,
            'display': 0
        }
        
        for j in range(Ninitpoint):
            print(f"  Run {j+1}/{Ninitpoint}...", end=' ')
            emin = 0
            T = 0
            run_count = 0
            
            # Each run starts fresh and we keep calling until time is up
            # This matches the E-AO approach where multiple random initializations
            # are tried within the time limit
            while T < timelimit:
                start = time.time()
                
                # Run the incremental strategy once with random initialization
                # This will go through the incremental steps and end with a final E-AO
                _, _, e = IncrementalStrategy(
                    A, 
                    options_incremental, 
                    growth_scheme=scheme,
                    initial_size=initial_size,
                    extension_method='small',
                    display=False,
                    final_run=USE_FINAL_RUN,  # Use the global setting
                    max_intermediate_iter=50
                )
                
                elapsed = time.time() - start
                T += elapsed
                run_count += 1
                
                # Keep the best result found so far
                if e[-1] < emin - tol:
                    emin = e[-1]
                
                # If a single incremental run takes too long, break
                if elapsed > timelimit:
                    break
            
            Incremental_all[scheme][i, j] = emin ** 2
            print(f"Best: {Incremental_all[scheme][i, j]:.2f} ({run_count} iterations in {T:.2f}s)")
        
        Incremental_mean[scheme][i] = np.mean(Incremental_all[scheme][i, :])
        print(f"  Average: {Incremental_mean[scheme][i]:.2f}")

# =========================================================================
# SUMMARY TABLE
# =========================================================================

print("\n" + "="*70)
print("SUMMARY TABLE - Maximum Edge Bicliques Found")
print("="*70)

# Print header
header = f"{'Graph':<10} {'Standard E-AO':<20}"
for scheme in growth_schemes:
    header += f" {f'Inc-{scheme}':<15}"
print(header)
print("-"*70)

# Print results for each graph
for i in range(4):
    row = f"{i+1:<10} {EAO_mean[i]:<20.2f}"
    for scheme in growth_schemes:
        row += f" {Incremental_mean[scheme][i]:<15.2f}"
    print(row)

# Print detailed results
print("\n" + "="*70)
print("DETAILED RESULTS (All runs)")
print("="*70)

for i in range(4):
    print(f"\nGraph {i+1}:")
    print(f"  Standard E-AO:     {EAO_all[i, :]}")
    for scheme in growth_schemes:
        print(f"  Inc-{scheme:12s}: {Incremental_all[scheme][i, :]}")

# =========================================================================
# COMPARISON ANALYSIS
# =========================================================================

print("\n" + "="*70)
print("IMPROVEMENT ANALYSIS")
print("="*70)

for i in range(4):
    print(f"\nGraph {i+1}:")
    baseline = EAO_mean[i]
    print(f"  Standard E-AO baseline: {baseline:.2f}")
    
    for scheme in growth_schemes:
        inc_value = Incremental_mean[scheme][i]
        improvement = ((inc_value - baseline) / baseline) * 100 if baseline > 0 else 0
        symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
        print(f"  {scheme:15s}: {inc_value:8.2f} ({symbol} {abs(improvement):5.2f}%)")

# =========================================================================
# SAVE RESULTS
# =========================================================================

results_dict = {
    'EAO_all': EAO_all,
    'EAO_mean': EAO_mean,
}

for scheme in growth_schemes:
    results_dict[f'Incremental_{scheme}_all'] = Incremental_all[scheme]
    results_dict[f'Incremental_{scheme}_mean'] = Incremental_mean[scheme]

np.savez('table3_results.npz', **results_dict)
print("\n" + "="*70)
print("Results saved to 'table3_results.npz'")
print("="*70)

# =========================================================================
# VISUALIZATION (if matplotlib available)
# =========================================================================

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(4):
        ax = axes[i]
        
        # Standard E-AO
        ax.scatter([0] * Ninitpoint, EAO_all[i, :], 
                  color=colors[0], alpha=0.6, s=100, label='Standard E-AO')
        ax.plot([-0.2, 0.2], [EAO_mean[i], EAO_mean[i]], 
               color=colors[0], linewidth=2)
        
        # Incremental strategies
        for idx, scheme in enumerate(growth_schemes):
            x_pos = idx + 1
            ax.scatter([x_pos] * Ninitpoint, Incremental_all[scheme][i, :], 
                      color=colors[idx+1], alpha=0.6, s=100, 
                      label=f'Inc-{scheme}')
            ax.plot([x_pos-0.2, x_pos+0.2], 
                   [Incremental_mean[scheme][i], Incremental_mean[scheme][i]], 
                   color=colors[idx+1], linewidth=2)
        
        ax.set_title(f'Graph {i+1}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Maximum Edge Biclique', fontsize=10)
        ax.set_xticks(range(len(growth_schemes)+1))
        ax.set_xticklabels(['Standard'] + [s[:8] for s in growth_schemes], 
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('table3_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'table3_comparison.png'")
    plt.show()
    
except ImportError:
    print("\nMatplotlib not available. Skipping visualization.")