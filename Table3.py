# % Numerical comparison for Gurobi, BFAS, E-AO and
# %  SRPL for the problem of finding the maximum edge biclique
# %   in four different bipartite graphs
# %
# % Reports the maximum edge biclique found in the timelimit
# %  (10 seconds) for Gurobi and BFAS
# % The reported numbers for E-AO and SRPL are instead the 
# %  values found at 10 seconds in 100 runs and their averages
# % Gurobi cannot be executed on the last graph due to its 
# %  excessive size

# % The largest edge bicliques found in the i-th graph in the 
# %  100 iterations are reported in EAO_all(i,:) and 
# %   SRPL_all(i,:), while their averages are reported in 
# %    EAO_mean(i) and SRPL_mean(i)

# % The largest edge bicliques found by Gur and BFAS in the
# %  i-th graph are reported in BFAS(i) and Gur(i)
# % Notice that Gur(4) does not exists since Gurobi crashes 
# %  on the last graph

# % The four bipartite graphs are taken from the dataset in
# %  https://github.com/shahamer/maximum-biclique-benchmark 
# %   Shaham, E.: maximum biclique benchmark. (Dic 2019)

import numpy as np
import time
from AlternatingOptimization import AlternatingOptimization

timelimit = 10
Ninitpoint = 5

EAO_all = np.zeros((4, Ninitpoint))
EAO_mean = np.zeros(4)
#SRPL_all = np.zeros((4, Ninitpoint))
#SRPL_mean = np.zeros(4)
#BFAS = np.zeros(4)
#Gur = np.zeros(3)

for i in range(4):
    Adj = np.loadtxt(f"Biclique_matrix_{i+1}.txt", delimiter=',', dtype=int)
    m, n = Adj.shape
    A = -Adj + (1 - Adj) * n

    # E-AO
    options = {}
    options['cone'] = {'P': 'nonnegort', 'Q': 'nonnegort'}
    options['accuracy'] = 1e-10
    options['maxiter'] = 500
    options['beta'] = 0.5
    tol = 1e-8
    for j in range(Ninitpoint):
        emin = 0
        T = 0
        while T < timelimit:
            start = time.time()
            options['v0'] = np.random.rand(n)
            options['v0'] /= np.linalg.norm(options['v0'])
            _, _, e = AlternatingOptimization(A, options)
            T += time.time() - start
            if e[-1] < emin - tol:
                emin = e[-1]
        EAO_all[i, j] = emin ** 2
    EAO_mean[i] = np.mean(EAO_all[i, :])

    print(f"\nEAO_all : \n{EAO_all}")
    print(f"\nEAO_mean : {EAO_mean}")

    # # SRPL
    # G = np.eye(A.shape[0])
    # H = np.eye(A.shape[1])
    # tol = 1e-8
    # mu1 = 0.25
    # mu2 = 0.01
    # for j in range(Ninitpoint):
    #     emin = 0
    #     T = 0
    #     while T < timelimit:
    #         start = time.time()
    #         x0 = np.random.rand(A.shape[0])
    #         x0 /= np.sum(x0)
    #         y0 = np.random.rand(A.shape[1])
    #         y0 /= np.sum(y0)
    #         _, _, la, _, _ = srpl_poly(A, G, H, x0, y0, mu1, mu2)
    #         T += time.time() - start
    #         if la < emin - tol:
    #             emin = la
    #     SRPL_all[i, j] = emin ** 2
    # SRPL_mean[i] = np.mean(SRPL_all[i, :])

    # # BFAS
    # lamvec, _, _, _ = bfas_timestamps_test(G, H, A, timelimit)
    # BFAS[i] = lamvec[-1] ** 2

    # # Gurobi
    # if i < 3:
    #     val, _ = generators_gurobi_uv_test(G, H, A, timelimit)
    #     Gur[i] = val ** 2
