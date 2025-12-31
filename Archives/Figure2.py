"""
E-AO compared on the problem of finding the maximum angle between the Schur 
cone and the nonnegative orthant, and between the Schur cone and itself in 
dimension n = 200, with a timelimit of 10 seconds

For E-AO, 100 iterations from random generated points are performed, and 
their performance over time is computed together with their average.

The data for the 100 iterations are reported in EAO_all where EAO_all[i,t] 
is the best value found on iteration i at time t/100 seconds, expressed as 
fractions of the angle pi.

The averages at time t/100 seconds are reported in EAO_mean[t], expressed 
as fractions of the angle pi.
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
# Parameters
n = 200
timelimit = 10
Ninitpoint = 5
# timemult is how many timestamps we log each second for E-AO
timemult = 100
Ndisctime = timelimit * timemult

## Schur - Positive Orthant

A = np.eye(n)
H = np.eye(n)
# G generates the Schur cone
G = (np.diag(np.ones(n-1), 1) - np.diag(np.ones(n))) / np.sqrt(2)
G = G[:, 1:]

# E-AO
options = {
    'cone': {'P': 'generator', 'Q': 'nonnegort'},
    'G': G,
    'accuracy': 1e-6,
    'maxiter': 500,
    'beta': 0.5,
    'display': 1
}
tol = 1e-8

EAO_all = np.zeros((Ninitpoint, Ndisctime + 1))
for j in range(Ninitpoint):  # for all random initial points
    T = 0
    EAO = []
    EAO_times = []
    emin = 1
    
    while T < timelimit:
        start_time = time.time() # tic in matlab
        
        options['v0'] = np.random.rand(n)
        options['v0'] = options['v0'] / np.linalg.norm(options['v0'])
        
        _, _, e = AlternatingOptimization(A, options)  # E-AO
        
        T = T + time.time() - start_time #toc en matlab
        
        if e[-1] < emin - tol:  # if the objective has improved
            EAO.append(e[-1])
            emin = e[-1]
            EAO_times.append(T)
    
    # for any timestamp, report the best value found
    pip = 0
    for i in range(len(EAO_times)):
        if EAO_times[i] > timelimit:
            break
        pipend = int(np.floor(EAO_times[i] * timemult))
        EAO_all[j, pip:pipend + 1] = EAO[i]
        pip = pipend + 1
    
    if len(EAO) == 0:
        EAO_all[j, :] = np.ones(Ndisctime + 1)
    else:
        EAO_all[j, pip:] = EAO[i]

EAO_all = np.arccos(EAO_all) / np.pi
# average of all runs
EAO_mean = np.mean(EAO_all, axis=0)

print(EAO_all)
print("\n")
print(EAO_mean)



'''
## Schur - Schur

A = np.eye(n)
# G generates the Schur cone
G = (np.diag(np.ones(n-1), 1) - np.diag(np.ones(n))) / np.sqrt(2)
G = G[:, 1:]
H = G.copy()

# E-AO
options = {
    'cone': {'P': 'generator', 'Q': 'generator'},
    'G': G,
    'H': H,
    'accuracy': 1e-10,
    'maxiter': 500,
    'beta': 0.5
}
tol = 1e-8

EAO_all_schur = np.zeros((Ninitpoint, Ndisctime + 1))
for j in range(Ninitpoint):  # for all random initial points
    T = 0
    EAO = []
    EAO_times = []
    emin = 1
    
    while T < timelimit:
        start_time = time.time()
        
        options['v0'] = np.random.rand(n)
        options['v0'] = options['v0'] / np.linalg.norm(options['v0'])
        
        _, _, e = AlternatingOptimization(A, options)  # E-AO
        
        T = T + time.time() - start_time
        
        if e[-1] < emin - tol:  # if the objective has improved
            EAO.append(e[-1])
            emin = e[-1]
            EAO_times.append(T)
    
    # for any timestamp, report the best value found
    pip = 0
    for i in range(len(EAO_times)):
        if EAO_times[i] > timelimit:
            break
        pipend = int(np.floor(EAO_times[i] * timemult))
        EAO_all_schur[j, pip:pipend + 1] = EAO[i]
        pip = pipend + 1
    
    if len(EAO) == 0:
        EAO_all_schur[j, :] = np.ones(Ndisctime + 1)
    else:
        EAO_all_schur[j, pip:] = EAO[i]

EAO_all_schur = np.arccos(EAO_all_schur) / np.pi
# average of all runs
EAO_mean_schur = np.mean(EAO_all_schur, axis=0)

'''