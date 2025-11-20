'''
% [u,v,e] = AlternatingOptimization(A,options)
%
% Alternating optimization for solving
%
%           min_{u in P, v in Q, ||u||=||v||=1} <u, Av>    (1)
%
% where
%       P = {u | u = G*x, x >= 0}, or
%         = {u | G*u >= g}, or
%         = {x | x >= 0}, (defaut) or
%         = {x | mat(x) is positive semidefinite (PSD)}.
%
% *** Input ***
%   A              : m-by-n matrix
%
% options:
%   maxiter        : maximum number of iterations (default = 500)
%   accuracy       : continue when  [ ||uk-1 - uk|| >= accuracy or
%                                     ||vk-1 - vk|| <= accuracy ]
%                               and objetive(k-1)-objective(k) >= accuracy
%                    (default = 1e-4)
%   cone.P         : type of cones, we have implemented 3 cases:
%                    1) 'generator': P = { u | u = options.G*x, x >= 0}
%                       In that case, options.G provides the generators
%                    2) 'nonnegort': P = { u | u >= 0}
%                    3) 'semidefin': P = { u | mat(u) is PSD}
%                    4) 'facetsrep': P = { u | options.G*u >= options.g}
%   cone.Q         : same structure as .P
%   G, H           : When options.cone.P = 'generator' or 'facetsrep'
%                     G is required, and similarly for Q and H.
%                    In the case 'generator':
%                    The columns of G (resp. H) are the rays of P (resp. Q)
%                    hence G (m-by-k) u = G*x in P for some x >= 0.
%                      and H (n-by-p) v = H*y in Q for some y >= 0.
%   g,h              In the case 'facetsrep':
%                    G provides the inequalities for P = {u | G*u >= g}
%                    H provides the inequalities for Q = {v | H*v >= h}
%                    default: g=0, h=0
%   v0             : initializations for the algorithm.
%                     default: v <-- argmin_{v in Q} (u^TA) v
%                               where u is randn(m,1).
%
% *** Output ***
%   (u,v) in PxQ   : approximate solution to Problem (1)
%    e             : evolution of u'*A*v
'''


import numpy as np
import time
from update_cone import update_cone


def AlternatingOptimization(A, options=None):
    start_time = time.time()
    m, n = A.shape
    if options is None:
        options = dict()
    
    # Parameters
    options.setdefault('maxiter', 500)
    options.setdefault('accuracy', 1e-6)
    
    
    # Cone structure
    if 'cone' not in options:
        print('Warning: No cones specified: nonnegative orthant is used')
        if 'P' not in options.get('cone', {}):
            options.setdefault('cone', dict())
            options['cone'].setdefault('P', 'nonnegort')
        if 'Q' not in options.get('cone', {}):
            options['cone'].setdefault('Q', 'nonnegort')
    # G matrix for P
    if 'G' not in options:
        if options['cone']['P'] == 'generator':
            raise ValueError('No generators specified for P.')
        elif options['cone']['P'] == 'facetsrep':
            raise ValueError('No facets specified for P.')
        else:
            options['G'] = None
    # H matrix for Q
    if 'H' not in options:
        if options['cone']['Q'] == 'generator':
            raise ValueError('No generators specified for Q.')
        elif options['cone']['Q'] == 'facetsrep':
            raise ValueError('No facets specified for Q.')
        else:
            options['H'] = None
    # g for facetsrep
    if options['cone']['P'] == 'facetsrep':
        if 'g' not in options:
            print('Warning: No right hand side g specified for P: zero is used')
            options['g'] = np.zeros((options['G'].shape[0], 1))
    else:
        options['g'] = None
    # h for facetsrep
    if options['cone']['Q'] == 'facetsrep':
        if 'h' not in options:
            print('Warning: No right hand side h specified for Q: zero is used')
            options['h'] = np.zeros((options['H'].shape[0], 1))
    else:
        options['h'] = None
    options.setdefault('display', 0)
   
    # Extrapolation parameters
    options.setdefault('beta', 0.5)
    options.setdefault('eta', 2)
    options.setdefault('gamma', 1.05)
    
    # Initialization
    if 'v0' not in options:
        u0 = np.random.randn(m)
        print(f"u0 : {u0}")
        options['v0'] = update_cone(A.T @ u0, options['cone']['Q'], options['H'], options['h'])
    v0 = options['v0']
    print(f"v0 : {v0}")
    v = np.zeros_like(v0)
    ve = v0.copy()
    vp = np.zeros_like(v0)
    u = np.zeros(m)
    ue = np.zeros(m)
    up = np.zeros(m)
    i = 1
    beta_p = options['beta']
    rs = 0
    e = []
    if options['display'] == 1:
        #print('Evolution of iteration number and the objective:')
        tdelay = 0.05
        numprint = 0
    
    # Iterative optimization loop
    while (
        i <= options['maxiter']
        and (rs == 1 or np.linalg.norm(v - vp) >= options['accuracy'] or np.linalg.norm(u - up) >= options['accuracy']
        or (len(e) < 2 or abs(e[-2] - e[-1]) >= options['accuracy'] * abs(e[-2])))
    ):
        rs = 0
        up = u.copy()
        vp = v.copy()
        # Update u
        u = update_cone(A @ ve, options['cone']['P'], options['G'], options['g'])
        # Extrapolate u
        if i >= 2:
            ue = u + options['beta'] * (u - up)
        else:
            ue = u.copy()
        # Update v
        Atu = A.T @ ue
        v = update_cone(Atu, options['cone']['Q'], options['H'], options['h'])
        # Record error
        e.append(float(u.T @ A @ v))
        # Extrapolate v
        if i >= 2:
            ve = v + options['beta'] * (v - vp)
        else:
            ve = v.copy()
        # Update extrapolation parameters
        if len(e) >= 2 and e[-2] < e[-1] and options['beta'] > 0:    # Attention à l'indice i : en python, premier élément d'une liste est e[0] mais en matlab e(1)
            beta_p = options['beta'] / options['eta']
            options['beta'] = 0
            u = up.copy()
            v = vp.copy()
            ve = vp.copy()
            e[-1] = e[-2]
            rs = 1
            # Next step will not extrapolate
        else:
            options['beta'] = min(1, beta_p * options['gamma'])
            beta_p = options['beta']
        # Display error
        if options['display'] == 1 and (time.time() - start_time) >= tdelay * (2 ** numprint):
            #print(f'{i:3d}: {e[i-1]:.4f}...', end='')
            numprint += 1
            if numprint % 5 == 0:
                print('\n')
        i += 1
    if options['display'] == 1:
        print('\n')
    
    # Output
    return u, v, e
