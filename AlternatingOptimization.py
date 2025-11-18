import numpy as np
import time

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
        options['v0'] = update_cone(A.T @ u0, options['cone']['Q'], options['H'], options['h'])
    v0 = options['v0']
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
        print('Evolution of iteration number and the objective:')
        tdelay = 0.05
        numprint = 0
    
    # Iterative optimization loop
    while (
        i <= options['maxiter']
        and (rs == 1 or np.linalg.norm(v - vp) >= options['accuracy'] or np.linalg.norm(u - up) >= options['accuracy']
        or (i <= 3 or abs(e[i-2] - e[i-1]) >= options['accuracy'] * abs(e[i-2]) if i >= 3 else True))
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
        if i >= 2 and e[i-2] < e[i-1] and options['beta'] > 0:    # Attention à l'indice i : en python, premier élément d'une liste est e[0] mais en matlab e(1)
            beta_p = options['beta'] / options['eta']
            options['beta'] = 0
            u = up.copy()
            v = vp.copy()
            ve = vp.copy()
            e[i-1] = e[i-2]
            rs = 1
            # Next step will not extrapolate
        else:
            options['beta'] = min(1, beta_p * options['gamma'])
            beta_p = options['beta']
        # Display error
        if options['display'] == 1 and (time.time() - start_time) >= tdelay * (2 ** numprint):
            print(f'{i:3d}: {e[i-1]:.4f}...', end='')
            numprint += 1
            if numprint % 5 == 0:
                print('\n')
        i += 1
    if options['display'] == 1:
        print('\n')
    
    # Output
    return u, v, e
