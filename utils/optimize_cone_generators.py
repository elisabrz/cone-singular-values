"""
Working cone projection with Gurobi

Key insights from diagnostic:
- Use <= 1 constraint (works well, fast convergence)
- Normalize result to get ||u|| = 1
- For MAXIMIZATION of <u, Av>, minimize <u, -Av>
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def optimize_cone_generators(G, Av, timelimit=10):
    """
    Solve: max <u, Av> s.t. u = G*x, x >= 0, ||u|| = 1
    
    Equivalent to: min <u, -Av> s.t. u = G*x, x >= 0, ||u|| = 1
    
    So we pass -Av as the direction to minimize.
    
    Parameters
    ----------
    G : ndarray (m x n)
        Generator matrix  
    Av : ndarray (m,)
        Direction vector (pass -Av for maximization)
    timelimit : float
        Gurobi time limit
    
    Returns
    -------
    u : ndarray
        Optimal unit vector
    info : dict
        Results info
    """
    m, n = G.shape
    
    try:
        # Create model
        model = gp.Model("cone_projection")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', timelimit)
        model.setParam('NonConvex', 2)
        
        # Variables: x >= 0
        x = model.addMVar(n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
        
        # Objective: min (G^T * Av) @ x
        # Note: Av should already have the correct sign from caller
        obj_coeffs = G.T @ Av
        model.setObjective(obj_coeffs @ x, GRB.MINIMIZE)
        
        # Constraint: ||G*x||² <= 1
        GtG = G.T @ G
        model.addConstr(x @ GtG @ x <= 1.0, "norm_constraint")
        
        # Solve
        model.optimize()
        
        # Extract solution
        if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            x_sol = x.X
            u = G @ x_sol
            norm_u = np.linalg.norm(u)
            
            if norm_u > 1e-8:
                # Normalize to ||u|| = 1
                u_normalized = u / norm_u
                
                return u_normalized, {
                    'status': 'optimal' if model.status == GRB.OPTIMAL else 'suboptimal',
                    'objective': u_normalized @ Av,
                    'norm_before': norm_u,
                    'gurobi_status': model.status
                }
        
        # Fallback if solution is zero or invalid
        return fallback_best_generator(G, Av)
        
    except Exception as e:
        # Any Gurobi error
        print(f"Gurobi error: {e}")
        return fallback_best_generator(G, Av)


def fallback_best_generator(G, Av):
    """
    Fallback: select best single generator
    """
    obj = G.T @ Av
    best_idx = np.argmin(obj)
    
    u = G[:, best_idx]
    u = u / np.linalg.norm(u)
    
    return u, {
        'status': 'fallback',
        'objective': u @ Av,
        'generator_index': best_idx
    }