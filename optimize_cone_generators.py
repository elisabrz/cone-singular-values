"""
Given G and Av, solve 

min_{u=Gx, x>=0} <u,Av> such that ||u|| <= 1.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def optimize_cone_generators(G, Av, timelimit=None):
    """
    Optimize cone generators
    
    Parameters
    ----------
    G : ndarray
        Generator matrix of shape (m, n)
    Av : ndarray
        Vector to optimize against
    timelimit : float, optional
        Time limit for Gurobi solver
    
    Returns
    -------
    u : ndarray
        Optimal vector
    results : gurobipy.Model
        Gurobi optimization results
    """
    m, n = G.shape
    
    ## Construct model and solve the Gurobi model for (Problem 1) above
    model = gp.Model("optimize_cone_generators")
    
    # Variables: x >= 0
    x = model.addMVar(n, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    
    # Objective: min G'*Av * x
    obj = G.T @ Av
    model.setObjective(obj @ x, GRB.MINIMIZE)
    
    # Constraint: sum(x) = 0 (equivalent to A = sparse(1,n), rhs = 0)
    model.addConstr(gp.quicksum(x[i] for i in range(n)) == 0, "sum_constraint")
    
    # Quadratic constraint: ||Gx|| <= 1
    # x^T * (G^T * G) * x <= 1
    model.addConstr(x @ (G.T @ G) @ x <= 1.0, "norm_constraint")
    
    # Parameters
    if timelimit is not None:
        model.Params.TimeLimit = timelimit
    model.Params.OutputFlag = 0  # display on/off
    
    # Solve
    model.optimize()
    
    # Extract solution
    x_sol = np.array([x[i].X for i in range(n)])
    u = G @ x_sol
    
    if np.linalg.norm(u) < 0.1:  # meaning ||u|| is zero because if it is positive, it is one
        # pick column of G that minimizes G^T Av
        b = np.argmin(obj)
        u = G[:, b]
        u = u / np.linalg.norm(u)
    
    return u, model