import numpy as np

def alternating_optimization(A, P_proj, Q_proj, v0, max_iter=500, tol=1e-6):
    """
    Résolution du problème min_{u in P, v in Q} u^T A v
    par l'optimisation alternée sans extrapolation.
    P_proj: fonction de projection sur le cône P et sphère unité,
    Q_proj: fonction de projection sur le cône Q et sphère unité,
    v0: initialisation dans Q, ||v0||=1
    """
    v = v0
    for k in range(max_iter):
        # Mise à jour de u
        u = P_proj(A @ v)
        # Mise à jour de v
        v_new = Q_proj(A.T @ u)
        # Critère d'arrêt
        if np.linalg.norm(v - v_new) < tol:
            break
        v = v_new
    return u, v


def extrapolated_ao(A, P_proj, Q_proj, v0, max_iter=500, tol=1e-6, gamma_init=0.5, gamma_inc=1.05, gamma_dec=2):
    """
    Version extrapolée de l'optimisation alternée pour min_{u in P, v in Q} u^T A v.
    P_proj, Q_proj : fonctions de projection sur les cônes et sphères unitaires.
    v0 : point initial.
    gamma_* : paramètres d'extrapolation.
    """
    v = v0
    gamma = gamma_init
    u_prev, v_prev = None, None
    for k in range(max_iter):
        # Mise à jour de u
        u = P_proj(A @ v)
        # Extrapolation sur u
        if u_prev is not None:
            u_e = u + gamma * (u - u_prev)
        else:
            u_e = u
        u_prev = u
        # Mise à jour de v
        v = Q_proj(A.T @ u_e)
        # Extrapolation sur v
        if v_prev is not None:
            v_e = v + gamma * (v - v_prev)
        else:
            v_e = v
        v_prev = v
        # Critère d'arrêt
        if np.linalg.norm(v - v_prev) < tol:
            break
        # Mise à jour gamma
        # Si la valeur de l'objectif augmente : restart (décroître gamma)
        # Si elle diminue : augmenter gamma
        # (cette partie dépend de l'implémentation du calcul de l'objectif)
    return u, v
