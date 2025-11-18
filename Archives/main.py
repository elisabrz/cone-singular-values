import numpy as np
from algo_alternating import alternating_optimization, extrapolated_ao

# Exemple de projection sur un cône polyédral et la sphère unité
def cone_sphere_proj(x, G):
    # Projeter x sur le cône {Gx | x >=0} puis sur la sphère unité
    # Implémenter la projection en fonction de la structure de G
    x_proj = G @ np.maximum(np.linalg.pinv(G) @ x, 0)  # Simple projection polyédral
    x_proj = x_proj / np.linalg.norm(x_proj)
    return x_proj

def main():
    # Paramètres du problème
    m, n = 30, 20
    np.random.seed(42)
    A = np.random.randn(m, n)
    
    # Cônes polyédraux générés aléatoirement (G et H matrice de générateurs)
    G = np.eye(m)
    H = np.eye(n)
    
    # Fonctions de projection pour P et Q
    P_proj = lambda x: cone_sphere_proj(x, G)
    Q_proj = lambda x: cone_sphere_proj(x, H)
    
    # Initialisation sur Q (v0)
    v0 = np.random.randn(n)
    v0 = Q_proj(v0)
    
    # Appel de la méthode sans extrapolation
    u_ao, v_ao = alternating_optimization(A, P_proj, Q_proj, v0)
    print("Résultat AO (sans extrapolation) :", u_ao.T @ A @ v_ao)
    
    # Appel de la méthode extrapolée
    u_eao, v_eao = extrapolated_ao(A, P_proj, Q_proj, v0)
    print("Résultat E-AO (extrapolée) :", u_eao.T @ A @ v_eao)

if __name__ == "__main__":
    main()
