"""
Project on norm1 nonnegative orthant

On veut trouver le vecteur u qui maximise le produit scalaire avec Av
u ≥ 0 (toutes les composantes sont positives ou nulles) et ||u|| = 1

max_{u>=0,||u||=1} u^T Av

Pour que ce produit soit maximum, on veut que u[i] soit grand quand Av[i] est négatif
Donc on prend -Av (pour inverser les signes) et on garde uniquement les valeurs positives
Si Av n'a que des valeurs positives, alors u serait complètement nul.
On cherche alors l'indice où Av est le moins positif (ou le plus négatif)
On met u[b] = 1 sur cet indice seulement
"""

import numpy as np


def projectNonnegOrthnorm1(Av):
    """
    Project on norm1 nonnegative orthant
    
    Parameters
    ----------
    Av : ndarray
        Vector to project
    
    Returns
    -------
    u : ndarray
        Projected vector with ||u||=1 and u>=0
    """
    u = np.maximum(0, -Av)
    
    if np.linalg.norm(u) <= 1e-9:
        b = np.argmax(-Av)
        u[b] = 1
    else:
        u = u / np.linalg.norm(u)
    
    return u