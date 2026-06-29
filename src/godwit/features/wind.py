"""Décomposition du vent par rapport au cap de l'oiseau :
tailwind (composante le long du cap) et crosswind (composante perpendiculaire)."""

import numpy as np


def wind_components_along_track(heading_deg, wind_u, wind_v):
    """Projette le vent (u=est, v=nord) sur le cap.
    Renvoie (tailwind, crosswind) en m/s.
      tailwind > 0  : vent porteur (dans le sens du déplacement)
      tailwind < 0  : vent de face
      crosswind : positif = déporté vers la droite du cap.
    """
    h = np.radians(heading_deg)
    # Vecteur unitaire DANS la direction du cap, en repère (est, nord)
    head_e, head_n = np.sin(h), np.cos(h)
    # Vecteur unitaire perpendiculaire (à 90° à droite du cap)
    perp_e, perp_n = np.cos(h), -np.sin(h)

    tailwind  = wind_u * head_e + wind_v * head_n   # produit scalaire vent · cap
    crosswind = wind_u * perp_e + wind_v * perp_n   # produit scalaire vent · perpendiculaire
    return tailwind, crosswind
