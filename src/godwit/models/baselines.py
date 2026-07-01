"""Baselines de prédiction de déplacement (dlat, dlon).
Chaque baseline prend X (n, window, n_feat) et renvoie y_pred (n, 2)."""

import numpy as np

# indices des features position dans la fenêtre (cf. make_windows: Lat=0, Lon=1)
LAT_IDX, LON_IDX = 0, 1


def predict_persistence(X):
    """Persistance / vitesse constante : répète le dernier déplacement observé.
    dernier pas de la fenêtre (-1) moins l'avant-dernier (-2)."""
    last  = X[:, -1, [LAT_IDX, LON_IDX]]     # (n, 2) dernière position connue
    prev  = X[:, -2, [LAT_IDX, LON_IDX]]     # (n, 2) avant-dernière
    dlat = last[:, 0] - prev[:, 0]
    dlon = last[:, 1] - prev[:, 1]
    dlon = (dlon + 180) % 360 - 180          # même unwrap antiméridien que dans la cible
    return np.column_stack([dlat, dlon])
def predict_mean_velocity(X, k=3):
    """Vitesse constante lissée : déplacement moyen sur les k derniers pas."""
    pos = X[:, :, [LAT_IDX, LON_IDX]]                # (n, window, 2)
    dpos = np.diff(pos, axis=1)                      # (n, window-1, 2) déplacements
    dpos[:, :, 1] = (dpos[:, :, 1] + 180) % 360 - 180  # unwrap dlon
    mean_d = dpos[:, -k:, :].mean(axis=1)            # moyenne des k derniers
    return mean_d                                    # (n, 2)


def predict_great_circle(X, step_hours=6):
    """Projette le long du grand cercle avec le cap et la vitesse récents.
    Utilise heading (feat 2) et ground_speed km/h (feat 3) du dernier pas."""
    from godwit.geo import haversine_km
    HEAD_IDX, SPEED_IDX = 2, 3
    R = 6371.0
    lat1 = np.radians(X[:, -1, LAT_IDX])
    lon1 = np.radians(X[:, -1, LON_IDX])
    theta = np.radians(X[:, -1, HEAD_IDX])           # cap du dernier pas
    d = X[:, -1, SPEED_IDX] * step_hours / R         # distance angulaire = v*t/R

    # formule de navigation : destination le long du grand cercle
    lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(theta))
    lon2 = lon1 + np.arctan2(np.sin(theta)*np.sin(d)*np.cos(lat1),
                             np.cos(d) - np.sin(lat1)*np.sin(lat2))
    dlat = np.degrees(lat2) - X[:, -1, LAT_IDX]
    dlon = np.degrees(lon2) - X[:, -1, LON_IDX]
    dlon = (dlon + 180) % 360 - 180
    return np.column_stack([dlat, dlon])
