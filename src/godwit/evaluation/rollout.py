"""Prédiction d'une trajectoire complète en 'roue libre' (rollout autorégressif) :
on prédit chaque pas à partir de nos propres prédictions précédentes."""

import numpy as np
from godwit.geo import initial_bearing, haversine_km


def great_circle_rollout(seg, window=4, step_hours=6):
    """Depuis les `window` premiers vrais pas d'un segment, prédit tout le reste
    en roue libre avec le modèle grand cercle (cap+vitesse constants du dernier pas).
    Renvoie (lat_pred, lon_pred) pour l'ensemble du segment."""
    lat = seg["Latitude"].values.copy()
    lon = seg["Longitude"].values.copy()
    n = len(seg)

    # on garde les `window` premiers pas réels comme amorce
    pred_lat = list(lat[:window])
    pred_lon = list(lon[:window])

    # cap et vitesse estimés sur les derniers pas RÉELS de l'amorce
    R = 6371.0
    for i in range(window, n):
        # cap et vitesse du dernier segment prédit (roue libre)
        b = initial_bearing(pred_lat[-2], pred_lon[-2], pred_lat[-1], pred_lon[-1])
        d_km = haversine_km(pred_lat[-2], pred_lon[-2], pred_lat[-1], pred_lon[-1])
        d = d_km / R                                  # distance angulaire
        lat1, lon1, th = np.radians(pred_lat[-1]), np.radians(pred_lon[-1]), np.radians(b)
        lat2 = np.arcsin(np.sin(lat1)*np.cos(d) + np.cos(lat1)*np.sin(d)*np.cos(th))
        lon2 = lon1 + np.arctan2(np.sin(th)*np.sin(d)*np.cos(lat1),
                                 np.cos(d) - np.sin(lat1)*np.sin(lat2))
        pred_lat.append(np.degrees(lat2))
        pred_lon.append((np.degrees(lon2) + 180) % 360 - 180)

    return np.array(pred_lat), np.array(pred_lon)
