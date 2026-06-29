"""Primitives géométriques sur la sphère terrestre (réutilisées partout)."""

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance (km) le long du grand cercle entre deux points lat/lon (degrés)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def initial_bearing(lat1, lon1, lat2, lon2):
    """Cap initial (degrés, 0=Nord, sens horaire) du point 1 vers le point 2."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360
