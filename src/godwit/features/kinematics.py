"""Calcul des quantités cinématiques pas-à-pas (longueur, cap, vitesse, virage),
appliqué LEG PAR LEG pour ne jamais calculer à travers une frontière de leg."""

import numpy as np
import pandas as pd

from godwit.geo import haversine_km, initial_bearing

STEP_HOURS = 6


def add_kinematics(df, leg_col="leg_uid", lat_col="Latitude",
                   lon_col="Longitude", time_col="Location_Timestamp_UTC"):
    df = df.sort_values([leg_col, time_col]).copy()
    parts = []
    for _, g in df.groupby(leg_col, sort=False):
        g = g.copy()
        lat, lon = g[lat_col].values, g[lon_col].values
        # position précédente (décalée d'un cran à l'intérieur du leg)
        lat_prev, lon_prev = np.roll(lat, 1), np.roll(lon, 1)

        step = haversine_km(lat_prev, lon_prev, lat, lon)
        head = initial_bearing(lat_prev, lon_prev, lat, lon)
        step[0], head[0] = np.nan, np.nan        # 1er point : pas de prédécesseur

        g["step_km"] = step
        g["heading_deg"] = head
        g["ground_speed_kmh"] = step / STEP_HOURS

        # angle de virage = variation de cap entre deux pas, ramenée dans [-180, 180]
        turn = np.diff(head, prepend=np.nan)
        g["turning_deg"] = (turn + 180) % 360 - 180

        parts.append(g)
    return pd.concat(parts, ignore_index=True)
