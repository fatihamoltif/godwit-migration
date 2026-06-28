"""Ré-échantillonnage de trajectoires sur grille temporelle régulière,
par interpolation sphérique (slerp) le long du grand cercle."""

import numpy as np
import pandas as pd


def _to_unit_vectors(lat, lon):
    """lat/lon en degrés (arrays) -> matrice (n,3) de vecteurs unitaires sur la sphère."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def _slerp(v1, v2, t):
    """Interpolation sphérique entre deux vecteurs unitaires, fraction t dans [0,1]."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-9:
        v = (1 - t) * v1 + t * v2
    else:
        s = np.sin(omega)
        v = (np.sin((1 - t) * omega) / s) * v1 + (np.sin(t * omega) / s) * v2
    return v / np.linalg.norm(v)


def resample_leg(leg, step="6h", time_col="Location_Timestamp_UTC",
                 lat_col="Latitude", lon_col="Longitude"):
    """Ré-échantillonne un leg (trié par temps) sur une grille alignée sur les heures
    rondes UTC, par slerp. Renvoie un DataFrame [time_col, lat_col, lon_col]."""
    leg = leg.sort_values(time_col).reset_index(drop=True)
    t_real = leg[time_col].values.astype("datetime64[ns]").astype("int64")
    t_start = pd.Timestamp(leg[time_col].iloc[0]).ceil(step)
    t_end   = pd.Timestamp(leg[time_col].iloc[-1]).floor(step)
    if t_start > t_end:
        return pd.DataFrame(columns=[time_col, lat_col, lon_col])
    grid = pd.date_range(t_start, t_end, freq=step)
    grid_ns = grid.values.astype("datetime64[ns]").astype("int64")
    idx = np.clip(np.searchsorted(t_real, grid_ns, side="right"), 1, len(t_real) - 1)
    i0, i1 = idx - 1, idx
    t0, t1 = t_real[i0], t_real[i1]
    frac = np.where(t1 > t0, (grid_ns - t0) / (t1 - t0), 0.0)

    V = _to_unit_vectors(leg[lat_col].values, leg[lon_col].values)
    out_lat, out_lon = np.empty(len(grid)), np.empty(len(grid))
    for k in range(len(grid)):
        v = _slerp(V[i0[k]], V[i1[k]], frac[k])
        out_lat[k] = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
        out_lon[k] = np.degrees(np.arctan2(v[1], v[0]))

    return pd.DataFrame({time_col: grid, lat_col: out_lat, lon_col: out_lon})
