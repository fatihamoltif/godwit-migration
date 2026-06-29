"""Jointure spatio-temporelle pas-de-vol <-> vent ERA5, mois par mois,
puis décomposition tailwind/crosswind à chaque niveau de pression."""

from pathlib import Path
import xarray as xr
import pandas as pd

from godwit.features.wind import wind_components_along_track


def join_wind(flights, era5_dir="data/raw/era5_flights", levels=(925, 850),
              time_col="Location_Timestamp_UTC",
              lat_col="Latitude", lon_col="Longitude"):
    """Pour chaque pas de vol, récupère le vent (u,v) ERA5 au point/instant le plus
    proche, à chaque niveau de `levels`, et calcule tailwind/crosswind.
    Boucle par mois car chaque fichier ERA5 a sa propre boîte géographique."""
    era5_dir = Path(era5_dir)
    f = flights.copy()
    f["lon360"] = f[lon_col] % 360
    f["_ym"] = list(zip(f[time_col].dt.year, f[time_col].dt.month))

    parts = []
    for (year, month), g in f.groupby("_ym", sort=True):
        ds = xr.open_dataset(era5_dir / f"era5_{year}_{month:02d}.nc")
        # cibles vectorisées (une dimension commune "pts")
        tlat  = xr.DataArray(g[lat_col].values, dims="pts")
        tlon  = xr.DataArray(g["lon360"].values, dims="pts")
        ttime = xr.DataArray(g[time_col].values, dims="pts")

        g = g.copy()
        for lvl in levels:
            sub = ds.sel(pressure_level=lvl).sel(
                latitude=tlat, longitude=tlon, valid_time=ttime, method="nearest")
            u, v = sub["u"].values, sub["v"].values
            tw, cw = wind_components_along_track(g["heading_deg"].values, u, v)
            g[f"wind_u_{lvl}"]    = u
            g[f"wind_v_{lvl}"]    = v
            g[f"tailwind_{lvl}"]  = tw
            g[f"crosswind_{lvl}"] = cw
        ds.close()
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    # tailwind "best" = le niveau le plus favorable, pas par pas
    out["tailwind_best"] = out[[f"tailwind_{l}" for l in levels]].max(axis=1)
    return out.drop(columns=["_ym"])
