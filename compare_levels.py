import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import xarray as xr
from godwit.features.kinematics import add_kinematics
from godwit.features.wind import wind_components_along_track

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
flights = df[df["ground_speed_kmh"] > 15].copy()
t = flights["Location_Timestamp_UTC"]
m = flights[(t.dt.year == 2008) & (t.dt.month == 5)].copy()
m["lon360"] = m["Longitude"] % 360
m["gs_ms"] = m["ground_speed_kmh"] / 3.6

ds = xr.open_dataset("data/raw/era5_flights/era5_2008_05.nc")

def tailwind_at(level):
    d = ds.sel(pressure_level=level)
    sub = d.sel(latitude=xr.DataArray(m["Latitude"].values, dims="pts"),
                longitude=xr.DataArray(m["lon360"].values, dims="pts"),
                valid_time=xr.DataArray(m["Location_Timestamp_UTC"].values, dims="pts"),
                method="nearest")
    tw, cw = wind_components_along_track(m["heading_deg"].values,
                                         sub["u"].values, sub["v"].values)
    return tw, cw

res = {}
for lvl in [925, 850]:
    tw, cw = tailwind_at(lvl)
    res[lvl] = tw
    print(f"=== {lvl} hPa ===")
    print(f"  tailwind moyen        : {np.nanmean(tw):+.2f} m/s")
    print(f"  % pas vent porteur    : {100*np.nanmean(tw>0):.0f} %")
    print(f"  crosswind moy (abs)   : {np.nanmean(np.abs(cw)):.2f} m/s")
    print(f"  corr tailwind/vit-sol : {np.corrcoef(tw, m['gs_ms'])[0,1]:+.2f}\n")

# Hypothèse "l'oiseau choisit son altitude" : meilleur des deux niveaux, pas par pas
tw_best = np.maximum(res[925], res[850])
print("=== Niveau le plus favorable par pas (max 925/850) ===")
print(f"  tailwind moyen        : {np.nanmean(tw_best):+.2f} m/s")
print(f"  % pas vent porteur    : {100*np.nanmean(tw_best>0):.0f} %")
print(f"  corr tailwind/vit-sol : {np.corrcoef(tw_best, m['gs_ms'])[0,1]:+.2f}")
