import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import xarray as xr
from godwit.features.kinematics import add_kinematics
from godwit.features.wind import wind_components_along_track

LEVEL = 925

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
flights = df[df["ground_speed_kmh"] > 15].copy()
t = flights["Location_Timestamp_UTC"]
m = flights[(t.dt.year == 2008) & (t.dt.month == 5)].copy()

ds = xr.open_dataset("data/raw/era5_flights/era5_2008_05.nc").sel(pressure_level=LEVEL)
m["lon360"] = m["Longitude"] % 360
sub = ds.sel(latitude=xr.DataArray(m["Latitude"].values, dims="pts"),
             longitude=xr.DataArray(m["lon360"].values, dims="pts"),
             valid_time=xr.DataArray(m["Location_Timestamp_UTC"].values, dims="pts"),
             method="nearest")
m["wind_u"], m["wind_v"] = sub["u"].values, sub["v"].values

# --- Décomposition du vent par rapport au cap ---
m["tailwind"], m["crosswind"] = wind_components_along_track(
    m["heading_deg"], m["wind_u"], m["wind_v"])

# --- Vitesses : convertir la vitesse-sol en m/s pour comparer au vent ---
m["ground_speed_ms"] = m["ground_speed_kmh"] / 3.6
# Vitesse-air ≈ vitesse-sol - tailwind (le long du cap)
m["airspeed_ms"] = m["ground_speed_ms"] - m["tailwind"]

print("=== Vent le long du cap (m/s) ===")
print(f"tailwind  : moy {m['tailwind'].mean():.1f}  (>0 = porteur)")
print(f"  pas avec vent porteur : {int((m['tailwind']>0).sum())} / {len(m)} "
      f"({100*(m['tailwind']>0).mean():.0f} %)")
print(f"crosswind : moy abs {m['crosswind'].abs().mean():.1f}")

print("\n=== La relation clé : vitesse-sol vs vitesse-air ===")
print(f"vitesse-sol moyenne : {m['ground_speed_ms'].mean():.1f} m/s "
      f"({m['ground_speed_ms'].mean()*3.6:.0f} km/h)")
print(f"vitesse-air moyenne : {m['airspeed_ms'].mean():.1f} m/s "
      f"({m['airspeed_ms'].mean()*3.6:.0f} km/h)")
print("  ^ la vitesse-air doit être plus PROCHE de la vitesse de croisière réelle")
print("    d'une barge (~15-20 m/s) que la vitesse-sol, si le vent est porteur.")

corr = m["tailwind"].corr(m["ground_speed_ms"])
print(f"\nCorrélation tailwind <-> vitesse-sol : {corr:+.2f}")
print("  (positive = quand le vent pousse, l'oiseau va plus vite par rapport au sol)")
