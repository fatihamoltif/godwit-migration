import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import xarray as xr
from godwit.features.kinematics import add_kinematics

LEVEL = 925

# --- Pas de vol de mai 2008 (le mois qu'on vient d'inspecter) ---
df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
flights = df[df["ground_speed_kmh"] > 15].copy()
t = flights["Location_Timestamp_UTC"]
m = flights[(t.dt.year == 2008) & (t.dt.month == 5)].copy()
print(f"Pas de vol en mai 2008 : {len(m)}")

# --- Charger le cube de vent de ce mois ---
ds = xr.open_dataset("data/raw/era5_flights/era5_2008_05.nc").sel(pressure_level=LEVEL)

# --- Convertir la longitude oiseau en 0-360 (repère du cube ERA5) ---
m["lon360"] = m["Longitude"] % 360

# --- Jointure : pour chaque pas, le u/v au point/instant le plus proche ---
target_lat = xr.DataArray(m["Latitude"].values,  dims="pts")
target_lon = xr.DataArray(m["lon360"].values,    dims="pts")
target_t   = xr.DataArray(m["Location_Timestamp_UTC"].values, dims="pts")

sub = ds.sel(latitude=target_lat, longitude=target_lon,
             valid_time=target_t, method="nearest")

m["wind_u"] = sub["u"].values
m["wind_v"] = sub["v"].values
m["wind_speed"] = np.sqrt(m["wind_u"]**2 + m["wind_v"]**2)

print("\nAperçu de la jointure :")
print(m[["Location_Timestamp_UTC", "Latitude", "lon360",
         "ground_speed_kmh", "heading_deg", "wind_u", "wind_v", "wind_speed"]]
      .head(10).to_string(index=False))

print("\nVent (m/s) sur ces pas de vol — min/moy/max :")
print(round(m["wind_speed"].min(),1), round(m["wind_speed"].mean(),1), round(m["wind_speed"].max(),1))
print("NaN éventuels (pas hors boîte) :", int(m["wind_speed"].isna().sum()))
