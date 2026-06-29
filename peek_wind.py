import xarray as xr

ds = xr.open_dataset("data/raw/era5_flights/era5_2008_05.nc")
print(ds)

print("\n--- Niveaux de pression présents ---")
print(ds["pressure_level"].values)

print("\n--- Vent à 925 hPa, premier pas de temps : min/moy/max de la vitesse ---")
u = ds["u"].sel(pressure_level=925).isel(valid_time=0)
v = ds["v"].sel(pressure_level=925).isel(valid_time=0)
speed = (u**2 + v**2) ** 0.5
print(f"{float(speed.min()):.1f} / {float(speed.mean()):.1f} / {float(speed.max()):.1f} m/s")
