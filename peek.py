import xarray as xr

ds = xr.open_dataset("data/raw/era5/test_10m_2008-09-15.nc")
print(ds)


speed = (ds["u10"] ** 2 + ds["v10"] ** 2) ** 0.5
print("\nVitesse du vent (m/s) — min/moyenne/max :")
print(float(speed.min()), float(speed.mean()), float(speed.max()))
