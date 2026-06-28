import xarray as xr

ds = xr.open_dataset("data/raw/era5/test_10m_2008-09-15.nc")
point = ds.sel(latitude=60.3, longitude=-164.7, method="nearest") \
          .sel(valid_time="2008-09-15T12:00")

print("Point de grille retenu :",
      float(point.latitude), float(point.longitude))
print("Vent u, v (m/s) :", float(point["u10"]), float(point["v10"]))
