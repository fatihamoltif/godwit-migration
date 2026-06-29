"""Télécharge le vent ERA5 (pressure-levels 925/850 hPa) sur les mois et zones
où les barges ont effectivement volé. Une requête par mois-de-vol, boîte ajustée.

Source : ERA5 hourly data on pressure levels, Copernicus Climate Change Service (C3S).
"""

from pathlib import Path
import cdsapi

DATASET = "reanalysis-era5-pressure-levels"
LEVELS = ["925", "850"]
MARGIN_DEG = 3.0          # marge autour de la boîte des vols du mois
GRID = [0.25, 0.25]


def _round_box(lat_min, lat_max, lon360_min, lon360_max, margin=MARGIN_DEG):
    """Boîte [Nord, Ouest, Sud, Est] arrondie à 0.25°, avec marge. Lon en 0-360."""
    n = min(90,  lat_max + margin)
    s = max(-90, lat_min - margin)
    w = lon360_min - margin
    e = lon360_max + margin
    # arrondir au quart de degré (la grille native)
    snap = lambda x: round(x * 4) / 4
    return [snap(n), snap(w), snap(s), snap(e)]


def download_flight_months(flights, out_dir="data/raw/era5_flights",
                           time_col="Location_Timestamp_UTC",
                           lat_col="Latitude", lon_col="Longitude"):
    """flights : DataFrame des pas EN VOL (déjà filtrés). Télécharge un fichier
    NetCDF par (année, mois) de vol. Renvoie la liste des fichiers produits."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    f = flights.copy()
    f["lon360"] = f[lon_col] % 360
    f["year"]  = f[time_col].dt.year
    f["month"] = f[time_col].dt.month

    produced = []
    groups = list(f.groupby(["year", "month"]))
    print(f"{len(groups)} mois-de-vol à télécharger.\n")

    for (year, month), g in groups:
        target = out / f"era5_{year}_{month:02d}.nc"
        if target.exists():                       # reprise : on saute l'existant
            print(f"  [skip] {target.name} déjà présent")
            produced.append(target); continue

        area = _round_box(g[lat_col].min(), g[lat_col].max(),
                          g["lon360"].min(), g["lon360"].max())
        days = sorted(g[time_col].dt.day.unique())
        request = {
            "product_type": ["reanalysis"],
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "pressure_level": LEVELS,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in days],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": area,
            "grid": GRID,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"  {year}-{month:02d} : {len(days)} jours, boîte {area}")
        client.retrieve(DATASET, request, str(target))
        produced.append(target)
        print(f"     -> {target.name}")

    return produced
