from pathlib import Path
import cdsapi


output_dir = Path("data/raw/era5")
output_dir.mkdir(parents=True, exist_ok=True)
target = output_dir / "test_10m_2008-09-15.nc"

dataset = "reanalysis-era5-single-levels"

request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "year": ["2008"],
    "month": ["09"],
    "day": ["15"],
    "time": ["00:00", "06:00", "12:00", "18:00"],  # pas de 6 h
    "area": [62, -168, 58, -160],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

client = cdsapi.Client()
client.retrieve(dataset, request, str(target))

print(f"Terminé ! Fichier écrit dans : {target}")
