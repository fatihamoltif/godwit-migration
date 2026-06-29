import sys; sys.path.insert(0, "src")
import pandas as pd
from godwit.features.kinematics import add_kinematics
from godwit.ingest.download_era5_flights import download_flight_months

FLIGHT_KMH = 15

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
flights = df[df["ground_speed_kmh"] > FLIGHT_KMH].copy()

files = download_flight_months(flights)
print(f"\nTerminé : {len(files)} fichiers dans data/raw/era5_flights/")
