import sys; sys.path.insert(0, "src")
import pandas as pd
from godwit.features.kinematics import add_kinematics

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)

cols = ["Animal_ID", "leg_uid", "Location_Timestamp_UTC",
        "step_km", "heading_deg", "ground_speed_kmh", "turning_deg"]
print("Aperçu (début d'un leg — voir les NaN attendus) :")
print(df[cols].head(8).to_string(index=False))

print("\nDistribution de la vitesse-sol (km/h) :")
print(df["ground_speed_kmh"].describe())

print("\n--- Structure activité ---")
print("Pas ~immobiles (<2 km/h)  :", int((df["ground_speed_kmh"] < 2).sum()))
print("Pas en vol      (>15 km/h):", int((df["ground_speed_kmh"] > 15).sum()))
print("Vitesse-sol max observée  :", round(df["ground_speed_kmh"].max(), 0), "km/h")
