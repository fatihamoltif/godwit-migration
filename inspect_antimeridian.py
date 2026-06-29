import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.features.kinematics import add_kinematics

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
flights = df[df["ground_speed_kmh"] > 15].copy()

# Longitudes en repère 0-360 (au lieu de -180/180)
lon360 = flights["Longitude"] % 360

print("=== Longitudes en repère -180/180 (brut) ===")
print(f"  min {flights['Longitude'].min():.1f}  /  max {flights['Longitude'].max():.1f}")
print("  -> trompeur : -180 et +180 sont en réalité le MÊME méridien (Pacifique central)")

print("\n=== Longitudes en repère 0-360 ===")
print(f"  min {lon360.min():.1f}  /  max {lon360.max():.1f}")
print(f"  étendue réelle : {lon360.max() - lon360.min():.1f}° de longitude")

print("\n=== Où sont vraiment les vols ? (comptage par tranche de 30° en 0-360) ===")
bins = pd.cut(lon360, bins=range(0, 361, 30))
print(bins.value_counts().sort_index().to_string())
