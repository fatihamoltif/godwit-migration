import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.features.kinematics import add_kinematics

FLIGHT_KMH = 15

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
df["is_flight"] = df["ground_speed_kmh"] > FLIGHT_KMH

segments = []
for leg_uid, g in df.groupby("leg_uid", sort=False):
    g = g.reset_index(drop=True)
    run_id = (g["is_flight"] != g["is_flight"].shift()).cumsum()
    for _, run in g.groupby(run_id):
        if run["is_flight"].iloc[0]:                 # ne garder que les runs EN VOL
            segments.append({"leg_uid": leg_uid, "length": len(run)})

seg = pd.DataFrame(segments)

print(f"Nombre de segments de vol continu : {len(seg)}")
print(f"Pas en vol au total               : {int(seg['length'].sum())}")
print("\nDistribution des longueurs (en pas de 6 h) :")
print(seg["length"].describe())

print("\nCombien de segments et de pas garde-t-on selon la longueur minimale ?")
print(f"{'min_len':>7} | {'nb segments':>11} | {'pas gardés':>10} | {'exemples (len-4)':>16}")
for min_len in [2, 3, 4, 5, 6, 8, 10]:
    keep = seg[seg["length"] >= min_len]
    exemples = int((keep["length"] - 4).clip(lower=0).sum())
    print(f"{min_len:>7} | {len(keep):>11} | {int(keep['length'].sum()):>10} | {exemples:>16}")
