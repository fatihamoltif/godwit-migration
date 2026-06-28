import sys; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from godwit.preprocessing.resample import resample_leg

GAP_H, MIN_POINTS, STEP = 48, 20, "6h"

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce")

clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"]).sort_values(
    ["Animal_ID", "Location_Timestamp_UTC"]).reset_index(drop=True)

frames, n_legs = [], 0
for bird_id, g in clean.groupby("Animal_ID"):
    g = g.reset_index(drop=True)
    gap_h = g["Location_Timestamp_UTC"].diff().dt.total_seconds() / 3600
    g["_leg"] = (gap_h > GAP_H).cumsum()
    for leg_local, leg in g.groupby("_leg"):
        if len(leg) < MIN_POINTS:
            continue
        res = resample_leg(leg, step=STEP)
        if res.empty:
            continue
        res.insert(0, "leg_uid", f"{bird_id}_{leg_local}")
        res.insert(0, "Animal_ID", bird_id)
        frames.append(res)
        n_legs += 1

resampled = pd.concat(frames, ignore_index=True)

out_dir = Path("data/interim"); out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "resampled_tracks_6h.csv"
resampled.to_csv(out_file, index=False)

print(f"Legs ré-échantillonnés : {n_legs}")
print(f"Oiseaux                : {resampled['Animal_ID'].nunique()}")
print(f"Points 6 h au total    : {len(resampled)}")
print(f"Enregistré -> {out_file}\n")
print(resampled.head().to_string(index=False))
print("\nPoints par oiseau (top 5) :")
print(resampled.groupby('Animal_ID').size().sort_values(ascending=False).head().to_string())
