import sys; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
from godwit.features.kinematics import add_kinematics
from godwit.features.wind_join import join_wind

FLIGHT_KMH = 15

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
df = add_kinematics(df)
flights = df[df["ground_speed_kmh"] > FLIGHT_KMH].copy()

feat = join_wind(flights)
feat["ground_speed_ms"] = feat["ground_speed_kmh"] / 3.6
feat["airspeed_925"] = feat["ground_speed_ms"] - feat["tailwind_925"]
feat["airspeed_850"] = feat["ground_speed_ms"] - feat["tailwind_850"]

out_dir = Path("data/processed"); out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "flight_features.csv"
feat.to_csv(out_file, index=False)

print(f"Pas de vol enrichis : {len(feat)}")
print(f"Oiseaux             : {feat['Animal_ID'].nunique()}")
print(f"NaN vent (hors boîte): {int(feat['tailwind_925'].isna().sum())}")
print(f"-> {out_file}\n")

print("=== Résultat 'wind selection' à pleine échelle (1412 pas) ===")
for lvl in [925, 850]:
    tw = feat[f"tailwind_{lvl}"]
    print(f"{lvl} hPa : tailwind moy {tw.mean():+.2f} m/s | "
          f"{100*(tw>0).mean():.0f}% porteur | "
          f"corr/vit-sol {tw.corr(feat['ground_speed_ms']):+.2f}")
twb = feat["tailwind_best"]
print(f"best   : tailwind moy {twb.mean():+.2f} m/s | {100*(twb>0).mean():.0f}% porteur")
