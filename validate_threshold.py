import numpy as np
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

GAP_THRESHOLD_H = 48
MIN_LEG_POINTS  = 20

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce")

clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"]).sort_values(
    ["Animal_ID", "Location_Timestamp_UTC"]).reset_index(drop=True)

rows = []
for bird_id, g in clean.groupby("Animal_ID"):
    g = g.copy()
    gap_h = g["Location_Timestamp_UTC"].diff().dt.total_seconds() / 3600
    leg_id = (gap_h > GAP_THRESHOLD_H).cumsum()
    sizes = g.groupby(leg_id).size()
    good = sizes[sizes >= MIN_LEG_POINTS]
    rows.append({
        "bird": bird_id,
        "points": len(g),
        "legs_total": len(sizes),
        "legs_ok": len(good),
        "points_kept": int(good.sum()),
    })

summary = pd.DataFrame(rows)

print("=== Récap global avec seuil 48 h / legs >= 20 pts ===")
print(f"Oiseaux total                     : {len(summary)}")
print(f"Oiseaux avec >=1 leg exploitable  : {int((summary['legs_ok'] > 0).sum())}")
print(f"Oiseaux SANS aucun leg exploitable: {int((summary['legs_ok'] == 0).sum())}")
print(f"Legs exploitables (total)         : {int(summary['legs_ok'].sum())}")
print(f"Points conservés / points propres : {int(summary['points_kept'].sum())} / {int(summary['points'].sum())}"
      f"  ({100*summary['points_kept'].sum()/summary['points'].sum():.0f} %)")

print("\n=== Les 5 oiseaux les plus riches (points conservés) ===")
print(summary.sort_values("points_kept", ascending=False).head().to_string(index=False))

print("\n=== Oiseaux sans leg exploitable (peu de données) ===")
poor = summary[summary["legs_ok"] == 0].sort_values("points", ascending=False)
print(f"{len(poor)} oiseaux — aperçu :")
print(poor.head(10).to_string(index=False))
