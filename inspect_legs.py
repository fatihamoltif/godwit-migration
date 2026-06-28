import numpy as np
import pandas as pd

def haversine_km(lat1, lon1, lat2, lon2):
    """Distance en km entre deux points (lat/lon en degrés), le long du grand cercle."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce")

clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"]).sort_values(
    ["Animal_ID", "Location_Timestamp_UTC"])

bird_id = clean["Animal_ID"].value_counts().index[0]
bird = clean[clean["Animal_ID"] == bird_id].copy().reset_index(drop=True)

bird["gap_h"]   = bird["Location_Timestamp_UTC"].diff().dt.total_seconds() / 3600
bird["step_km"] = haversine_km(bird["Latitude"].shift(), bird["Longitude"].shift(),
                               bird["Latitude"], bird["Longitude"])

print(f"Oiseau {bird_id} — {len(bird)} points\n")
print("Sensibilité du découpage en legs selon le seuil de coupure :")
print(f"{'seuil':>6} | {'nb legs':>7} | {'plus gros':>9} | {'legs>=20 pts':>12} | {'pts couverts':>12}")
for thr in [12, 24, 30, 36, 42, 48, 72]:
    leg_id = (bird["gap_h"] > thr).cumsum()
    sizes = bird.groupby(leg_id).size()
    n_legs = len(sizes)
    big = sizes[sizes >= 20]
    print(f"{thr:>4}h  | {n_legs:>7} | {sizes.max():>9} | {len(big):>12} | {int(big.sum()):>12}")
