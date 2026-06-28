import pandas as pd

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
)

clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"])
clean = clean.sort_values(["Animal_ID", "Location_Timestamp_UTC"])


bird_id = clean["Animal_ID"].value_counts().index[0]
bird = clean[clean["Animal_ID"] == bird_id].copy()

bird["gap_h"] = bird["Location_Timestamp_UTC"].diff().dt.total_seconds() / 3600

print(f"Oiseau : {bird_id} — {len(bird)} points\n")
print("Distribution des intervalles entre points (heures) :")
print(bird["gap_h"].describe())

print("\nQuelques repères :")
print("  intervalle médian :", round(bird["gap_h"].median(), 2), "h")
print("  plus grand trou   :", round(bird["gap_h"].max(), 1), "h")
print("  nb d'intervalles > 24 h :", int((bird["gap_h"] > 24).sum()))
