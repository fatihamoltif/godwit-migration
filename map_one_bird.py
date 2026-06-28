from pathlib import Path
import pandas as pd
import folium

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
)
clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"])
clean = clean.sort_values(["Animal_ID", "Location_Timestamp_UTC"])
counts = clean["Animal_ID"].value_counts()
print("Top 5 oiseaux par nombre de points plausibles :")
print(counts.head())

bird_id = counts.index[0]
bird = clean[clean["Animal_ID"] == bird_id]
print(f"\nOiseau retenu : {bird_id} — {len(bird)} points, "
      f"du {bird['Location_Timestamp_UTC'].min().date()} au {bird['Location_Timestamp_UTC'].max().date()}")

m = folium.Map(location=[bird["Latitude"].mean(), bird["Longitude"].mean()],
               zoom_start=3, tiles="CartoDB positron")


coords = list(zip(bird["Latitude"], bird["Longitude"]))
folium.PolyLine(coords, weight=2, opacity=0.7).add_to(m)


n = len(bird)
for i, (_, row) in enumerate(bird.iterrows()):
    color = "green" if i == 0 else "red" if i == n - 1 else "blue"
    folium.CircleMarker([row["Latitude"], row["Longitude"]],
                        radius=3, color=color, fill=True, fill_opacity=0.7,
                        popup=str(row["Location_Timestamp_UTC"])).add_to(m)

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
out_file = out / f"trajectory_{bird_id}.html"
m.save(str(out_file))
print(f"\nCarte enregistrée : {out_file}")
