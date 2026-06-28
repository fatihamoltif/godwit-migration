import sys; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
import folium
from godwit.preprocessing.resample import resample_leg

path = "data/raw/usgs/barTailedGodwit_USGS_ASC_argos_diag_filteredLocations.csv"
df = pd.read_csv(path, low_memory=False)
df["Location_Timestamp_UTC"] = pd.to_datetime(
    df["Location_Timestamp_UTC"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
clean = df[(df["Tracking_Status"] == "alive") & (df["Location_DAF_Filter"] == 0)].copy()
clean = clean.dropna(subset=["Latitude", "Longitude"]).sort_values(
    ["Animal_ID", "Location_Timestamp_UTC"]).reset_index(drop=True)

bird_id = clean["Animal_ID"].value_counts().index[0]
bird = clean[clean["Animal_ID"] == bird_id].copy().reset_index(drop=True)
gap_h = bird["Location_Timestamp_UTC"].diff().dt.total_seconds() / 3600
bird["leg"] = (gap_h > 48).cumsum()
biggest = bird["leg"].value_counts().index[0]
leg = bird[bird["leg"] == biggest].reset_index(drop=True)

res = resample_leg(leg)

span_h = (leg["Location_Timestamp_UTC"].iloc[-1] - leg["Location_Timestamp_UTC"].iloc[0]).total_seconds()/3600
print(f"Oiseau {bird_id}, plus gros leg :")
print(f"  durée du leg            : {span_h:.0f} h  (~{span_h/24:.1f} jours)")
print(f"  points bruts            : {len(leg)}")
print(f"  points ré-échantillonnés: {len(res)}  (attendu ~ {span_h/6:.0f})")
print("\nPremières positions sur la grille 6 h :")
print(res.head().to_string(index=False))
m = folium.Map(location=[leg["Latitude"].mean(), leg["Longitude"].mean()],
               zoom_start=3, tiles="CartoDB positron")
for _, r in leg.iterrows():
    folium.CircleMarker([r["Latitude"], r["Longitude"]], radius=2,
                        color="gray", fill=True, fill_opacity=0.5).add_to(m)
for _, r in res.iterrows():
    folium.CircleMarker([r["Latitude"], r["Longitude"]], radius=4,
                        color="red", fill=True, fill_opacity=0.9,
                        popup=str(r["Location_Timestamp_UTC"])).add_to(m)

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
f = out / f"resampled_{bird_id}_leg{biggest}.html"
m.save(str(f))
print(f"\nCarte enregistrée : {f}")
