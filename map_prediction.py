import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import folium
from pathlib import Path
from godwit.data.windowing import flight_segments
from godwit.data.split import train_test_split_by_bird
from godwit.data.windowing import make_windows
from godwit.evaluation.rollout import great_circle_rollout
from godwit.geo import haversine_km

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)

# retrouver les oiseaux de TEST (même split que l'éval)
_, _, meta = make_windows(segs)
_, _, test_birds = train_test_split_by_bird(meta, 0.25, random_state=42)

# choisir, parmi les segments d'oiseaux de test, le plus long (le plus démonstratif)
test_segs = [s for s in segs if s["Animal_ID"].iloc[0] in test_birds]
seg = max(test_segs, key=len).reset_index(drop=True)
bird_id = seg["Animal_ID"].iloc[0]
print(f"Segment choisi : {bird_id}, {len(seg)} pas "
      f"({(len(seg)*6)/24:.1f} jours de vol)")

# rollout grand cercle
plat, plon = great_circle_rollout(seg, window=4)

# erreur finale (dernier point) et moyenne
err = haversine_km(plat, plon, seg["Latitude"].values, seg["Longitude"].values)
print(f"Erreur rollout — finale : {err[-1]:.0f} km | moyenne : {err.mean():.0f} km")

# --- carte ---
# --- carte (longitudes en 0-360 pour éviter l'artefact antiméridien) ---
def to360(lons):
    return np.array(lons) % 360

true_lat = seg["Latitude"].values
true_lon = to360(seg["Longitude"].values)
pred_lon = to360(plon)

center_lat = float(np.mean(true_lat))
center_lon = float(np.mean(true_lon))

m = folium.Map(location=[center_lat, center_lon],
               zoom_start=3, tiles="CartoDB positron")

folium.PolyLine(list(zip(true_lat, true_lon)),
                color="blue", weight=3, opacity=0.8,
                tooltip="Trajectoire réelle").add_to(m)
folium.PolyLine(list(zip(plat, pred_lon)),
                color="red", weight=3, opacity=0.8, dash_array="8",
                tooltip="Prédiction grand cercle (roue libre)").add_to(m)

folium.CircleMarker([true_lat[0], true_lon[0]],
                    radius=6, color="green", fill=True, fill_opacity=1,
                    popup="Départ").add_to(m)
folium.CircleMarker([true_lat[3], true_lon[3]],
                    radius=5, color="orange", fill=True, fill_opacity=1,
                    popup="Fin de l'amorce — début de la prédiction").add_to(m)

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
f = out / f"prediction_map_{bird_id}.html"
m.save(str(f))
print(f"-> {f}")
