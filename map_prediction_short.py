import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import folium
from pathlib import Path
from godwit.data.windowing import flight_segments
from godwit.models.baselines import predict_great_circle
from godwit.geo import haversine_km

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)
seg = max((s for s in segs if s["Animal_ID"].iloc[0] == "BARG_Z0"),
          key=len).reset_index(drop=True)

# Prédiction "un pas à la fois" : pour chaque pas i>=4, on prédit depuis les vraies positions
lat = seg["Latitude"].values
lon = seg["Longitude"].values
WINDOW = 4
pred_lat, pred_lon = [], []
for i in range(WINDOW, len(seg)):
    # fenêtre des vraies positions [i-4, i-1], features géo pour le grand cercle
    win = seg.iloc[i-WINDOW:i]
    X = np.zeros((1, WINDOW, 4))
    X[0, :, 0] = win["Latitude"].values
    X[0, :, 1] = win["Longitude"].values
    X[0, :, 2] = win["heading_deg"].values
    X[0, :, 3] = win["ground_speed_kmh"].values
    d = predict_great_circle(X)[0]
    pred_lat.append(lat[i-1] + d[0])
    pred_lon.append(lon[i-1] + d[1])

pred_lat, pred_lon = np.array(pred_lat), np.array(pred_lon)
true_lat, true_lon = lat[WINDOW:], lon[WINDOW:]
err = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
print(f"BARG_Z0 — prédiction 1 pas : ADE {err.mean():.1f} km | médiane {np.median(err):.1f} km")

# carte en 0-360
to360 = lambda a: np.array(a) % 360
m = folium.Map(location=[float(true_lat.mean()), float(to360(true_lon).mean())],
               zoom_start=3, tiles="CartoDB positron")
folium.PolyLine(list(zip(true_lat, to360(true_lon))), color="blue", weight=3,
                opacity=0.8, tooltip="Trajectoire réelle").add_to(m)
# chaque prédiction 1-pas comme petit point rouge
for la, lo in zip(pred_lat, to360(pred_lon)):
    folium.CircleMarker([la, lo], radius=3, color="red", fill=True,
                        fill_opacity=0.8).add_to(m)
folium.CircleMarker([true_lat[0], to360(true_lon)[0]], radius=6, color="green",
                    fill=True, fill_opacity=1, popup="Départ").add_to(m)

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
f = out / "prediction_map_short_BARG_Z0.html"
m.save(str(f))
print(f"-> {f}")
