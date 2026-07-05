import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.data.windowing import flight_segments, make_windows
from godwit.models.baselines import predict_great_circle
from godwit.evaluation.metrics import displacement_errors

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)
X, y, meta = make_windows(segs)

# Prédiction géométrique du grand cercle (dlat, dlon)
gc = predict_great_circle(X)

# Résidu = déplacement vrai - déplacement grand cercle
res = y - gc
res[:,1] = (res[:,1] + 180) % 360 - 180

print("Amplitude des cibles (degrés) :")
print(f"  déplacement BRUT  : dlat std {y[:,0].std():.3f}   dlon std {y[:,1].std():.3f}")
print(f"  RÉSIDU (vs GC)    : dlat std {res[:,0].std():.3f}   dlon std {res[:,1].std():.3f}")

# En km : quelle est l'amplitude typique du résidu à prédire ?
zero = np.zeros(len(res))
res_km = displacement_errors(meta["last_lat"].values, meta["last_lon"].values,
                             meta["last_lat"].values + res[:,0],
                             meta["last_lon"].values + res[:,1])
print(f"\nAmplitude du résidu à apprendre : médiane {np.median(res_km):.1f} km, "
      f"moy {res_km.mean():.1f} km")
print("(c'est CA que le LSTM devra prédire — bien plus petit que les ~290 km du pas complet)")
