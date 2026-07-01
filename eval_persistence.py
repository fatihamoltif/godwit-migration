import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.data.windowing import flight_segments, make_windows
from godwit.models.baselines import predict_persistence
from godwit.evaluation.metrics import displacement_errors

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)
X, y, meta = make_windows(segs)

# Prédiction de déplacement
y_pred = predict_persistence(X)

# Reconstruire la position prédite = dernière position connue + déplacement prédit
pred_lat = meta["last_lat"].values + y_pred[:, 0]
pred_lon = meta["last_lon"].values + y_pred[:, 1]

# Erreur de position (km) via Haversine, contre la vérité
err = displacement_errors(pred_lat, pred_lon,
                          meta["true_lat"].values, meta["true_lon"].values)

print("=== Baseline : PERSISTANCE (vitesse constante) ===")
print(f"Exemples évalués   : {len(err)}")
print(f"ADE (erreur moy)   : {err.mean():.1f} km")
print(f"Erreur médiane     : {np.median(err):.1f} km")
print(f"Erreur P90         : {np.percentile(err, 90):.1f} km")
print(f"Erreur max         : {err.max():.1f} km")

# Pour référence : quelle distance parcourt l'oiseau en un pas de 6h ?
step_km = displacement_errors(meta["last_lat"].values, meta["last_lon"].values,
                              meta["true_lat"].values, meta["true_lon"].values)
print(f"\nPour référence — distance réelle parcourue en 1 pas : "
      f"médiane {np.median(step_km):.0f} km, moy {step_km.mean():.0f} km")
print(f"Erreur relative (ADE / pas moyen) : {100*err.mean()/step_km.mean():.0f} %")
