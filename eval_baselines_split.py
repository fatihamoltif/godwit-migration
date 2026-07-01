import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.data.windowing import flight_segments, make_windows
from godwit.data.split import train_test_split_by_bird
from godwit.models.baselines import (predict_persistence,
                                      predict_mean_velocity, predict_great_circle)
from godwit.evaluation.metrics import displacement_errors

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)
X, y, meta = make_windows(segs)

train_mask, test_mask, test_birds = train_test_split_by_bird(meta, test_fraction=0.25)

print(f"Oiseaux en test : {len(test_birds)}  -> {test_birds}")
print(f"Exemples train  : {int(train_mask.sum())}")
print(f"Exemples test   : {int(test_mask.sum())}\n")

X_te, meta_te = X[test_mask], meta[test_mask].reset_index(drop=True)

def evaluate(name, y_pred):
    pred_lat = meta_te["last_lat"].values + y_pred[:, 0]
    pred_lon = meta_te["last_lon"].values + y_pred[:, 1]
    err = displacement_errors(pred_lat, pred_lon,
                              meta_te["true_lat"].values, meta_te["true_lon"].values)
    return {"baseline": name, "ADE_km": err.mean(),
            "médiane_km": np.median(err), "P90_km": np.percentile(err, 90)}

rows = [
    evaluate("persistance",        predict_persistence(X_te)),
    evaluate("vitesse lissée (3)", predict_mean_velocity(X_te)),
    evaluate("grand cercle",       predict_great_circle(X_te)),
]
res = pd.DataFrame(rows).set_index("baseline").round(1)
print("=== Baselines évaluées sur le TEST (oiseaux jamais vus) ===")
print(res.to_string())
