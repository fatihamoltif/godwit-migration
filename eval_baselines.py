import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.data.windowing import flight_segments, make_windows
from godwit.models.baselines import (predict_persistence,
                                      predict_mean_velocity, predict_great_circle)
from godwit.evaluation.metrics import displacement_errors

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)
X, y, meta = make_windows(segs)

def evaluate(name, y_pred):
    pred_lat = meta["last_lat"].values + y_pred[:, 0]
    pred_lon = meta["last_lon"].values + y_pred[:, 1]
    err = displacement_errors(pred_lat, pred_lon,
                              meta["true_lat"].values, meta["true_lon"].values)
    return {"baseline": name, "ADE_km": err.mean(),
            "médiane_km": np.median(err), "P90_km": np.percentile(err, 90)}

rows = [
    evaluate("persistance",        predict_persistence(X)),
    evaluate("vitesse lissée (3)", predict_mean_velocity(X)),
    evaluate("grand cercle",       predict_great_circle(X)),
]
res = pd.DataFrame(rows).set_index("baseline").round(1)
print(res.to_string())
print(f"\n(distance moyenne parcourue par pas : ~290 km — pour situer les erreurs)")
