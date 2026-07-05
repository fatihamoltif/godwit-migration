import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import torch
from godwit.data.windowing_wind import make_windows_wind
from godwit.data.split import train_test_split_by_bird

tracks = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                     parse_dates=["Location_Timestamp_UTC"])
wind = pd.read_csv("data/processed/flight_features.csv")

X, resid, y_raw, gc, meta = make_windows_wind(tracks, wind)
print(f"Exemples : {len(X)}   features par pas : {X.shape[-1]} (4 géo + 4 vent)")

# Encodage réseau des features géométriques (comme avant) + vent tel quel
def encode(X):
    lat    = X[:, :, 0]
    lon360 = X[:, :, 1] % 360
    hrad   = np.radians(X[:, :, 2])
    speed  = X[:, :, 3]
    wind   = X[:, :, 4:8]                      # tailwind/crosswind 925 & 850
    geo = np.stack([lat, lon360, np.sin(hrad), np.cos(hrad), speed], axis=-1)
    return np.concatenate([geo, wind], axis=-1)   # (n, 4, 9)

Xf = encode(X)

# Split 3 voies PAR OISEAU, identique au précédent (mêmes graines => même test)
train_mask, test_mask, test_birds = train_test_split_by_bird(meta, 0.25, random_state=42)
meta_tr_full = meta[train_mask].reset_index(drop=True)
tr2, val2, _ = train_test_split_by_bird(meta_tr_full, 0.20, random_state=7)

Xtr_full, rtr_full = Xf[train_mask], resid[train_mask]
X_tr,  r_tr  = Xtr_full[tr2],  rtr_full[tr2]
X_val, r_val = Xtr_full[val2], rtr_full[val2]
X_te,  r_te  = Xf[test_mask],  resid[test_mask]

# Standardisation des FEATURES sur le train
flat = X_tr.reshape(-1, X_tr.shape[-1])
x_mean, x_std = flat.mean(0), flat.std(0) + 1e-8
sx = lambda A: (A - x_mean) / x_std
# Standardisation de la CIBLE (le résidu) sur le train
r_mean, r_std = r_tr.mean(0), r_tr.std(0) + 1e-8
sr = lambda A: (A - r_mean) / r_std

t = lambda A: torch.tensor(A, dtype=torch.float32)
bundle = {
    "X_tr": t(sx(X_tr)), "y_tr": t(sr(r_tr)),
    "X_val": t(sx(X_val)), "y_val": t(sr(r_val)),
    "X_te": t(sx(X_te)),  "y_te": t(sr(r_te)),
    "r_mean": r_mean, "r_std": r_std,
    # pour l'éval : le grand cercle et la reconstruction sur le test
    "gc_te": gc[test_mask],
    "meta_te": {k: meta[test_mask].reset_index(drop=True)[k].values
                for k in ["last_lat", "last_lon", "true_lat", "true_lon"]},
}
torch.save(bundle, "data/processed/lstm_wind_tensors.pt")
print(f"Forme X_tr : {tuple(bundle['X_tr'].shape)}  (n, 4, 9)")
print(f"Exemples train/val/test : {len(X_tr)}/{len(X_val)}/{len(X_te)}")
print("-> data/processed/lstm_wind_tensors.pt")
