import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import torch
from godwit.data.windowing import flight_segments, make_windows
from godwit.data.split import train_test_split_by_bird

# 1) Fenêtres (features géométriques brutes : Lat, Lon, cap, vitesse, step)
df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])
segs = flight_segments(df)
X, y, meta = make_windows(segs)          # X:(n,4,5)  y:(n,2)=[dlat,dlon]

# 2) Encodage "réseau" : lon en 0-360, cap -> (sin,cos), on abandonne step_km
#    (step_km = vitesse*6h, donc redondant avec la vitesse)
def encode(X):
    lat    = X[:, :, 0]
    lon360 = X[:, :, 1] % 360                    # continuité antiméridien
    hrad   = np.radians(X[:, :, 2])
    hsin, hcos = np.sin(hrad), np.cos(hrad)      # cap cyclique -> sin/cos
    speed  = X[:, :, 3]
    return np.stack([lat, lon360, hsin, hcos, speed], axis=-1)  # (n,4,5)

Xf = encode(X)

# 3) Split 3 voies PAR OISEAU : test 25%, puis val = 20% des oiseaux de train
train_mask, test_mask, test_birds = train_test_split_by_bird(meta, 0.25, random_state=42)
meta_tr_full = meta[train_mask].reset_index(drop=True)
tr2, val2, val_birds = train_test_split_by_bird(meta_tr_full, 0.20, random_state=7)

X_train_full, y_train_full = Xf[train_mask], y[train_mask]
X_tr,  y_tr  = X_train_full[tr2],  y_train_full[tr2]
X_val, y_val = X_train_full[val2], y_train_full[val2]
X_te,  y_te  = Xf[test_mask], y[test_mask]
meta_te = meta[test_mask].reset_index(drop=True)

# 4) Standardisation : stats calculées SUR LE TRAIN UNIQUEMENT
flat = X_tr.reshape(-1, X_tr.shape[-1])
x_mean, x_std = flat.mean(0), flat.std(0) + 1e-8
sx = lambda A: (A - x_mean) / x_std
y_mean, y_std = y_tr.mean(0), y_tr.std(0) + 1e-8
sy = lambda A: (A - y_mean) / y_std

t = lambda A: torch.tensor(A, dtype=torch.float32)
bundle = {
    "X_tr": t(sx(X_tr)), "y_tr": t(sy(y_tr)),
    "X_val": t(sx(X_val)), "y_val": t(sy(y_val)),
    "X_te": t(sx(X_te)),  "y_te": t(sy(y_te)),
    "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
    "meta_te": {k: meta_te[k].values for k in
                ["last_lat", "last_lon", "true_lat", "true_lon"]},
}
torch.save(bundle, "data/processed/lstm_tensors.pt")

# --- Vérifications ---
print(f"Oiseaux  train/val/test : {25-len(val_birds)} / {len(val_birds)} / {len(test_birds)}")
print(f"Exemples train/val/test : {len(X_tr)} / {len(X_val)} / {len(X_te)}")
print(f"Forme X_tr : {tuple(bundle['X_tr'].shape)}   Forme y_tr : {tuple(bundle['y_tr'].shape)}")
print(f"\nContrôle standardisation (train) : moyenne={bundle['X_tr'].mean():.3f} (~0)  "
      f"écart-type={bundle['X_tr'].std():.3f} (~1)")
print(f"Contrôle test (NE doit PAS être exactement 0/1) : "
      f"moyenne={bundle['X_te'].mean():.3f}  écart-type={bundle['X_te'].std():.3f}")
print(f"NaN ? X_tr={int(torch.isnan(bundle['X_tr']).sum())}  "
      f"y_tr={int(torch.isnan(bundle['y_tr']).sum())}")
print("\n-> data/processed/lstm_tensors.pt")
