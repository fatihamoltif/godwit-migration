import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import torch
from godwit.data.windowing_wind import make_windows_wind
from godwit.data.split import train_test_split_by_bird
from godwit.models.lstm import GodwitLSTM
from godwit.models.train import train_model
from godwit.evaluation.metrics import displacement_errors

# --- Données construites une seule fois (fenêtres + vent + cible résiduelle) ---
tracks = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                     parse_dates=["Location_Timestamp_UTC"])
wind = pd.read_csv("data/processed/flight_features.csv")
X, resid, y_raw, gc, meta = make_windows_wind(tracks, wind)

def encode(X):
    lat, lon360 = X[:, :, 0], X[:, :, 1] % 360
    hrad, speed = np.radians(X[:, :, 2]), X[:, :, 3]
    geo = np.stack([lat, lon360, np.sin(hrad), np.cos(hrad), speed], axis=-1)
    return np.concatenate([geo, X[:, :, 4:8]], axis=-1)   # (n,4,9) : 5 géo + 4 vent
Xf = encode(X)

def run_once(use_wind, split_seed):
    """Entraîne+évalue le LSTM résiduel, avec ou sans les features vent."""
    feat_cols = slice(0, 9) if use_wind else slice(0, 5)   # 9 = géo+vent, 5 = géo seul
    Xu = Xf[:, :, feat_cols]

    tr_mask, te_mask, _ = train_test_split_by_bird(meta, 0.25, random_state=split_seed)
    meta_tr = meta[tr_mask].reset_index(drop=True)
    tr2, val2, _ = train_test_split_by_bird(meta_tr, 0.20, random_state=7)

    Xtr_full, rtr_full = Xu[tr_mask], resid[tr_mask]
    X_tr, r_tr   = Xtr_full[tr2],  rtr_full[tr2]
    X_val, r_val = Xtr_full[val2], rtr_full[val2]
    X_te, r_te   = Xu[te_mask], resid[te_mask]
    gc_te, meta_te = gc[te_mask], meta[te_mask].reset_index(drop=True)

    # standardisation sur le train
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    xm, xs = flat.mean(0), flat.std(0) + 1e-8
    rm, rs = r_tr.mean(0), r_tr.std(0) + 1e-8
    t = lambda A: torch.tensor(A, dtype=torch.float32)
    sx = lambda A: t((A - xm) / xs); sr = lambda A: t((A - rm) / rs)

    torch.manual_seed(0)   # MÊME init pour les deux conditions
    model = GodwitLSTM(n_features=X_tr.shape[-1], hidden_size=32)
    model, _ = train_model(model, sx(X_tr), sr(r_tr), sx(X_val), sr(r_val),
                           epochs=300, batch_size=32, lr=1e-3, patience=30, verbose=False)
    model.eval()
    with torch.no_grad():
        rp = model(sx(X_te)).numpy() * rs + rm
    dlat, dlon = gc_te[:,0] + rp[:,0], gc_te[:,1] + rp[:,1]
    err = displacement_errors(meta_te["last_lat"].values + dlat,
                              meta_te["last_lon"].values + dlon,
                              meta_te["true_lat"].values, meta_te["true_lon"].values)
    return err.mean()

SEEDS = [42, 7, 123, 2024, 99]
print(f"{'split':>6} | {'sans vent':>10} | {'avec vent':>10} | {'gain km':>8}")
sans_all, avec_all = [], []
for s in SEEDS:
    sans = run_once(False, s)
    avec = run_once(True, s)
    sans_all.append(sans); avec_all.append(avec)
    print(f"{s:>6} | {sans:>10.1f} | {avec:>10.1f} | {sans-avec:>+8.1f}")

sans_all, avec_all = np.array(sans_all), np.array(avec_all)
print(f"\n{'MOYENNE':>6} | {sans_all.mean():>10.1f} | {avec_all.mean():>10.1f} | "
      f"{(sans_all-avec_all).mean():>+8.1f}")
print(f"\nLe vent aide sur {int((avec_all < sans_all).sum())}/{len(SEEDS)} splits.")
print("Gain positif = 'avec vent' fait MOINS d'erreur (le vent aide la prédiction).")
