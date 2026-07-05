import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import torch
from godwit.models.lstm import GodwitLSTM
from godwit.evaluation.metrics import displacement_errors

bundle = torch.load("data/processed/lstm_tensors.pt", weights_only=False)
ckpt   = torch.load("data/processed/lstm_trained.pt", weights_only=False)

# Recharger le modèle entraîné
model = GodwitLSTM(n_features=bundle["X_tr"].shape[-1], hidden_size=32)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 1) Prédiction sur le test (en unités standardisées)
with torch.no_grad():
    y_pred_std = model(bundle["X_te"]).numpy()

# 2) Dé-standardiser : revenir aux vrais (dlat, dlon) en degrés
y_pred = y_pred_std * bundle["y_std"] + bundle["y_mean"]

# 3) Reconstruire la position prédite = dernière position connue + déplacement
m = bundle["meta_te"]
pred_lat = m["last_lat"] + y_pred[:, 0]
pred_lon = m["last_lon"] + y_pred[:, 1]

# 4) Erreur en km via Haversine (même métrique que les baselines)
err = displacement_errors(pred_lat, pred_lon, m["true_lat"], m["true_lon"])

print("=== LSTM — évaluation sur le TEST (8 oiseaux jamais vus) ===")
print(f"Exemples      : {len(err)}")
print(f"ADE (moy)     : {err.mean():.1f} km")
print(f"médiane       : {np.median(err):.1f} km")
print(f"P90           : {np.percentile(err, 90):.1f} km")

print("\n=== COMPARAISON avec la barre de référence ===")
print(f"{'modèle':<22}{'ADE_km':>8}{'médiane':>9}{'P90':>8}")
print(f"{'grand cercle (baseline)':<22}{31.3:>8}{3.7:>9}{103.3:>8}")
print(f"{'LSTM':<22}{err.mean():>8.1f}{np.median(err):>9.1f}{np.percentile(err,90):>8.1f}")
gain = 100 * (31.3 - err.mean()) / 31.3
print(f"\nGain d'ADE vs grand cercle : {gain:+.0f} %")
