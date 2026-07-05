import sys; sys.path.insert(0, "src")
import numpy as np
import torch
from godwit.models.lstm import GodwitLSTM
from godwit.evaluation.metrics import displacement_errors

bundle = torch.load("data/processed/lstm_wind_tensors.pt", weights_only=False)
ckpt   = torch.load("data/processed/lstm_wind_trained.pt", weights_only=False)
model = GodwitLSTM(n_features=bundle["X_tr"].shape[-1], hidden_size=32)
model.load_state_dict(ckpt["state_dict"]); model.eval()

with torch.no_grad():
    resid_pred = model(bundle["X_te"]).numpy() * bundle["r_std"] + bundle["r_mean"]

# Reconstruction : déplacement = grand cercle + résidu prédit
gc = bundle["gc_te"]
dlat = gc[:, 0] + resid_pred[:, 0]
dlon = gc[:, 1] + resid_pred[:, 1]

m = bundle["meta_te"]
pred_lat = m["last_lat"] + dlat
pred_lon = m["last_lon"] + dlon
err = displacement_errors(pred_lat, pred_lon, m["true_lat"], m["true_lon"])

print("=== LSTM + VENT (résiduel) — TEST (8 oiseaux jamais vus) ===")
print(f"ADE {err.mean():.1f} km | médiane {np.median(err):.1f} km | P90 {np.percentile(err,90):.1f} km")
print("\n=== MATCH FINAL ===")
print(f"{'modèle':<26}{'ADE':>7}{'méd.':>7}{'P90':>7}")
print(f"{'grand cercle (baseline)':<26}{31.3:>7}{3.7:>7}{103.3:>7}")
print(f"{'LSTM géo seul (avant)':<26}{51.5:>7}{37.5:>7}{112.1:>7}")
print(f"{'LSTM + vent (résiduel)':<26}{err.mean():>7.1f}{np.median(err):>7.1f}{np.percentile(err,90):>7.1f}")
print(f"\nGain vs grand cercle : {100*(31.3-err.mean())/31.3:+.0f}%")
