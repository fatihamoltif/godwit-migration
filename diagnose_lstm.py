import sys; sys.path.insert(0, "src")
import numpy as np
import torch
from godwit.models.lstm import GodwitLSTM

bundle = torch.load("data/processed/lstm_tensors.pt", weights_only=False)
ckpt   = torch.load("data/processed/lstm_trained.pt", weights_only=False)
model = GodwitLSTM(n_features=bundle["X_tr"].shape[-1], hidden_size=32)
model.load_state_dict(ckpt["state_dict"]); model.eval()

with torch.no_grad():
    pred_std = model(bundle["X_te"]).numpy()
pred = pred_std * bundle["y_std"] + bundle["y_mean"]     # dé-standardisé (degrés)
true = bundle["y_te"].numpy() * bundle["y_std"] + bundle["y_mean"]

print("Comparaison prédiction vs vérité sur les déplacements (degrés) :")
print(f"{'':12}{'dlat_moy':>10}{'dlat_std':>10}{'dlon_moy':>10}{'dlon_std':>10}")
print(f"{'VÉRITÉ':12}{true[:,0].mean():>10.3f}{true[:,0].std():>10.3f}"
      f"{true[:,1].mean():>10.3f}{true[:,1].std():>10.3f}")
print(f"{'PRÉDICTION':12}{pred[:,0].mean():>10.3f}{pred[:,0].std():>10.3f}"
      f"{pred[:,1].mean():>10.3f}{pred[:,1].std():>10.3f}")

r_lat = np.corrcoef(pred[:,0], true[:,0])[0,1]
r_lon = np.corrcoef(pred[:,1], true[:,1])[0,1]
print(f"\nCorrélation pred/vérité : dlat {r_lat:+.2f}   dlon {r_lon:+.2f}")
print("(si les std de prédiction << std de vérité, le modèle 'joue la moyenne')")
