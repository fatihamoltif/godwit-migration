import sys; sys.path.insert(0, "src")
import numpy as np
import torch
from godwit.evaluation.metrics import displacement_errors
from godwit.models.lstm import GodwitLSTM

bundle = torch.load("data/processed/lstm_tensors.pt", weights_only=False)
ckpt   = torch.load("data/processed/lstm_trained.pt", weights_only=False)
model = GodwitLSTM(n_features=bundle["X_tr"].shape[-1], hidden_size=32)
model.load_state_dict(ckpt["state_dict"]); model.eval()

with torch.no_grad():
    pred = model(bundle["X_te"]).numpy() * bundle["y_std"] + bundle["y_mean"]
true = bundle["y_te"].numpy() * bundle["y_std"] + bundle["y_mean"]
m = bundle["meta_te"]

def ade_of(dlat, dlon):
    return displacement_errors(m["last_lat"]+dlat, m["last_lon"]+dlon,
                               m["true_lat"], m["true_lon"]).mean()

ade_brut = ade_of(pred[:,0], pred[:,1])
# retirer le biais moyen mesuré sur le test (oracle : juste pour diagnostiquer)
bias = pred.mean(0) - true.mean(0)
ade_debias = ade_of(pred[:,0]-bias[0], pred[:,1]-bias[1])

print(f"ADE brute du LSTM            : {ade_brut:.1f} km")
print(f"ADE si on retire le biais moy: {ade_debias:.1f} km")
print(f"grand cercle (référence)     : 31.3 km")
print(f"\n-> {100*(ade_brut-ade_debias)/ade_brut:.0f}% de l'erreur vient du biais systématique")
