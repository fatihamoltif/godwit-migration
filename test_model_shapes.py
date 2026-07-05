import sys; sys.path.insert(0, "src")
import torch
from godwit.models.lstm import GodwitLSTM

bundle = torch.load("data/processed/lstm_tensors.pt", weights_only=False)
X_tr = bundle["X_tr"]

model = GodwitLSTM(n_features=X_tr.shape[-1], hidden_size=32)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modèle créé — {n_params} paramètres au total")

# Passe avant sur un mini-lot de 8 exemples
batch = X_tr[:8]
out = model(batch)
print(f"Entrée  : {tuple(batch.shape)}  (batch, temps, features)")
print(f"Sortie  : {tuple(out.shape)}   (batch, 2=[dlat,dlon] standardisés)")
print("Plomberie OK" if out.shape == (8, 2) else "PROBLÈME de forme")
