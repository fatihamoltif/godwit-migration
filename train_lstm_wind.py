import sys; sys.path.insert(0, "src")
import torch
from godwit.models.lstm import GodwitLSTM
from godwit.models.train import train_model

torch.manual_seed(0)
bundle = torch.load("data/processed/lstm_wind_tensors.pt", weights_only=False)

model = GodwitLSTM(n_features=bundle["X_tr"].shape[-1], hidden_size=32)
model, history = train_model(model, bundle["X_tr"], bundle["y_tr"],
                             bundle["X_val"], bundle["y_val"],
                             epochs=300, batch_size=32, lr=1e-3, patience=30)
torch.save({"state_dict": model.state_dict()}, "data/processed/lstm_wind_trained.pt")
val = [v for _, v in history]
print(f"\nMeilleure perte val : {min(val):.4f}")
