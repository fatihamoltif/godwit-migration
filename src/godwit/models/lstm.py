"""Petit LSTM encodeur + tête de régression pour prédire le déplacement (dlat, dlon)."""

import torch
import torch.nn as nn


class GodwitLSTM(nn.Module):
    def __init__(self, n_features=5, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        # Encodeur : lit la séquence (batch, seq_len, n_features)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # format (batch, temps, features)
            dropout=dropout,
        )
        # Tête : du résumé (hidden_size) vers 2 sorties (dlat, dlon)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        # out: sorties à chaque pas ; (h_n, c_n): états finaux
        out, (h_n, c_n) = self.lstm(x)
        summary = h_n[-1]              # (batch, hidden_size) = mémoire après le dernier pas
        return self.head(summary)      # (batch, 2)
