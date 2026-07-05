"""Boucle d'entraînement du LSTM avec suivi de validation et early stopping."""

import copy
import torch
import torch.nn as nn


def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=300, batch_size=32, lr=1e-3, patience=30, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(X_tr)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_best = 0
    history = []

    for epoch in range(epochs):
        # --- entraînement : on parcourt le train en lots mélangés ---
        model.train()
        perm = torch.randperm(n)
        train_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]

            opt.zero_grad()            # remettre les gradients à zéro
            pred = model(xb)           # 1. forward
            loss = loss_fn(pred, yb)   # 2. loss
            loss.backward()            # 3. backward (calcul des gradients)
            opt.step()                 # 4. step (mise à jour des poids)
            train_loss += loss.item() * len(idx)
        train_loss /= n

        # --- validation : pas de gradient, juste la mesure ---
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        history.append((train_loss, val_loss))

        # --- early stopping : garder le meilleur modèle sur la val ---
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  époque {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}"
                  f"{'  <- meilleur' if epochs_since_best == 0 else ''}")

        if epochs_since_best >= patience:
            if verbose:
                print(f"  arrêt anticipé à l'époque {epoch} "
                      f"(pas d'amélioration depuis {patience} époques)")
            break

    model.load_state_dict(best_state)   # restaurer le meilleur état
    return model, history
