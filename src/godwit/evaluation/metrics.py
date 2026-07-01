"""Métriques d'erreur de trajectoire, en km, bâties sur la distance de Haversine.
ADE = erreur moyenne sur tous les pas prédits ; FDE = erreur sur le dernier pas."""

import numpy as np
from godwit.geo import haversine_km


def displacement_errors(pred_lat, pred_lon, true_lat, true_lon):
    """Erreur de position (km) à chaque pas prédit : Haversine(pred, vérité)."""
    return haversine_km(np.asarray(pred_lat), np.asarray(pred_lon),
                        np.asarray(true_lat), np.asarray(true_lon))


def ade(pred_lat, pred_lon, true_lat, true_lon):
    """Average Displacement Error : moyenne des erreurs de position (km)."""
    return float(np.mean(displacement_errors(pred_lat, pred_lon, true_lat, true_lon)))


def fde(pred_lat, pred_lon, true_lat, true_lon):
    """Final Displacement Error : erreur (km) sur le dernier pas de l'horizon.
    Pour de la prédiction one-step, FDE == erreur du pas unique."""
    err = displacement_errors(pred_lat, pred_lon, true_lat, true_lon)
    return float(err[-1])
