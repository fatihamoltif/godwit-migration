import numpy as np

def initial_bearing(lat1, lon1, lat2, lon2):
    """Cap initial (degrés, 0=Nord, sens horaire) du point 1 vers le point 2."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    theta = np.arctan2(x, y)                 # atan2(est, nord)
    return (np.degrees(theta) + 360) % 360   # ramène dans [0, 360)

# --- Tests sur des cas évidents pour valider la formule ---
print("Vers le nord  (attendu   0) :", round(initial_bearing(0, 0,  1, 0), 1))
print("Vers l'est    (attendu  90) :", round(initial_bearing(0, 0,  0, 1), 1))
print("Vers le sud   (attendu 180) :", round(initial_bearing(0, 0, -1, 0), 1))
print("Vers l'ouest  (attendu 270) :", round(initial_bearing(0, 0,  0, -1), 1))

# --- Cas réel : un pas du vol trans-Pacifique de BARG_Z4 (vu au jalon 2) ---
# P1=(39.349,147.830) -> P2=(48.230,171.529)
print("\nPas de vol réel (NW Pacifique) :",
      round(initial_bearing(39.349, 147.830, 48.230, 171.529), 1), "°")
