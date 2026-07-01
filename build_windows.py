import sys; sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from godwit.data.windowing import flight_segments, make_windows

df = pd.read_csv("data/interim/resampled_tracks_6h.csv",
                 parse_dates=["Location_Timestamp_UTC"])

segs = flight_segments(df)
X, y, meta = make_windows(segs)

print(f"Segments de vol            : {len(segs)}")
print(f"Exemples (fenêtres)        : {len(X)}")
print(f"Forme de X (n, window, feat): {X.shape}")
print(f"Forme de y (n, [dlat,dlon]) : {y.shape}")

print("\n--- Contrôle de cohérence sur le 1er exemple ---")
print("dernière position connue :", round(meta['last_lat'][0],3), round(meta['last_lon'][0],3))
print("déplacement cible (dlat,dlon) :", np.round(y[0], 3))
print("=> position reconstruite :",
      round(meta['last_lat'][0]+y[0][0],3), round(meta['last_lon'][0]+y[0][1],3))
print("   position vraie          :", round(meta['true_lat'][0],3), round(meta['true_lon'][0],3))
print("   (les deux doivent être identiques)")

print("\n--- Ampleur des déplacements cibles ---")
dist = np.sqrt(y[:,0]**2 + y[:,1]**2)
print("déplacement médian (°) :", round(np.median(dist),3))
print("déplacement max (°)    :", round(dist.max(),3))
