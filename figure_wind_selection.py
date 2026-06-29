import sys; sys.path.insert(0, "src")
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feat = pd.read_csv("data/processed/flight_features.csv")
tw = feat["tailwind_850"]
gs = feat["ground_speed_ms"]
pct = 100 * (tw > 0).mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- (1) Distribution du tailwind ---
ax1.hist(tw, bins=40, color="#4C9", edgecolor="white")
ax1.axvline(0, color="black", lw=1.2, ls="--")
ax1.axvline(tw.mean(), color="crimson", lw=2,
            label=f"moyenne = {tw.mean():+.1f} m/s")
ax1.set_title(f"Vent le long du cap (850 hPa)\n{pct:.0f}% des vols avec vent porteur")
ax1.set_xlabel("Tailwind (m/s)   —   négatif = vent de face")
ax1.set_ylabel("Nombre de pas de vol")
ax1.legend()

# --- (2) Tailwind vs vitesse-sol ---
ax2.scatter(tw, gs * 3.6, s=10, alpha=0.3, color="#268")
# droite de tendance
b, a = np.polyfit(tw, gs * 3.6, 1)
xs = np.linspace(tw.min(), tw.max(), 100)
ax2.plot(xs, a + b * xs, color="crimson", lw=2,
         label=f"tendance (corr {tw.corr(gs):+.2f})")
ax2.set_title("Plus le vent pousse, plus le sol défile vite")
ax2.set_xlabel("Tailwind (m/s)")
ax2.set_ylabel("Vitesse-sol (km/h)")
ax2.legend()

fig.suptitle("Sélection du vent par la barge rousse — 1412 pas de vol, 33 oiseaux (2005-2010)",
             fontsize=13, weight="bold")
fig.tight_layout()

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
f = out / "wind_selection.png"
fig.savefig(f, dpi=130, bbox_inches="tight")
print(f"Figure enregistrée : {f}")
