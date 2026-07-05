import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Résultats consolidés (test, 8 oiseaux jamais vus) — ceux qu'on a mesurés
models = ["Persistance", "Vitesse\nlissée", "Grand\ncercle", "LSTM géo", "LSTM+vent\n(résiduel)"]
ade    = [32.3,          47.7,             31.3,           51.5,        44.1]
median = [6.1,           29.3,             3.7,            37.5,        26.6]

x = np.arange(len(models))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))

b1 = ax.bar(x - w/2, ade,    w, label="ADE (erreur moyenne)", color="#4C72B0")
b2 = ax.bar(x + w/2, median, w, label="Erreur médiane",       color="#8FB0DD")

# ligne repère : la meilleure baseline (grand cercle)
ax.axhline(31.3, color="crimson", ls="--", lw=1.3, alpha=0.8)
ax.text(len(models)-0.5, 32.5, "barre à battre : grand cercle (31 km)",
        color="crimson", ha="right", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Erreur de position (km)")
ax.set_title("Prédiction du pas suivant (6 h) — évaluation sur oiseaux jamais vus\n"
             "La baseline géométrique reste la plus performante",
             fontsize=12, weight="bold")
ax.legend()
for b in list(b1) + list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.6,
            f"{b.get_height():.0f}", ha="center", fontsize=8)

fig.tight_layout()
out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
f = out / "results_comparison.png"
fig.savefig(f, dpi=130, bbox_inches="tight")
print(f"-> {f}")
