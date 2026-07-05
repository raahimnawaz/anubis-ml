"""
Generate the figures embedded in the README, from data/segments.parquet.

  1. acceptance_turnon.png -- P(segment) vs Delta-R to proANUBIS: the detector's
     geometric turn-on curve.
  2. acceptance_map.png    -- P(segment) across the (eta, phi) plane: a 2D map of the
     proANUBIS field of view, with the nominal detector direction marked.

Pure matplotlib (Agg backend), no ROOT. Writes to docs/figures/.
Usage:
    python src/make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import PANUBIS_ETA, PANUBIS_PHI

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/segments.parquet")

# --- 1. turn-on curve: P(segment) vs Delta-R --------------------------------------
edges = np.linspace(0, 3.0, 31)
mid = 0.5 * (edges[:-1] + edges[1:])
idx = np.digitize(df["dR_proanub"], edges) - 1
prob = [df["has_segment"].values[idx == b].mean() if (idx == b).any() else np.nan
        for b in range(len(mid))]

plt.figure(figsize=(7, 4.2))
plt.plot(mid, prob, "o-", color="#1f77b4")
plt.axvline(0.3, ls="--", color="grey", label="acceptance cut (dR=0.3)")
plt.xlabel(r"$\Delta R$ from muon to proANUBIS direction")
plt.ylabel("P(reconstructed segment)")
plt.title("proANUBIS acceptance turn-on")
plt.ylim(-0.03, 1.03)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "acceptance_turnon.png", dpi=130)
plt.close()

# --- 2. 2D acceptance map in (eta, phi) -------------------------------------------
eta_edges = np.linspace(-1.3, 1.3, 40)
phi_edges = np.linspace(-np.pi, np.pi, 48)
num, _, _ = np.histogram2d(df["eta"], df["phi"], bins=[eta_edges, phi_edges],
                           weights=df["has_segment"].astype(float))
den, _, _ = np.histogram2d(df["eta"], df["phi"], bins=[eta_edges, phi_edges])
with np.errstate(invalid="ignore", divide="ignore"):
    p = np.where(den > 20, num / den, np.nan)

plt.figure(figsize=(7, 4.6))
im = plt.pcolormesh(eta_edges, phi_edges, p.T, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, label="P(reconstructed segment)")
plt.scatter([PANUBIS_ETA], [PANUBIS_PHI], marker="*", s=260,
            color="red", edgecolor="white", zorder=5, label="proANUBIS direction")
plt.xlabel(r"muon $\eta$")
plt.ylabel(r"muon $\phi$")
plt.title("proANUBIS 'field of view': segment probability vs muon direction")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(OUT / "acceptance_map.png", dpi=130)
plt.close()

print("wrote", OUT / "acceptance_turnon.png")
print("wrote", OUT / "acceptance_map.png")
