#!/usr/bin/env python3
"""Render figures/results.png from results/{rolling_origin,lead_time}.csv."""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
roll = pd.read_csv(REPO / "results/rolling_origin.csv")
lead = pd.read_csv(REPO / "results/lead_time.csv")

C = {"hgb": "#2a78d6", "lgbm": "#eb6834", "rf": "#1baf7a", "logreg": "#eda100"}
NAMES = {"hgb": "HistGradientBoosting", "lgbm": "LightGBM", "rf": "Random Forest", "logreg": "Logistic Regression"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160, facecolor="white")

# --- left: rolling-origin test ROC-AUC by held-out year ---
years = [2022, 2023, 2024]; models = ["hgb", "lgbm", "rf", "logreg"]
w = 0.19
for i, m in enumerate(models):
    d = roll[roll.model == m].set_index("test_year").loc[years, "roc_auc"]
    xs = [y + (i - 1.5) * w for y in years]
    ax1.bar(xs, d.values, width=w - 0.02, color=C[m], label=NAMES[m], zorder=3)
    for x, v in zip(xs, d.values):
        ax1.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
ax1.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=2)
ax1.set_xticks(years); ax1.set_xticklabels([f"test {y}" for y in years])
ax1.set_ylim(0.45, 0.9); ax1.set_ylabel("Test ROC-AUC (same-day occurrence)")
ax1.set_title("Rolling-origin evaluation: train on prior years", loc="left", fontsize=11, color=INK)
ax1.grid(axis="y", color=GRID, zorder=0); ax1.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

# --- right: lead-time curve (new ignitions) ---
ni = lead[lead.population == "new_ignition"]
mean = ni.groupby("horizon_d")[["roc_auc", "pr_lift"]].mean()
ax2.plot(mean.index, mean.roc_auc, color=C["hgb"], lw=2, marker="o", ms=6, zorder=3, label="ROC-AUC, mean over test years")
for ty, g in ni.groupby("test_year"):
    ax2.scatter(g.horizon_d, g.roc_auc, color=C["hgb"], alpha=0.35, s=18, zorder=2)
for h, r in mean.iterrows():
    off, ha, va = ((12, 0), "left", "center") if h == 14 else ((0, -12), "center", "top")
    ax2.annotate(f"AUC {r.roc_auc:.2f}\nlift {r.pr_lift:.1f}×", (h, r.roc_auc), textcoords="offset points",
                 xytext=off, ha=ha, va=va, fontsize=8, color=INK)
ax2.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
ax2.set_xticks([14, 28, 42]); ax2.set_xticklabels(["14 d", "28 d", "42 d"])
ax2.set_xlim(8, 48); ax2.set_ylim(0.45, 0.9)
ax2.set_xlabel("Forecast horizon"); ax2.set_ylabel("Test ROC-AUC (new ignition in cell)")
ax2.set_title("Lead-time skill for new ignitions (HGB)", loc="left", fontsize=11, color=INK)
ax2.grid(axis="y", color=GRID, zorder=0)
ax2.text(0.02, 0.03, "faint dots: individual test years (2022–2024)\nlift = PR-AUC ÷ base rate (~0.4%)",
         transform=ax2.transAxes, fontsize=7.5, color=MUTED, va="bottom")

fig.tight_layout()
out = REPO / "figures/results.png"; out.parent.mkdir(exist_ok=True)
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
