from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
scores = pd.read_csv(OUT / "GSE301781_direction_scores.csv")
genes = pd.read_csv(OUT / "GSE301781_candidate_gene_validation.csv")

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), dpi=300)
groups = ["Control", "CLP_6w"]
colors = {"Control": "#8A9BA2", "CLP_6w": "#A74A43"}
for i, group in enumerate(groups):
    values = scores.loc[scores["group"] == group, "score"].to_numpy()
    axes[0].scatter(
        np.full(len(values), i) + np.linspace(-0.07, 0.07, len(values)),
        values,
        color=colors[group],
        s=36,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    axes[0].hlines(
        values.mean(), i - 0.2, i + 0.2, color="#222222", linewidth=1.4
    )
axes[0].set_xticks([0, 1], ["Control", "CLP 6 weeks"])
axes[0].set_ylabel("Six-gene direction score")
axes[0].axhline(0, color="#888888", linestyle=":", linewidth=0.7)
axes[0].set_title("Cardiac CD45+ cells", fontsize=9)
axes[0].spines[["top", "right"]].set_visible(False)

plot = genes.sort_values("effect")
y = np.arange(len(plot))
bar_colors = np.where(plot["concordant"], "#4F6D7A", "#B07A45")
axes[1].barh(y, plot["effect"], color=bar_colors, height=0.65)
axes[1].set_yticks(y, plot["gene"], fontsize=7)
axes[1].axvline(0, color="#333333", linewidth=0.8)
axes[1].set_xlabel("CLP 6 weeks - Control")
axes[1].set_title("Candidate-gene effects", fontsize=9)
axes[1].spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "supplementary_figure_gse301781.png", bbox_inches="tight")
plt.close(fig)
