from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
RESULT_DIR = ROOT / "results"

HUB_GENES = {
    "RELA": "ENSG00000173039",
    "ATP6V1B2": "ENSG00000147416",
    "HMOX1": "ENSG00000100292",
    "MITF": "ENSG00000187098",
    "ATP6V1D": "ENSG00000100554",
    "TFE3": "ENSG00000068323",
}

DIRECTIONS = {
    "RELA": 1,
    "ATP6V1B2": 1,
    "HMOX1": 1,
    "MITF": -1,
    "ATP6V1D": -1,
    "TFE3": 1,
}


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), len(labels))
        if len(set(labels[indices])) < 2:
            continue
        values.append(roc_auc_score(labels[indices], scores[indices]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(
        DATA_DIR / "GSE294985_Processed_data_FPKM.xlsx"
    ).set_index("gene_id")
    expression = raw.loc[list(HUB_GENES.values())].copy()
    expression.index = list(HUB_GENES)
    controls = [column for column in expression.columns if column.startswith("ctrl")]
    cases = [column for column in expression.columns if column.endswith("-Seq")]

    case_values = expression[cases].to_numpy(float)
    control_values = expression[controls].to_numpy(float)
    effect = case_values.mean(axis=1) - control_values.mean(axis=1)
    p_values = stats.ttest_ind(
        case_values,
        control_values,
        axis=1,
        equal_var=False,
    ).pvalue
    fdr = multipletests(p_values, method="fdr_bh")[1]
    result = pd.DataFrame(
        {
            "gene": expression.index,
            "effect": effect,
            "p_value": p_values,
            "fdr": fdr,
            "expected_direction": [
                "up" if DIRECTIONS[gene] > 0 else "down"
                for gene in expression.index
            ],
            "observed_direction": [
                "up" if value > 0 else "down" for value in effect
            ],
            "direction_concordant": [
                (value > 0) == (DIRECTIONS[gene] > 0)
                for value, gene in zip(effect, expression.index)
            ],
        }
    )
    result.to_csv(
        TABLE_DIR / "GSE294985_hub_gene_validation.csv",
        index=False,
    )

    z = expression.sub(expression.mean(axis=1), axis=0).div(
        expression.std(axis=1, ddof=1).replace(0, np.nan),
        axis=0,
    )
    directions = np.array([DIRECTIONS[gene] for gene in expression.index])
    score = np.nanmean(z.to_numpy().T * directions, axis=1)
    score_series = pd.Series(score, index=expression.columns, name="score")
    labels = np.array([1] * len(cases) + [0] * len(controls))
    ordered_score = score_series[cases + controls].to_numpy()
    auc = roc_auc_score(labels, ordered_score)
    ci_low, ci_high = bootstrap_auc(labels, ordered_score)
    score_table = pd.DataFrame(
        {
            "sample": cases + controls,
            "group": ["Sepsis"] * len(cases) + ["Control"] * len(controls),
            "score": ordered_score,
        }
    )
    score_table.to_csv(
        TABLE_DIR / "GSE294985_signature_scores.csv",
        index=False,
    )
    metrics = {
        "dataset": "GSE294985",
        "comparison": "sepsis versus healthy control",
        "n": int(len(labels)),
        "cases": int(labels.sum()),
        "controls": int((labels == 0).sum()),
        "direction_AUC": float(auc),
        "direction_AUC_95CI_low": ci_low,
        "direction_AUC_95CI_high": ci_high,
        "direction_concordant_genes": int(
            result["direction_concordant"].sum()
        ),
        "total_genes": int(result.shape[0]),
    }
    (RESULT_DIR / "GSE294985_validation_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_data = expression.T.reset_index(names="sample")
    plot_data["group"] = [
        "Sepsis" if sample.endswith("-Seq") else "Control"
        for sample in plot_data["sample"]
    ]
    long_data = plot_data.melt(
        id_vars=["sample", "group"],
        var_name="gene",
        value_name="expression",
    )
    graph = sns.catplot(
        data=long_data,
        x="group",
        y="expression",
        col="gene",
        col_wrap=3,
        kind="box",
        sharey=False,
        showfliers=False,
        height=3.2,
        aspect=1.0,
    )
    graph.map_dataframe(
        sns.stripplot,
        x="group",
        y="expression",
        color="black",
        size=3,
        alpha=0.7,
    )
    graph.fig.suptitle(
        "GSE294985 independent whole-blood validation",
        y=1.02,
    )
    graph.fig.tight_layout()
    graph.savefig(
        FIGURE_DIR / "GSE294985_hub_gene_validation.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(7.5, 5.2))
    sns.boxplot(
        data=score_table,
        x="group",
        y="score",
        showfliers=False,
    )
    sns.stripplot(
        data=score_table,
        x="group",
        y="score",
        color="black",
        size=4,
        alpha=0.8,
    )
    plt.title(
        f"GSE294985 direction score, AUC={auc:.3f} "
        f"(95% CI {ci_low:.3f}-{ci_high:.3f})"
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "GSE294985_signature_score.png",
        dpi=300,
    )
    plt.close()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
