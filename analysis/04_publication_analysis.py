from __future__ import annotations

import json
from pathlib import Path

import gseapy as gp
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
RESULT_DIR = ROOT / "results"


def collapse_symbols(expression: pd.DataFrame) -> pd.DataFrame:
    expression = expression.copy()
    expression.index = expression.index.astype(str).str.upper()
    valid = (
        pd.notna(expression.index)
        & (pd.Series(expression.index, dtype="object").astype(str) != "")
        & (
            pd.Series(expression.index, dtype="object").astype(str).str.upper()
            != "NAN"
        )
    )
    expression = expression.loc[np.asarray(valid, dtype=bool)]
    ranked = expression.assign(__mean=expression.mean(axis=1)).sort_values(
        "__mean",
        ascending=False,
    )
    return ranked.groupby(level=0).head(1).drop(columns="__mean")


def row_zscore(values: pd.DataFrame) -> pd.DataFrame:
    means = values.mean(axis=1)
    sds = values.std(axis=1, ddof=1).replace(0, np.nan)
    return values.sub(means, axis=0).div(sds, axis=0)


def recreate_clp_expression() -> pd.DataFrame:
    clp = pd.read_csv(
        DATA_DIR / "GSE229925_anno_genes.FPKM.txt.gz",
        sep="\t",
    )
    sample_columns = [
        column
        for column in clp.columns
        if column in {
            "S76",
            "S77",
            "S79",
            "S80",
            "C85",
            "C86",
            "C92",
            "C97",
            "C81",
            "C83",
            "C93",
            "C84",
            "C88",
            "C90",
            "C95",
        }
    ]
    expression = clp.set_index("gene_name")[sample_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    expression = np.log2(expression + 1)
    expression = collapse_symbols(expression)
    return expression.loc[expression.mean(axis=1) >= 0.5]


def recreate_lps_expression() -> pd.DataFrame:
    lps = pd.read_csv(
        DATA_DIR / "GSE267388_gene_expression.xls.gz",
        sep="\t",
        compression="gzip",
    )
    samples = [column for column in lps.columns if column.startswith("fpkm_")]
    expression = lps.set_index("gene_symbol")[samples].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    expression = np.log2(expression + 1)
    expression = collapse_symbols(expression)
    return expression.loc[expression.mean(axis=1) >= 0.5]


def subtype_analysis(hub_genes: list[str]) -> pd.DataFrame:
    frames = {}
    for subtype in ["HEF", "LEF", "NEF"]:
        table = pd.read_csv(
            TABLE_DIR / f"GSE229925_{subtype}_vs_sham.csv",
            index_col=0,
        )
        frames[subtype] = table
    rows = []
    for gene in hub_genes:
        row = {"gene": gene}
        for subtype, frame in frames.items():
            effect_col = f"{subtype.lower()}_effect"
            fdr_col = f"{subtype.lower()}_fdr"
            row[f"{subtype}_effect"] = (
                frame.loc[gene, effect_col] if gene in frame.index else np.nan
            )
            row[f"{subtype}_fdr"] = (
                frame.loc[gene, fdr_col] if gene in frame.index else np.nan
            )
        signs = [
            np.sign(row[f"{subtype}_effect"])
            for subtype in ["HEF", "LEF", "NEF"]
        ]
        row["same_direction_all_subtypes"] = len(set(signs)) == 1
        rows.append(row)
    result = pd.DataFrame(rows).set_index("gene")
    result.to_csv(TABLE_DIR / "CLP_subtype_consistency.csv")

    heatmap = result[[f"{s}_effect" for s in ["HEF", "LEF", "NEF"]]]
    plt.figure(figsize=(6.5, max(3.5, len(heatmap) * 0.45)))
    sns.heatmap(
        heatmap,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "log2 effect"},
    )
    plt.title("Hub gene effects across CLP subtypes")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "CLP_subtype_consistency_heatmap.png", dpi=300)
    plt.close()
    return result


def auc_analysis(
    expression: pd.DataFrame,
    positive_samples: list[str],
    negative_samples: list[str],
    hub_genes: list[str],
    dataset: str,
) -> pd.DataFrame:
    present = [gene for gene in hub_genes if gene in expression.index]
    labels = np.array([1] * len(positive_samples) + [0] * len(negative_samples))
    rows = []
    z = row_zscore(expression.loc[present])
    for gene in present:
        values = expression.loc[gene, positive_samples + negative_samples].to_numpy()
        auc = roc_auc_score(labels, values)
        if auc < 0.5:
            auc = 1 - auc
        rows.append({"dataset": dataset, "gene": gene, "auc": auc, "type": "single"})

    directions = pd.read_csv(TABLE_DIR / "top_candidate_genes.csv", index_col=0)
    direction_sign = np.array(
        [1 if directions.loc[gene, "clp_effect"] > 0 else -1 for gene in present]
    )
    panel_values = (z.loc[present].to_numpy().T * direction_sign).mean(axis=1)
    panel_frame = pd.Series(
        panel_values,
        index=expression.columns,
    )
    panel_auc = roc_auc_score(
        labels,
        panel_frame.loc[positive_samples + negative_samples].to_numpy(),
    )
    if panel_auc < 0.5:
        panel_auc = 1 - panel_auc
    rows.append(
        {
            "dataset": dataset,
            "gene": "Six-gene panel",
            "auc": panel_auc,
            "type": "panel",
        }
    )

    x = z.loc[present].T.loc[positive_samples + negative_samples]
    loo = LeaveOneOut()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", random_state=42),
    )
    probabilities = cross_val_predict(
        model,
        x,
        labels,
        cv=loo,
        method="predict_proba",
    )[:, 1]
    loo_auc = roc_auc_score(labels, probabilities)
    if loo_auc < 0.5:
        loo_auc = 1 - loo_auc
    rows.append(
        {
            "dataset": dataset,
            "gene": "Six-gene LOOCV model",
            "auc": loo_auc,
            "type": "LOOCV",
        }
    )
    return pd.DataFrame(rows)


def correlation_analysis(
    expression: pd.DataFrame,
    samples: list[str],
    hub_genes: list[str],
    dataset: str,
) -> pd.DataFrame:
    genes = [gene for gene in hub_genes if gene in expression.index]
    corr = expression.loc[genes, samples].T.corr(method="spearman")
    corr.to_csv(TABLE_DIR / f"hub_gene_correlation_{dataset}.csv")
    return corr


def enrichment_analysis() -> None:
    clp = pd.read_csv(TABLE_DIR / "GSE229925_CLP_vs_sham.csv", index_col=0)
    lps = pd.read_csv(TABLE_DIR / "GSE267388_LPS_vs_PBS.csv", index_col=0)
    for dataset, table, effect_col, fdr_col in [
        ("CLP", clp, "clp_effect", "clp_fdr"),
        ("LPS", lps, "lps_effect", "lps_fdr"),
    ]:
        significant = table.loc[
            (table[fdr_col] < 0.05) & (table[effect_col].abs() > 0.5)
        ]
        genes = (
            significant.assign(abs_effect=significant[effect_col].abs())
            .sort_values("abs_effect", ascending=False)
            .head(500)
            .index.tolist()
        )
        result = gp.enrichr(
            gene_list=genes,
            gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2025"],
            organism="human",
            outdir=None,
            no_plot=True,
        ).results.sort_values("Adjusted P-value")
        result.to_csv(TABLE_DIR / f"{dataset}_publication_enrichment.csv", index=False)


def single_cell_plots() -> None:
    composition = pd.read_csv(
        TABLE_DIR / "GSE207177_group_celltype_counts.csv"
    )
    composition["fraction"] = composition["n_cells"] / composition.groupby(
        "group"
    )["n_cells"].transform("sum")
    matrix = composition.pivot(
        index="cell_type",
        columns="group",
        values="fraction",
    ).fillna(0)
    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(matrix, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Cell-type composition")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "single_cell_composition_heatmap.png", dpi=300)
    plt.close()

    module_scores = pd.read_csv(
        TABLE_DIR / "GSE207177_celltype_module_scores.csv"
    )
    module_matrix = module_scores.pivot(
        index="cell_type",
        columns="module",
        values="mean_score",
    )
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(module_matrix, cmap="vlag", center=0)
    plt.title("Single-cell module scores")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "single_cell_module_heatmap.png", dpi=300)
    plt.close()


def workflow_plot() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, 0.18, 0.25, "GSE229925\nMouse CLP heart"),
        (0.03, 0.15, 0.18, 0.25, "GSE267388\nMouse LPS heart"),
        (0.29, 0.42, 0.16, 0.25, "Cross-species\nmechanistic analysis"),
        (0.52, 0.62, 0.17, 0.25, "GSE79962\nHuman septic heart"),
        (0.52, 0.15, 0.17, 0.25, "GSE207177\nSingle-cell localization"),
        (0.78, 0.42, 0.19, 0.25, "Ferroptosis-lysosome\nhub gene panel"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="#eef3f8",
                edgecolor="#335c81",
                linewidth=1.5,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((0.21, 0.74), (0.29, 0.58)),
        ((0.21, 0.27), (0.29, 0.50)),
        ((0.45, 0.58), (0.52, 0.72)),
        ((0.45, 0.50), (0.52, 0.27)),
        ((0.69, 0.74), (0.78, 0.58)),
        ((0.69, 0.27), (0.78, 0.49)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=14,
                color="#335c81",
                linewidth=1.5,
            )
        )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "study_workflow.png", dpi=300)
    plt.close()


def main() -> None:
    candidates = pd.read_csv(TABLE_DIR / "top_candidate_genes.csv", index_col=0)
    hub_genes = candidates.index.tolist()

    clp_expression = recreate_clp_expression()
    lps_expression = recreate_lps_expression()
    subtype_table = subtype_analysis(hub_genes)

    clp_positive = ["C85", "C86", "C92", "C97", "C81", "C83", "C93", "C84", "C88", "C90", "C95"]
    clp_negative = ["S76", "S77", "S79", "S80"]
    lps_positive = [column for column in lps_expression.columns if "LPS" in column]
    lps_negative = [column for column in lps_expression.columns if "PBS" in column]
    auc_table = pd.concat(
        [
            auc_analysis(
                clp_expression,
                clp_positive,
                clp_negative,
                hub_genes,
                "CLP",
            ),
            auc_analysis(
                lps_expression,
                lps_positive,
                lps_negative,
                hub_genes,
                "LPS",
            ),
        ],
        ignore_index=True,
    )
    auc_table.to_csv(TABLE_DIR / "hub_gene_AUC.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=auc_table, x="gene", y="auc", hue="dataset")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    plt.ylim(0.4, 1.02)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("AUC")
    plt.title("Hub gene discrimination performance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hub_gene_AUC.png", dpi=300)
    plt.close()

    clp_corr = correlation_analysis(
        clp_expression,
        clp_positive + clp_negative,
        hub_genes,
        "CLP",
    )
    lps_corr = correlation_analysis(
        lps_expression,
        lps_positive + lps_negative,
        hub_genes,
        "LPS",
    )
    combined_corr = (clp_corr + lps_corr) / 2
    plt.figure(figsize=(6.5, 5.8))
    sns.heatmap(
        combined_corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
    )
    plt.title("Average hub-gene Spearman correlation")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hub_gene_correlation_heatmap.png", dpi=300)
    plt.close()

    enrichment_analysis()
    single_cell_plots()
    workflow_plot()

    summary = {
        "hub_genes": hub_genes,
        "subtypes_same_direction": subtype_table[
            "same_direction_all_subtypes"
        ].to_dict(),
        "auc": auc_table.to_dict(orient="records"),
    }
    (RESULT_DIR / "publication_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
