from __future__ import annotations

import json
import re
from pathlib import Path

import gseapy as gp
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
RESULT_DIR = ROOT / "results"


def collapse_symbols(expression: pd.DataFrame) -> pd.DataFrame:
    expression = expression.copy()
    valid = (
        pd.notna(expression.index)
        & (pd.Series(expression.index, dtype="object").astype(str).str.strip() != "")
        & (
            pd.Series(expression.index, dtype="object").astype(str).str.upper()
            != "NAN"
        )
    )
    expression = expression.loc[valid.to_numpy()]
    expression.index = expression.index.astype(str).str.upper()
    ranked = expression.assign(__mean=expression.mean(axis=1)).sort_values(
        "__mean",
        ascending=False,
    )
    return ranked.groupby(level=0).head(1).drop(columns="__mean")


def filter_expression(expression: pd.DataFrame, min_mean: float) -> pd.DataFrame:
    return expression.loc[expression.mean(axis=1) >= min_mean]


def two_group_effect(expression: pd.DataFrame, group1: list[str], group2: list[str], name: str) -> pd.DataFrame:
    a = expression[group1].to_numpy(dtype=float)
    b = expression[group2].to_numpy(dtype=float)
    test = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy="omit")
    effect = a.mean(axis=1) - b.mean(axis=1)
    fdr = multipletests(np.asarray(test.pvalue), method="fdr_bh")[1]
    return pd.DataFrame(
        {
            "gene": expression.index,
            f"{name}_effect": effect,
            f"{name}_p": test.pvalue,
            f"{name}_fdr": fdr,
        }
    ).set_index("gene")


def module_score(expression: pd.DataFrame, genes: set[str]) -> tuple[pd.Series, int]:
    present = [gene for gene in expression.index if gene in genes]
    if not present:
        raise ValueError("No module genes are present.")
    sub = expression.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1, ddof=1).replace(0, np.nan),
        axis=0,
    )
    return z.mean(axis=0), len(present)


def module_test(
    expression: pd.DataFrame,
    genes: set[str],
    group1: list[str],
    group2: list[str],
    name: str,
) -> dict[str, float | str]:
    scores, n_genes = module_score(expression, genes)
    a = scores.loc[group1].to_numpy(dtype=float)
    b = scores.loc[group2].to_numpy(dtype=float)
    test = stats.ttest_ind(a, b, equal_var=False)
    return {
        "dataset": name,
        "effect": float(a.mean() - b.mean()),
        "p_value": float(test.pvalue),
        "n_genes": n_genes,
    }


def pca_plot(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    top = expression.var(axis=1).sort_values(ascending=False).head(2000).index
    scaled = StandardScaler().fit_transform(expression.loc[top].T)
    pca = PCA(n_components=2, random_state=42).fit(scaled)
    scores = pca.transform(scaled)
    plot_data = metadata.copy()
    plot_data["PC1"] = scores[:, 0]
    plot_data["PC2"] = scores[:, 1]
    plt.figure(figsize=(7, 5.4))
    sns.scatterplot(
        data=plot_data,
        x="PC1",
        y="PC2",
        hue="group",
        style="group",
        s=80,
        edgecolor="black",
        linewidth=0.5,
    )
    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def volcano_plot(result: pd.DataFrame, effect_col: str, fdr_col: str, output: Path, title: str) -> None:
    plot_data = result.copy()
    plot_data["minus_log10_fdr"] = -np.log10(plot_data[fdr_col].clip(lower=1e-300))
    plot_data["class"] = "Not significant"
    plot_data.loc[
        (plot_data[fdr_col] < 0.05) & (plot_data[effect_col] > 0.5),
        "class",
    ] = "Up"
    plot_data.loc[
        (plot_data[fdr_col] < 0.05) & (plot_data[effect_col] < -0.5),
        "class",
    ] = "Down"
    plt.figure(figsize=(7, 5.6))
    sns.scatterplot(
        data=plot_data,
        x=effect_col,
        y="minus_log10_fdr",
        hue="class",
        palette={"Not significant": "#b8b8b8", "Up": "#c44e52", "Down": "#4c72b0"},
        s=16,
        linewidth=0,
    )
    plt.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(-0.5, color="black", linestyle="--", linewidth=0.8)
    plt.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8)
    plt.title(title)
    plt.xlabel("Adjusted log2 effect")
    plt.ylabel("-log10(FDR)")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def read_human_validation() -> pd.DataFrame:
    supplement = DATA_DIR / "GSE79962_original_supplement.xlsx"
    table5 = pd.read_excel(supplement, sheet_name="SupplTable5", header=1)
    table5 = table5.rename(
        columns={
            "Gene Symbol": "gene",
            "Fold-Change(Septic vs NF)": "human_fold_change",
            "p-value(Septic vs NF)": "human_p",
        }
    )
    table5 = table5[["gene", "human_fold_change", "human_p"]]
    table5["gene"] = table5["gene"].astype(str).str.upper()
    table5 = table5.loc[table5["gene"].ne("NAN")]
    human_fold_change = pd.to_numeric(
        table5["human_fold_change"],
        errors="coerce",
    )
    table5["human_effect"] = np.sign(human_fold_change) * np.log2(
        human_fold_change.abs().clip(lower=1e-6)
    )
    table5["human_direction"] = np.where(
        human_fold_change > 0,
        "up",
        "down",
    )
    return table5.set_index("gene")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # GSE229925: CLP phenotypes vs sham.
    clp = pd.read_csv(
        DATA_DIR / "GSE229925_anno_genes.FPKM.txt.gz",
        sep="\t",
    )
    sample_columns = [column for column in clp.columns if column.startswith(("C", "S"))]
    sample_to_group = {
        "S76": "Sham",
        "S77": "Sham",
        "S79": "Sham",
        "S80": "Sham",
        "C85": "HEF",
        "C86": "HEF",
        "C92": "HEF",
        "C97": "HEF",
        "C81": "LEF",
        "C83": "LEF",
        "C93": "LEF",
        "C84": "NEF",
        "C88": "NEF",
        "C90": "NEF",
        "C95": "NEF",
    }
    sample_columns = [column for column in sample_columns if column in sample_to_group]
    metadata = pd.DataFrame(
        {
            "sample": sample_columns,
            "group": [sample_to_group[column] for column in sample_columns],
        }
    ).set_index("sample")
    expression = clp.set_index("gene_name")[sample_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    expression = filter_expression(
        collapse_symbols(np.log2(expression + 1)),
        min_mean=0.5,
    )
    sham = metadata.index[metadata["group"].eq("Sham")].tolist()
    clp_samples = metadata.index[metadata["group"].isin(["HEF", "LEF", "NEF"])].tolist()
    all_clp = two_group_effect(expression, clp_samples, sham, "clp")
    all_clp.to_csv(TABLE_DIR / "GSE229925_CLP_vs_sham.csv")
    for subtype in ["HEF", "LEF", "NEF"]:
        samples = metadata.index[metadata["group"].eq(subtype)].tolist()
        two_group_effect(expression, samples, sham, subtype.lower()).to_csv(
            TABLE_DIR / f"GSE229925_{subtype}_vs_sham.csv"
        )
    pca_plot(
        expression,
        metadata,
        FIGURE_DIR / "GSE229925_PCA.png",
        "GSE229925 sepsis-induced myocardial injury",
    )
    volcano_plot(
        all_clp,
        "clp_effect",
        "clp_fdr",
        FIGURE_DIR / "GSE229925_CLP_volcano.png",
        "CLP vs sham",
    )

    # GSE267388: LPS vs PBS validation.
    lps = pd.read_csv(
        DATA_DIR / "GSE267388_gene_expression.xls.gz",
        sep="\t",
        compression="gzip",
    )
    lps["gene_symbol"] = lps["gene_symbol"].astype(str)
    lps_samples = [column for column in lps.columns if column.startswith("fpkm_")]
    lps_expression = lps.set_index("gene_symbol")[lps_samples].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    lps_expression = filter_expression(
        collapse_symbols(np.log2(lps_expression + 1)),
        min_mean=0.5,
    )
    lps_samples = [column for column in lps_expression.columns if "WT_LPS" in column]
    pbs_samples = [column for column in lps_expression.columns if "WT_PBS" in column]
    lps_result = two_group_effect(
        lps_expression,
        lps_samples,
        pbs_samples,
        "lps",
    )
    lps_result.to_csv(TABLE_DIR / "GSE267388_LPS_vs_PBS.csv")
    lps_meta = pd.DataFrame(
        {
            "sample": lps_expression.columns,
            "group": ["LPS" if "LPS" in column else "PBS" for column in lps_expression.columns],
        }
    ).set_index("sample")
    pca_plot(
        lps_expression,
        lps_meta,
        FIGURE_DIR / "GSE267388_PCA.png",
        "GSE267388 LPS-induced myocardial injury",
    )
    volcano_plot(
        lps_result,
        "lps_effect",
        "lps_fdr",
        FIGURE_DIR / "GSE267388_LPS_volcano.png",
        "LPS vs PBS",
    )

    # Human validation from the original GSE79962 publication.
    human = read_human_validation()
    human.to_csv(TABLE_DIR / "GSE79962_human_validation.csv")

    kegg = gp.get_library(name="KEGG_2021_Human")
    modules = {
        "Ferroptosis": {gene.upper() for gene in kegg["Ferroptosis"]},
        "Mitophagy": {gene.upper() for gene in kegg["Mitophagy"]},
        "Oxidative phosphorylation": {
            gene.upper() for gene in kegg["Oxidative phosphorylation"]
        },
    }
    modules["Mitochondrial quality control"] = (
        modules["Mitophagy"] | modules["Oxidative phosphorylation"]
    )

    module_rows = []
    for dataset_name, expression_data, group1, group2 in [
        ("GSE229925_CLP_vs_sham", expression, clp_samples, sham),
        ("GSE267388_LPS_vs_PBS", lps_expression, lps_samples, pbs_samples),
    ]:
        for module_name, genes in modules.items():
            row = module_test(
                expression_data,
                genes,
                group1,
                group2,
                dataset_name,
            )
            row["module"] = module_name
            module_rows.append(row)
    module_results = pd.DataFrame(module_rows)
    module_results.to_csv(TABLE_DIR / "mouse_module_scores.csv", index=False)

    mechanisms = modules["Mitochondrial quality control"] | modules["Ferroptosis"]
    combined = all_clp.join(
        lps_result,
        how="inner",
    ).join(
        human[["human_effect", "human_p", "human_direction"]],
        how="inner",
    )
    combined["mechanistic_set"] = combined.index.isin(mechanisms)
    combined["mouse_same_direction"] = (
        np.sign(combined["clp_effect"]) == np.sign(combined["lps_effect"])
    )
    combined["human_same_direction"] = (
        np.sign(combined["clp_effect"])
        == np.where(combined["human_effect"] > 0, 1, -1)
    )
    combined["evidence_layers"] = (
        combined["mechanistic_set"].astype(int)
        + combined["mouse_same_direction"].astype(int)
        + combined["human_same_direction"].astype(int)
        + (combined["clp_fdr"] < 0.1).astype(int)
        + (combined["lps_fdr"] < 0.1).astype(int)
        + (combined["human_p"] < 0.01).astype(int)
    )
    combined["candidate_score"] = (
        -np.log10(combined["clp_fdr"].clip(lower=1e-300))
        + -np.log10(combined["lps_fdr"].clip(lower=1e-300))
        + -np.log10(combined["human_p"].clip(lower=1e-300))
        + combined["evidence_layers"]
    )
    combined = combined.sort_values("candidate_score", ascending=False)
    combined.to_csv(TABLE_DIR / "cross_species_candidate_genes.csv")

    candidates = combined.loc[
        combined["mechanistic_set"]
        & combined["mouse_same_direction"]
        & combined["human_same_direction"]
        & (
            (combined["clp_p"] < 0.05)
            | (combined["lps_p"] < 0.05)
        )
    ].head(100)
    candidates.to_csv(TABLE_DIR / "top_candidate_genes.csv")

    summary = {
        "clp_genes_tested": int(all_clp.shape[0]),
        "clp_significant_fdr_0_05": int((all_clp["clp_fdr"] < 0.05).sum()),
        "lps_genes_tested": int(lps_result.shape[0]),
        "lps_significant_fdr_0_05": int((lps_result["lps_fdr"] < 0.05).sum()),
        "human_validation_genes": int(human.shape[0]),
        "mechanistic_candidates": int(candidates.shape[0]),
        "top_genes": candidates.head(20).index.tolist(),
        "module_results": module_results.to_dict(orient="records"),
    }
    (RESULT_DIR / "bulk_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
