from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
PB_DIR = OUT / "pseudobulk"
PB_DIR.mkdir(parents=True, exist_ok=True)
DATA = Path(
    os.environ.get("SEPSIS_DATA_DIR", ROOT / "data")
).expanduser().resolve()
SOURCE_DATA = Path(
    os.environ.get("SEPSIS_SOURCE_DATA_DIR", ROOT / "source_data")
).expanduser().resolve()
H5AD = Path(
    os.environ.get("GSE207177_H5AD", DATA / "GSE207177_processed.h5ad")
).expanduser().resolve()
RNG = np.random.default_rng(20261019)
HUB_GENES = ["RELA", "ATP6V1B2", "HMOX1", "MITF", "ATP6V1D", "TFE3"]


def load_validation():
    path = ROOT / "analysis" / "05_external_validation_calibration.py"
    spec = importlib.util.spec_from_file_location("validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = DATA
    return module


def sample_scores(
    z_all: pd.DataFrame,
    genes: list[str],
) -> pd.Series:
    return z_all.loc[genes].mean(axis=0)


def module_permutations() -> None:
    validation = load_validation()
    kegg = gp.get_library(name="KEGG_2021_Human")
    modules = {
        "Ferroptosis": {g.upper() for g in kegg["Ferroptosis"]},
        "Mitophagy": {g.upper() for g in kegg["Mitophagy"]},
        "Oxidative phosphorylation": {
            g.upper() for g in kegg["Oxidative phosphorylation"]
        },
    }
    modules["Mitochondrial quality control"] = (
        modules["Mitophagy"] | modules["Oxidative phosphorylation"]
    )
    clp = validation.clp_expression()
    lps = validation.lps_expression()
    datasets = {
        "GSE229925 CLP vs sham": (
            clp,
            [c for c in clp.columns if c.startswith("C")],
            ["S76", "S77", "S79", "S80"],
        ),
        "GSE267388 LPS vs PBS": (
            lps,
            [c for c in lps.columns if "LPS" in c],
            [c for c in lps.columns if "PBS" in c],
        ),
    }
    rows = []
    for dataset, (expression, group1, group2) in datasets.items():
        z_all = expression.sub(expression.mean(axis=1), axis=0).div(
            expression.std(axis=1, ddof=1).replace(0, np.nan), axis=0
        ).dropna(how="any")
        z_matrix = z_all.to_numpy(dtype=float)
        gene_to_index = {gene: i for i, gene in enumerate(z_all.index)}
        col_index = {sample: i for i, sample in enumerate(z_all.columns)}
        group1_idx = np.array([col_index[s] for s in group1])
        group2_idx = np.array([col_index[s] for s in group2])
        mean_expr = expression.mean(axis=1)
        eligible = [g for g in z_all.index if np.isfinite(z_all.loc[g]).all()]
        bins = pd.qcut(
            mean_expr.reindex(eligible).rank(method="first"),
            q=10,
            labels=False,
            duplicates="drop",
        )
        bins = pd.Series(bins.to_numpy(), index=eligible)
        bin_index = {
            g: i for i, g in enumerate(z_all.index)
        }
        for module_name, module_genes in modules.items():
            genes = [g for g in eligible if g in module_genes]
            if len(genes) < 5:
                continue
            scores = z_matrix[[gene_to_index[g] for g in genes]].mean(axis=0)
            observed = scores[group1_idx].mean() - scores[group2_idx].mean()
            target_bins = bins.reindex(genes).to_numpy()
            pools = {
                b: np.array(
                    [bin_index[g] for g in bins[bins == b].index]
                )
                for b in sorted(set(target_bins))
            }
            null = np.empty(5000)
            for i in range(len(null)):
                sampled = [RNG.choice(pools[b]) for b in target_bins]
                random_scores = z_matrix[np.asarray(sampled)].mean(axis=0)
                null[i] = (
                    random_scores[group1_idx].mean()
                    - random_scores[group2_idx].mean()
                )
            p_value = (
                np.sum(np.abs(null) >= abs(observed)) + 1
            ) / (len(null) + 1)
            rows.append(
                {
                    "dataset": dataset,
                    "module": module_name,
                    "n_genes": len(genes),
                    "effect": float(observed),
                    "empirical_p": float(p_value),
                    "null_mean": float(null.mean()),
                    "null_sd": float(null.std(ddof=1)),
                }
            )
    results = pd.DataFrame(rows)
    results["fdr_bh_across_modules"] = multipletests(
        results["empirical_p"], method="fdr_bh"
    )[1]
    results.to_csv(OUT / "module_permutation_results.csv", index=False)
    print(results.to_string(index=False))


def pseudobulk_differential() -> None:
    adata = ad.read_h5ad(H5AD, backed="r")
    directions = pd.read_csv(
        SOURCE_DATA / "top_candidate_genes.csv", index_col=0
    )
    result_rows = []
    candidate_rows = []
    for cell_type in sorted(adata.obs["cell_type"].astype(str).unique()):
        mask = (adata.obs["cell_type"].astype(str) == cell_type).to_numpy()
        if mask.sum() < 100:
            continue
        sub = adata[mask, :]
        obs = sub.obs.copy()
        samples = sorted(obs["sample"].astype(str).unique())
        rows = {}
        for sample in samples:
            sample_mask = (obs["sample"].astype(str) == sample).to_numpy()
            selected = sub.X[sample_mask, :]
            rows[sample] = np.asarray(selected.mean(axis=0)).ravel()
        frame = pd.DataFrame.from_dict(
            rows,
            orient="index",
            columns=[str(g) for g in sub.var_names],
        )
        groups = obs[["sample", "group"]].astype(str).drop_duplicates().set_index("sample")
        group_map = groups.reindex(frame.index)["group"]
        frame = frame.loc[:, frame.mean(axis=0) > 0.05]
        frame.to_csv(PB_DIR / f"{cell_type}.csv")
        comparisons = {
            "CLP_combined_vs_Control": (
                group_map.index[group_map.isin(["CLP_12h", "CLP_24h"])],
                group_map.index[group_map.eq("Control")],
            ),
            "CLP_12h_vs_Control": (
                group_map.index[group_map.eq("CLP_12h")],
                group_map.index[group_map.eq("Control")],
            ),
            "CLP_24h_vs_Control": (
                group_map.index[group_map.eq("CLP_24h")],
                group_map.index[group_map.eq("Control")],
            ),
        }
        for comparison, (case, control) in comparisons.items():
            table = []
            for gene in frame.columns:
                a = frame.loc[case, gene].to_numpy(float)
                b = frame.loc[control, gene].to_numpy(float)
                effect = a.mean() - b.mean()
                try:
                    p_value = stats.ttest_ind(a, b, equal_var=False).pvalue
                except Exception:
                    p_value = np.nan
                table.append((gene, effect, p_value))
            table = pd.DataFrame(table, columns=["gene", "effect", "p_value"])
            table["fdr"] = multipletests(table["p_value"], method="fdr_bh")[1]
            table["cell_type"] = cell_type
            table["comparison"] = comparison
            result_rows.append(
                table.reindex(
                    table["effect"].abs().sort_values(ascending=False).head(500).index
                )
            )
            candidate = table[table["gene"].isin(HUB_GENES)].copy()
            candidate_rows.append(candidate)
    results = pd.concat(result_rows, ignore_index=True)
    candidates = pd.concat(candidate_rows, ignore_index=True)
    results.to_csv(OUT / "pseudobulk_top500_by_celltype.csv", index=False)
    candidates.to_csv(OUT / "pseudobulk_candidate_genes.csv", index=False)
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    module_permutations()
    pseudobulk_differential()
