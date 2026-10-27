from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import anndata as ad
import gseapy as gp
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EXTRACT_DIR = DATA_DIR / "GSE207177"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
RESULT_DIR = ROOT / "results"


def extract_archive() -> None:
    archive = DATA_DIR / "GSE207177_RAW.tar"
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.glob("*.matrix.mtx.gz")):
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile()
            and member.name.endswith((".mtx.gz", ".tsv.gz"))
        ]
        tar.extractall(EXTRACT_DIR, members=members)


def sample_group(prefix: str) -> tuple[str, str]:
    if "clp-12h" in prefix:
        return "CLP_12h", prefix
    if "clp-24h" in prefix:
        return "CLP_24h", prefix
    return "Control", prefix


def discover_samples() -> dict[str, ad.AnnData]:
    samples: dict[str, ad.AnnData] = {}
    for matrix in sorted(EXTRACT_DIR.glob("*.matrix.mtx.gz")):
        prefix = matrix.name.removesuffix(".matrix.mtx.gz")
        barcodes = EXTRACT_DIR / f"{prefix}.barcodes.tsv.gz"
        features = EXTRACT_DIR / f"{prefix}.features.tsv.gz"
        sample_dir = EXTRACT_DIR / prefix
        sample_dir.mkdir(exist_ok=True)
        shutil.copy2(matrix, sample_dir / "matrix.mtx.gz")
        shutil.copy2(barcodes, sample_dir / "barcodes.tsv.gz")
        shutil.copy2(features, sample_dir / "features.tsv.gz")
        samples[prefix] = sc.read_10x_mtx(sample_dir)
    if len(samples) < 6:
        raise RuntimeError(f"Expected six samples, found {len(samples)}")
    return samples


def row_zscore(values: pd.DataFrame) -> pd.DataFrame:
    means = values.mean(axis=1)
    standard_deviations = values.std(axis=1, ddof=1).replace(0, np.nan)
    return values.sub(means, axis=0).div(standard_deviations, axis=0)


def annotate_clusters(adata: ad.AnnData) -> None:
    marker_sets = {
        "Cardiomyocyte": ["NPPA", "NPPB", "MYH7", "ACTC1", "TNNT2", "RYR2"],
        "Fibroblast": ["DCN", "COL1A1", "POSTN", "PDGFRA", "LUM"],
        "Endothelial": ["PECAM1", "CDH5", "KDR", "EMCN"],
        "Macrophage": ["C1QA", "C1QB", "ADGRE1", "LYZ2", "CD68", "CSF1R"],
        "Monocyte": ["LY6C2", "CCR2", "F13A1", "VCAN"],
        "Neutrophil": ["S100A8", "S100A9", "LY6G", "RETNLG"],
        "T_NK": ["CD3D", "CD3E", "NKG7", "CCL5"],
        "B_cell": ["CD79A", "MS4A1", "CD74"],
        "Pericyte_SMC": ["RGS5", "PDGFRB", "MYH11", "ACTA2"],
    }
    rows = []
    for cell_type, genes in marker_sets.items():
        present = [gene for gene in genes if gene in adata.var_names]
        if not present:
            continue
        score_name = f"score_{cell_type}"
        sc.tl.score_genes(adata, present, score_name=score_name, random_state=42)
        values = adata.obs.groupby("leiden", observed=True)[score_name].mean()
        for cluster, score in values.items():
            rows.append(
                {
                    "cluster": cluster,
                    "cell_type": cell_type,
                    "score": float(score),
                    "n_marker_genes": len(present),
                }
            )
    score_table = pd.DataFrame(rows)
    score_table.to_csv(
        TABLE_DIR / "GSE207177_cluster_marker_scores.csv",
        index=False,
    )
    annotation = (
        score_table.sort_values("score", ascending=False)
        .drop_duplicates("cluster")
        .set_index("cluster")["cell_type"]
    )
    adata.obs["cell_type"] = adata.obs["leiden"].map(annotation).astype("category")


def candidate_tables(adata: ad.AnnData) -> None:
    genes = ["RELA", "HMOX1", "ATP6V1B2", "MITF", "ATP6V1D", "TFE3"]
    present = [gene for gene in genes if gene in adata.var_names]
    rows = []
    for cell_type, sub in adata.obs.groupby("cell_type", observed=True):
        subset = adata[sub.index, present]
        values = subset.X.toarray() if hasattr(subset.X, "toarray") else subset.X
        for idx, gene in enumerate(present):
            expression = np.asarray(values[:, idx]).ravel()
            rows.append(
                {
                    "cell_type": cell_type,
                    "gene": gene,
                    "mean_expression": float(expression.mean()),
                    "fraction_expressing": float((expression > 0).mean()),
                    "n_cells": int(subset.n_obs),
                }
            )
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "GSE207177_candidate_celltype_expression.csv",
        index=False,
    )

    sample_rows = []
    for (group, cell_type), sub in adata.obs.groupby(
        ["group", "cell_type"],
        observed=True,
    ):
        subset = adata[sub.index, present]
        values = subset.X.toarray() if hasattr(subset.X, "toarray") else subset.X
        for idx, gene in enumerate(present):
            sample_rows.append(
                {
                    "group": group,
                    "cell_type": cell_type,
                    "gene": gene,
                    "mean_expression": float(
                        np.asarray(values[:, idx]).ravel().mean()
                    ),
                    "n_cells": int(subset.n_obs),
                }
            )
    pd.DataFrame(sample_rows).to_csv(
        TABLE_DIR / "GSE207177_candidate_group_celltype_expression.csv",
        index=False,
    )


def module_tables(adata: ad.AnnData) -> None:
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
    rows = []
    for module, genes in modules.items():
        present = [gene for gene in genes if gene in adata.var_names]
        score_name = f"score_{module.replace(' ', '_')}"
        sc.tl.score_genes(adata, present, score_name=score_name, random_state=42)
        for cell_type, value in (
            adata.obs.groupby("cell_type", observed=True)[score_name].mean().items()
        ):
            rows.append(
                {
                    "module": module,
                    "cell_type": cell_type,
                    "mean_score": float(value),
                    "n_genes": len(present),
                }
            )
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "GSE207177_celltype_module_scores.csv",
        index=False,
    )


def figures(adata: ad.AnnData) -> None:
    sc.pl.umap(
        adata,
        color=["group", "cell_type", "leiden"],
        ncols=2,
        wspace=0.35,
        show=False,
    )
    plt.savefig(
        FIGURE_DIR / "GSE207177_UMAP.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    genes = [
        gene
        for gene in ["RELA", "HMOX1", "ATP6V1B2", "MITF", "ATP6V1D", "TFE3"]
        if gene in adata.var_names
    ]
    sc.pl.dotplot(
        adata,
        genes,
        groupby="cell_type",
        standard_scale="var",
        show=False,
    )
    plt.savefig(
        FIGURE_DIR / "GSE207177_candidate_dotplot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    extract_archive()
    samples = discover_samples()

    adatas = []
    for sample, sample_adata in samples.items():
        group, sample_id = sample_group(sample.lower())
        sample_adata.var_names = sample_adata.var_names.str.upper()
        sample_adata.var_names_make_unique()
        sample_adata.obs["sample"] = sample_id
        sample_adata.obs["group"] = group
        adatas.append(sample_adata)

    adata = ad.concat(
        adatas,
        join="outer",
        fill_value=0,
        label="source_sample",
        index_unique="-",
    )
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    adata = adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["pct_counts_mt"] <= 20),
    ].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata_full = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=30, random_state=42)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=42)
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(
        adata,
        resolution=0.8,
        random_state=42,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    adata_full.obs["leiden"] = adata.obs["leiden"].astype("category")
    adata_full.obsm["X_pca"] = adata.obsm["X_pca"]
    adata_full.obsm["X_umap"] = adata.obsm["X_umap"]
    adata_full.obsp = adata.obsp
    adata_full.uns = adata.uns

    annotate_clusters(adata_full)
    candidate_tables(adata_full)
    module_tables(adata_full)
    figures(adata_full)
    adata_full.write_h5ad(DATA_DIR / "GSE207177_processed.h5ad")

    composition = (
        adata_full.obs.groupby(["group", "cell_type"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    composition.to_csv(
        TABLE_DIR / "GSE207177_group_celltype_counts.csv",
        index=False,
    )
    summary = {
        "n_cells_after_qc": int(adata_full.n_obs),
        "n_genes_after_qc": int(adata_full.n_vars),
        "n_genes_for_clustering": int(adata.n_vars),
        "groups": sorted(adata_full.obs["group"].unique().tolist()),
        "cell_types": sorted(adata_full.obs["cell_type"].unique().tolist()),
        "clusters": sorted(adata_full.obs["leiden"].unique().tolist()),
    }
    (RESULT_DIR / "single_cell_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
