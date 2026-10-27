from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    os.environ.get("GSE301781_DATA_DIR", ROOT / "data" / "GSE301781")
).expanduser().resolve()
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261019)
HUB = {
    "RELA": 1,
    "ATP6V1B2": 1,
    "HMOX1": 1,
    "MITF": -1,
    "ATP6V1D": -1,
    "TFE3": 1,
}
SAMPLE_GROUP = {
    "GSM9089582": "Control",
    "GSM9089583": "Control",
    "GSM9089584": "Control",
    "GSM9089585": "Control",
    "GSM9089586": "CLP_6w",
    "GSM9089587": "CLP_6w",
    "GSM9089588": "CLP_6w",
    "GSM9089589": "CLP_6w",
    "GSM9089590": "Control",
    "GSM9089591": "Control",
    "GSM9089592": "CLP_6w",
    "GSM9089593": "CLP_6w",
}


def scale_counts(x):
    totals = np.asarray(x.sum(axis=1)).ravel()
    scale = 1e4 / np.maximum(totals, 1)
    scaled = x.multiply(scale[:, None])
    scaled = sp.csr_matrix(scaled)
    scaled.data = np.log1p(scaled.data)
    return scaled


def read_10x_h5(path: Path) -> ad.AnnData:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = tuple(matrix["shape"][:])
        barcodes = matrix["barcodes"][:].astype(str)
        genes = matrix["features/name"][:].astype(str)
    x = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    return ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=genes),
    )


def main() -> None:
    rows = {}
    expression_rows = {}
    for path in sorted(DATA.glob("*.h5")):
        gsm = path.name.split("_")[0]
        adata = read_10x_h5(path)
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()
        adata.X = scale_counts(adata.X)
        upper = pd.Index(adata.var_names.str.upper())
        present = [g for g in HUB if g in upper]
        indices = [int(upper.get_loc(g)) for g in present]
        mean = np.asarray(adata.X[:, indices].mean(axis=0)).ravel()
        expression_rows[gsm] = dict(zip(present, mean))
        rows[gsm] = {
            "sample": gsm,
            "group": SAMPLE_GROUP[gsm],
            "n_cells": adata.n_obs,
        }
        print(gsm, SAMPLE_GROUP[gsm], adata.n_obs, flush=True)
    expression = pd.DataFrame.from_dict(expression_rows, orient="index")
    metadata = pd.DataFrame.from_dict(rows, orient="index")
    expression = expression.reindex(
        columns=[g for g in HUB if g in expression.columns]
    )
    expression.to_csv(OUT / "GSE301781_candidate_pseudobulk.csv")
    metadata.to_csv(OUT / "GSE301781_sample_metadata.csv")

    gene_rows = []
    for gene in expression.columns:
        disease = expression.loc[metadata.group.eq("CLP_6w"), gene]
        control = expression.loc[metadata.group.eq("Control"), gene]
        effect = disease.mean() - control.mean()
        p_value = stats.ttest_ind(disease, control, equal_var=False).pvalue
        expected = "up" if HUB[gene] > 0 else "down"
        observed = "up" if effect > 0 else "down"
        gene_rows.append(
            {
                "gene": gene,
                "control_mean": control.mean(),
                "clp6w_mean": disease.mean(),
                "effect": effect,
                "p_value": p_value,
                "expected_direction": expected,
                "observed_direction": observed,
                "concordant": expected == observed,
            }
        )
    genes = pd.DataFrame(gene_rows)
    genes["fdr"] = multipletests(genes["p_value"], method="fdr_bh")[1]
    genes.to_csv(OUT / "GSE301781_candidate_gene_validation.csv", index=False)

    z = expression.sub(expression.mean(axis=0), axis=1).div(
        expression.std(axis=0, ddof=1).replace(0, np.nan), axis=1
    )
    direction = np.array([HUB[g] for g in expression.columns])
    score = pd.Series(
        np.nanmean(z.to_numpy() * direction, axis=1),
        index=expression.index,
    )
    labels = metadata.group.eq("CLP_6w").astype(int).reindex(score.index).to_numpy()
    auc = roc_auc_score(labels, score.to_numpy())
    null = []
    for _ in range(10000):
        permuted = RNG.permutation(labels)
        null.append(roc_auc_score(permuted, score.to_numpy()))
    p_one_sided = (np.sum(np.asarray(null) >= auc) + 1) / 10001
    score_table = pd.DataFrame(
        {
            "sample": score.index,
            "group": metadata.group.reindex(score.index).to_numpy(),
            "score": score.to_numpy(),
        }
    )
    score_table.to_csv(OUT / "GSE301781_direction_scores.csv", index=False)
    summary = {
        "dataset": "GSE301781",
        "comparison": "CLP 6 weeks vs control, cardiac CD45+ cells",
        "n_control": int((labels == 0).sum()),
        "n_clp": int((labels == 1).sum()),
        "direction_auc": float(auc),
        "permutation_p_one_sided": float(p_one_sided),
        "concordant_genes": int(genes["concordant"].sum()),
        "total_genes": int(len(genes)),
    }
    (OUT / "GSE301781_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(genes.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
