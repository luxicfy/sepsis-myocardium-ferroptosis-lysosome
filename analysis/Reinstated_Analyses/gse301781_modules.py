from __future__ import annotations

import h5py
import gseapy as gp
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(
    os.environ.get("GSE301781_DATA_DIR", ROOT / "data" / "GSE301781")
).expanduser().resolve()
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
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


def read_10x_h5(path: Path):
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        shape = tuple(matrix["shape"][:])
        genes = matrix["features/name"][:].astype(str)
    x = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    totals = np.asarray(x.sum(axis=1)).ravel()
    scaled = sp.csr_matrix(x.multiply((1e4 / np.maximum(totals, 1))[:, None]))
    scaled.data = np.log1p(scaled.data)
    return scaled, genes


def main() -> None:
    rows = {}
    genes_ref = None
    for path in sorted(DATA.glob("*.h5")):
        gsm = path.name.split("_")[0]
        matrix, genes = read_10x_h5(path)
        if genes_ref is None:
            genes_ref = genes
        elif not np.array_equal(genes_ref, genes):
            raise ValueError("Feature order differs between samples.")
        rows[gsm] = np.asarray(matrix.mean(axis=0)).ravel()
    expression = pd.DataFrame.from_dict(rows, orient="index", columns=genes_ref)
    expression.columns = expression.columns.str.upper()
    expression = expression.T.groupby(level=0).mean().T
    expression = expression.loc[:, expression.mean(axis=0) > 0.05]
    expression.index.name = "sample"

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

    result_rows = []
    score_rows = []
    groups = pd.Series(
        {sample: SAMPLE_GROUP[sample] for sample in expression.index}
    )
    for name, genes in modules.items():
        present = [g for g in expression.columns if g in genes]
        sub = expression[present].T
        z = sub.sub(sub.mean(axis=1), axis=0).div(
            sub.std(axis=1, ddof=1).replace(0, np.nan), axis=0
        )
        score = z.mean(axis=0)
        disease = score[groups == "CLP_6w"]
        control = score[groups == "Control"]
        p_value = stats.ttest_ind(disease, control, equal_var=False).pvalue
        result_rows.append(
            {
                "module": name,
                "n_genes": len(present),
                "control_mean": control.mean(),
                "clp6w_mean": disease.mean(),
                "effect": disease.mean() - control.mean(),
                "p_value": p_value,
            }
        )
        for sample, value in score.items():
            score_rows.append(
                {
                    "module": name,
                    "sample": sample,
                    "group": SAMPLE_GROUP[sample],
                    "score": value,
                }
            )
    results = pd.DataFrame(result_rows)
    results["fdr"] = multipletests(results["p_value"], method="fdr_bh")[1]
    results.to_csv(OUT / "GSE301781_module_scores.csv", index=False)
    pd.DataFrame(score_rows).to_csv(
        OUT / "GSE301781_module_sample_scores.csv", index=False
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
