from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
H5AD = Path(
    os.environ.get(
        "GSE207177_H5AD",
        Path(os.environ.get("SEPSIS_DATA_DIR", ROOT / "data"))
        / "GSE207177_processed.h5ad",
    )
).expanduser().resolve()
RNG = np.random.default_rng(20261018)
TARGET_CELLS = {
    "HMOX1": "Macrophage",
    "TFE3": "Monocyte",
    "RELA": "Pericyte_SMC",
}
N_HVG = 300
ALPHA = 0.5
THRESHOLD = 0.05
N_NULL = 2000
SENSITIVITY_THRESHOLDS = [0.03, 0.05, 0.10]


def choose_network_genes(
    x,
    var_names: pd.Index,
    target: str,
) -> tuple[list[int], list[str]]:
    mean = np.asarray(x.mean(axis=0)).ravel()
    second = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    variance = second - mean**2
    order = np.argsort(variance)[::-1]
    selected = [int(i) for i in order[:N_HVG]]
    target_idx = int(var_names.get_loc(target))
    if target_idx not in selected:
        selected.append(target_idx)
    return selected, [str(g) for g in var_names[selected]]


def propagate(
    inverse: np.ndarray,
    target_index: int,
) -> np.ndarray:
    effect = -inverse[:, target_index]
    effect[target_index] = -1.0
    return effect


def main() -> None:
    adata = ad.read_h5ad(H5AD, backed="r")
    rows = []
    sensitivity_rows = []
    downstream_rows = []
    for target, cell_type in TARGET_CELLS.items():
        mask = (adata.obs["cell_type"].astype(str) == cell_type).to_numpy()
        if mask.sum() < 100:
            continue
        sub = adata[mask, :].to_memory()
        selected, genes = choose_network_genes(sub.X, sub.var_names, target)
        matrix = sub.X[:, selected].toarray().astype(float)
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        target_index = genes.index(target)
        eligible_array = np.asarray(
            [i for i in range(len(genes)) if genes[i] not in TARGET_CELLS]
        )
        sampled = RNG.choice(eligible_array, size=N_NULL, replace=True)
        effect = None
        observed = None
        p_value = None
        primary_null_mean = None
        primary_null_sd = None
        for threshold in SENSITIVITY_THRESHOLDS:
            adjacency = np.where(
                np.abs(corr) >= threshold, np.abs(corr), 0.0
            )
            np.fill_diagonal(adjacency, 0.0)
            degree = adjacency.sum(axis=1)
            normalized = adjacency / np.maximum(degree[:, None], 1e-9)
            inverse = np.linalg.inv(
                np.eye(normalized.shape[0]) - ALPHA * normalized
            )
            current_effect = propagate(inverse, target_index)
            current_observed = float(
                np.sum(np.abs(np.delete(current_effect, target_index)))
            )
            null = []
            for index in sampled:
                null_effect = propagate(inverse, int(index))
                null.append(
                    float(
                        np.sum(
                            np.abs(np.delete(null_effect, int(index)))
                        )
                    )
                )
            null = np.asarray(null)
            current_p = (np.sum(null >= current_observed) + 1) / (
                N_NULL + 1
            )
            sensitivity_rows.append(
                {
                    "target": target,
                    "cell_type": cell_type,
                    "threshold": threshold,
                    "n_network_genes": len(genes),
                    "observed_propagated_effect": current_observed,
                    "null_mean": float(null.mean()),
                    "null_sd": float(null.std(ddof=1)),
                    "empirical_p": float(current_p),
                }
            )
            if threshold == THRESHOLD:
                effect = current_effect
                observed = current_observed
                p_value = current_p
                primary_null_mean = float(null.mean())
                primary_null_sd = float(null.std(ddof=1))
        rows.append(
            {
                "target": target,
                "cell_type": cell_type,
                "n_cells": int(mask.sum()),
                "n_network_genes": len(genes),
                "observed_propagated_effect": observed,
                "null_mean": primary_null_mean,
                "null_sd": primary_null_sd,
                "empirical_p": float(p_value),
            }
        )
        order = np.argsort(np.abs(effect))[::-1]
        rank = 0
        for index in order:
            if index == target_index:
                continue
            downstream_rows.append(
                {
                    "target": target,
                    "cell_type": cell_type,
                    "rank": rank + 1,
                    "gene": genes[index],
                    "propagated_effect": effect[index],
                }
            )
            rank += 1
            if rank >= 25:
                break

    results = pd.DataFrame(rows)
    downstream = pd.DataFrame(downstream_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    results.to_csv(OUT / "linear_network_target_results.csv", index=False)
    downstream.to_csv(OUT / "linear_network_top_downstream.csv", index=False)
    sensitivity.to_csv(OUT / "linear_network_sensitivity.csv", index=False)
    (OUT / "linear_network_method.json").write_text(
        json.dumps(
            {
                "method": (
                    "Cell-type-specific absolute Pearson co-expression network "
                    f"among the top {N_HVG} variable genes plus target; "
                    f"edges below |r|={THRESHOLD} removed; row-normalized "
                    f"adjacency; alpha={ALPHA} linear propagation; "
                    f"{N_NULL} random target permutations."
                ),
                "interpretation": (
                    "Exploratory computational sensitivity analysis only. "
                    "No perturbation ground truth or causal inference."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
