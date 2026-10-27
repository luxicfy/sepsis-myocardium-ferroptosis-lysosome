from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_DATA = Path(
    os.environ.get("SEPSIS_SOURCE_DATA_DIR", ROOT / "source_data")
).expanduser().resolve()
DATA = Path(
    os.environ.get("SEPSIS_DATA_DIR", ROOT / "data")
).expanduser().resolve()
H5AD = Path(
    os.environ.get("GSE207177_H5AD", DATA / "GSE207177_processed.h5ad")
).expanduser().resolve()
RNG = np.random.default_rng(20261018)
HUB_GENES = ["RELA", "ATP6V1B2", "HMOX1", "MITF", "ATP6V1D", "TFE3"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation = load_module(
    "sepsis_validation",
    ROOT / "analysis" / "05_external_validation_calibration.py",
)
human = load_module(
    "sepsis_human",
    ROOT / "analysis" / "06_external_human_validation.py",
)
validation.DATA_DIR = DATA
human.DATA_DIR = DATA


def loocv_probabilities(x: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, solver="liblinear", random_state=42),
    )
    probabilities = cross_val_predict(
        model,
        x,
        y,
        cv=LeaveOneOut(),
        method="predict_proba",
    )[:, 1]
    return probabilities


def model_checks() -> None:
    clp = validation.clp_expression()
    lps = validation.lps_expression()
    model, train_labels = validation.fit_model(clp)
    x = clp.loc[HUB_GENES].T
    y = train_labels.to_numpy()
    clp_probabilities = loocv_probabilities(x, y)
    observed_clp = roc_auc_score(y, clp_probabilities)
    if observed_clp < 0.5:
        observed_clp = 1 - observed_clp

    n_perm = 5000
    null_clp = np.empty(n_perm)
    for i in range(n_perm):
        yp = RNG.permutation(y)
        null_clp[i] = roc_auc_score(yp, clp_probabilities)
    p_clp = (np.sum(null_clp >= observed_clp) + 1) / (n_perm + 1)

    lps_samples = [c for c in lps.columns if "LPS" in c]
    pbs_samples = [c for c in lps.columns if "PBS" in c]
    external_x = lps.loc[HUB_GENES, lps_samples + pbs_samples].T
    external_labels = np.array([1] * len(lps_samples) + [0] * len(pbs_samples))
    external_probabilities = model.predict_proba(external_x)[:, 1]
    observed_lps = roc_auc_score(external_labels, external_probabilities)
    null_lps = np.empty(n_perm)
    for i in range(n_perm):
        yp = RNG.permutation(external_labels)
        null_lps[i] = roc_auc_score(yp, external_probabilities)
    p_lps = (np.sum(null_lps >= observed_lps) + 1) / (n_perm + 1)

    thresholds = np.linspace(0.05, 0.95, 91)
    n = len(external_labels)
    prevalence = external_labels.mean()
    decision_rows = []
    for threshold in thresholds:
        prediction = external_probabilities >= threshold
        tp = np.sum(prediction & (external_labels == 1))
        fp = np.sum(prediction & (external_labels == 0))
        net_benefit = tp / n - fp / n * threshold / (1 - threshold)
        treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        decision_rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": net_benefit,
                "net_benefit_treat_all": treat_all,
                "net_benefit_treat_none": 0.0,
            }
        )
    decision = pd.DataFrame(decision_rows)
    decision.to_csv(OUT / "decision_curve_GSE267388.csv", index=False)

    human_raw = pd.read_excel(
        DATA / "GSE294985_Processed_data_FPKM.xlsx"
    ).set_index("gene_id")
    ensembl = human.HUB_GENES
    expression = human_raw.loc[list(ensembl.values())].copy()
    expression.index = list(ensembl)
    cases = [c for c in expression.columns if str(c).endswith("-Seq")]
    controls = [c for c in expression.columns if str(c).startswith("ctrl")]
    z = expression.sub(expression.mean(axis=1), axis=0).div(
        expression.std(axis=1, ddof=1).replace(0, np.nan), axis=0
    )
    directions = np.array([human.DIRECTIONS[g] for g in expression.index])
    score = np.nanmean(z.to_numpy().T * directions, axis=1)
    score = pd.Series(score, index=expression.columns)
    labels = np.array([1] * len(cases) + [0] * len(controls))
    observed_human = roc_auc_score(labels, score[cases + controls])
    null_human = np.empty(n_perm)
    for i in range(n_perm):
        yp = RNG.permutation(labels)
        null_human[i] = roc_auc_score(yp, score[cases + controls])
    p_human_one_sided = (np.sum(null_human <= observed_human) + 1) / (n_perm + 1)
    p_human_two_sided = (
        np.sum(np.abs(null_human - 0.5) >= abs(observed_human - 0.5)) + 1
    ) / (n_perm + 1)

    summary = {
        "n_permutations": n_perm,
        "permutation_method": (
            "Labels were permuted against fixed out-of-fold predictions. "
            "This tests score-label association without refitting the model "
            "in every permutation."
        ),
        "clp_loocv_auc": observed_clp,
        "clp_permutation_p_one_sided": float(p_clp),
        "lps_external_auc": float(observed_lps),
        "lps_permutation_p_one_sided": float(p_lps),
        "gse294985_auc": float(observed_human),
        "gse294985_permutation_p_opposite_direction": float(p_human_one_sided),
        "gse294985_permutation_p_two_sided": float(p_human_two_sided),
    }
    (OUT / "permutation_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"clp_null_loocv_auc": null_clp}).to_csv(
        OUT / "clp_permutation_null.csv", index=False
    )
    pd.DataFrame({"lps_null_auc": null_lps}).to_csv(
        OUT / "lps_permutation_null.csv", index=False
    )
    pd.DataFrame({"gse294985_null_auc": null_human}).to_csv(
        OUT / "gse294985_permutation_null.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=300)
    axes[0].plot(
        decision["threshold"],
        decision["net_benefit_model"],
        label="Six-gene model",
        color="#2F5D6B",
    )
    axes[0].plot(
        decision["threshold"],
        decision["net_benefit_treat_all"],
        label="Treat all",
        color="#A65A3A",
        linestyle="--",
    )
    axes[0].axhline(0, color="#777777", linestyle=":", label="Treat none")
    axes[0].set_xlabel("Threshold probability")
    axes[0].set_ylabel("Net benefit")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].hist(null_clp, bins=35, color="#8A9AA0", alpha=0.8)
    axes[1].axvline(
        observed_clp, color="#A13D3D", linewidth=1.5, label="Observed"
    )
    axes[1].set_xlabel("Permuted LOOCV AUC")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_model_permutation_decision_curve.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


def pseudobulk_mean(x) -> np.ndarray:
    if sp.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x.mean(axis=0)).ravel()


def cell_state_scores() -> None:
    adata = ad.read_h5ad(H5AD, backed="r")
    directions = pd.read_csv(
        SOURCE_DATA / "top_candidate_genes.csv", index_col=0
    )
    signs = np.array([1 if directions.loc[g, "clp_effect"] > 0 else -1 for g in HUB_GENES])
    gene_indices = [list(adata.var_names).index(g) for g in HUB_GENES]
    score_rows = []
    summary_rows = []
    for cell_type in sorted(adata.obs["cell_type"].astype(str).unique()):
        mask = (adata.obs["cell_type"].astype(str) == cell_type).to_numpy()
        if mask.sum() < 150:
            continue
        sub = adata[mask, :]
        obs = sub.obs.copy()
        samples = sorted(obs["sample"].astype(str).unique())
        if len(samples) < 6:
            continue
        rows = {}
        all_rows = {}
        for sample in samples:
            sample_mask = (obs["sample"].astype(str) == sample).to_numpy()
            selected = sub.X[sample_mask, :]
            rows[sample] = pseudobulk_mean(selected[:, gene_indices])
            all_rows[sample] = pseudobulk_mean(selected)
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=HUB_GENES)
        all_frame = pd.DataFrame.from_dict(
            all_rows, orient="index", columns=[str(g) for g in sub.var_names]
        )
        groups = obs[["sample", "group"]].astype(str).drop_duplicates().set_index("sample")
        frame["group"] = groups.reindex(frame.index)["group"]
        z = frame[HUB_GENES].sub(frame[HUB_GENES].mean(axis=0), axis=1).div(
            frame[HUB_GENES].std(axis=0, ddof=0).replace(0, np.nan), axis=1
        )
        score = np.nanmean(z.to_numpy() * signs, axis=1)
        frame["candidate_score"] = score
        for sample, row in frame.iterrows():
            score_rows.append(
                {
                    "cell_type": cell_type,
                    "sample": sample,
                    "group": row["group"],
                    "candidate_score": row["candidate_score"],
                }
            )
        group_means = frame.groupby("group")["candidate_score"].mean()
        if not {"Control", "CLP_12h", "CLP_24h"}.issubset(group_means.index):
            continue
        magnitude = np.mean(
            [
                abs(group_means["CLP_12h"] - group_means["Control"]),
                abs(group_means["CLP_24h"] - group_means["Control"]),
            ]
        )
        all_mean = all_frame.mean(axis=0)
        all_sd = all_frame.std(axis=0, ddof=0).replace(0, np.nan)
        z_all = all_frame.sub(all_mean, axis=1).div(all_sd, axis=1)
        eligible = [
            g for g in z_all.columns if np.isfinite(z_all[g]).all()
        ]
        bins = pd.qcut(
            all_mean.reindex(eligible).rank(method="first"),
            q=10,
            labels=False,
            duplicates="drop",
        )
        bins = pd.Series(bins.to_numpy(), index=eligible)
        target_bins = [bins.loc[g] for g in HUB_GENES]
        pool_by_bin = {
            b: bins[bins == b].index.to_numpy()
            for b in sorted(set(target_bins))
        }
        null_magnitudes = np.empty(2000)
        for i in range(2000):
            sampled = [
                RNG.choice(pool_by_bin[b]) for b in target_bins
            ]
            random_score = np.nanmean(
                z_all[sampled].to_numpy() * signs, axis=1
            )
            random_series = pd.Series(random_score, index=z_all.index)
            random_group = random_series.groupby(frame["group"]).mean()
            null_magnitudes[i] = np.mean(
                [
                    abs(
                        random_group["CLP_12h"]
                        - random_group["Control"]
                    ),
                    abs(
                        random_group["CLP_24h"]
                        - random_group["Control"]
                    ),
                ]
            )
        empirical_p = (
            np.sum(null_magnitudes >= magnitude) + 1
        ) / (len(null_magnitudes) + 1)
        summary_rows.append(
            {
                "cell_type": cell_type,
                "control_mean": group_means["Control"],
                "clp12_mean": group_means["CLP_12h"],
                "clp24_mean": group_means["CLP_24h"],
                "perturbation_magnitude": magnitude,
                "n_cells": int(mask.sum()),
                "matched_null_mean": float(null_magnitudes.mean()),
                "matched_null_sd": float(null_magnitudes.std(ddof=1)),
                "matched_gene_set_p": float(empirical_p),
            }
        )
    pd.DataFrame(score_rows).to_csv(
        OUT / "cell_state_candidate_scores.csv", index=False
    )
    summary = pd.DataFrame(summary_rows).sort_values(
        "perturbation_magnitude", ascending=False
    )
    summary.to_csv(OUT / "cell_state_perturbation_magnitude.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=300)
    plot = summary.sort_values("perturbation_magnitude")
    ax.barh(
        np.arange(len(plot)),
        plot["perturbation_magnitude"],
        color="#4F6D7A",
        height=0.65,
    )
    ax.set_yticks(np.arange(len(plot)), plot["cell_type"], fontsize=7)
    ax.set_xlabel("Mean absolute disease-control candidate-score difference")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure_cell_state_perturbation.png", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    model_checks()
    cell_state_scores()
