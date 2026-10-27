from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
RESULT_DIR = ROOT / "results"


HUB_GENES = ["RELA", "ATP6V1B2", "HMOX1", "MITF", "ATP6V1D", "TFE3"]


def collapse_symbols(expression: pd.DataFrame) -> pd.DataFrame:
    expression = expression.copy()
    expression.index = expression.index.astype(str).str.upper()
    valid = (
        pd.notna(expression.index)
        & pd.Series(expression.index, dtype="object").ne("")
        & pd.Series(expression.index, dtype="object").ne("NAN")
    )
    expression = expression.loc[np.asarray(valid, dtype=bool)]
    ranked = expression.assign(__mean=expression.mean(axis=1)).sort_values(
        "__mean",
        ascending=False,
    )
    return ranked.groupby(level=0).head(1).drop(columns="__mean")


def clp_expression() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "GSE229925_anno_genes.FPKM.txt.gz", sep="\t")
    samples = [
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
    ]
    expression = raw.set_index("gene_name")[samples].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    return collapse_symbols(np.log2(expression + 1))


def lps_expression() -> pd.DataFrame:
    raw = pd.read_csv(
        DATA_DIR / "GSE267388_gene_expression.xls.gz",
        sep="\t",
        compression="gzip",
    )
    samples = [column for column in raw.columns if column.startswith("fpkm_")]
    expression = raw.set_index("gene_symbol")[samples].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0)
    return collapse_symbols(np.log2(expression + 1))


def fit_model(expression: pd.DataFrame) -> tuple[object, pd.Series]:
    x = expression.loc[HUB_GENES].T
    y = pd.Series(
        [0] * 4 + [1] * 11,
        index=x.index,
        name="label",
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            solver="liblinear",
            random_state=42,
        ),
    )
    model.fit(x, y)
    return model, y


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(42)
    aucs = []
    briers = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), len(labels))
        if len(set(labels[indices])) < 2:
            continue
        aucs.append(roc_auc_score(labels[indices], probabilities[indices]))
        briers.append(brier_score_loss(labels[indices], probabilities[indices]))
    return {
        "auc_ci_low": float(np.percentile(aucs, 2.5)),
        "auc_ci_high": float(np.percentile(aucs, 97.5)),
        "brier_ci_low": float(np.percentile(briers, 2.5)),
        "brier_ci_high": float(np.percentile(briers, 97.5)),
    }


def main() -> None:
    clp = clp_expression()
    lps = lps_expression()
    model, train_labels = fit_model(clp)
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    model_parameters = {
        "genes": HUB_GENES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": logistic.coef_[0].tolist(),
        "intercept": float(logistic.intercept_[0]),
        "default_threshold": 0.5,
    }
    (RESULT_DIR / "six_gene_model_parameters.json").write_text(
        json.dumps(model_parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    clp_probabilities = model.predict_proba(clp.loc[HUB_GENES].T)[:, 1]
    fpr, tpr, thresholds = roc_curve(train_labels, clp_probabilities)
    youden_index = tpr - fpr
    youden_threshold = float(thresholds[np.argmax(youden_index)])
    sensitivity_threshold = float(
        np.min(thresholds[tpr >= 0.90])
        if np.any(tpr >= 0.90)
        else youden_threshold
    )
    threshold_table = pd.DataFrame(
        [
            {
                "threshold_name": "Default",
                "threshold": 0.50,
                "prespecification": "Fixed before external validation",
            },
            {
                "threshold_name": "Youden",
                "threshold": youden_threshold,
                "prespecification": "Derived in GSE229925 training set",
            },
            {
                "threshold_name": "Sensitivity_90",
                "threshold": sensitivity_threshold,
                "prespecification": "Training-set threshold yielding sensitivity >= 0.90",
            },
        ]
    )
    threshold_table.to_csv(TABLE_DIR / "diagnostic_thresholds.csv", index=False)
    model_parameters["thresholds"] = {
        row["threshold_name"]: float(row["threshold"])
        for _, row in threshold_table.iterrows()
    }
    (RESULT_DIR / "six_gene_model_parameters.json").write_text(
        json.dumps(model_parameters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lps_samples = [column for column in lps.columns if "LPS" in column]
    pbs_samples = [column for column in lps.columns if "PBS" in column]
    external_x = lps.loc[HUB_GENES, lps_samples + pbs_samples].T
    external_labels = np.array([1] * len(lps_samples) + [0] * len(pbs_samples))
    external_probabilities = model.predict_proba(external_x)[:, 1]
    external_auc = roc_auc_score(external_labels, external_probabilities)
    external_brier = brier_score_loss(external_labels, external_probabilities)
    bootstrap = bootstrap_metrics(external_labels, external_probabilities)

    logit_prediction = np.log(
        external_probabilities / (1 - external_probabilities)
    )
    slope_model = LogisticRegression(
        solver="liblinear",
        random_state=42,
    ).fit(logit_prediction.reshape(-1, 1), external_labels)
    calibration_slope = float(slope_model.coef_[0, 0])
    intercept_only = LogisticRegression(
        fit_intercept=True,
        solver="liblinear",
        random_state=42,
    ).fit(np.zeros((len(external_labels), 1)), external_labels)
    observed_rate = float(
        intercept_only.predict_proba(np.zeros((1, 1)))[0, 1]
    )
    mean_predicted = float(external_probabilities.mean())
    calibration_intercept = float(
        np.log(observed_rate / (1 - observed_rate))
        - np.log(mean_predicted / (1 - mean_predicted))
    )

    external_metrics = pd.DataFrame(
        [
            {
                "cohort": "GSE267388 independent external cohort",
                "n": len(external_labels),
                "events": int(external_labels.sum()),
                "auc": external_auc,
                **bootstrap,
                "brier_score": external_brier,
                "calibration_slope": calibration_slope,
                "calibration_intercept": calibration_intercept,
            }
        ]
    )
    external_metrics.to_csv(
        TABLE_DIR / "external_validation_and_calibration_metrics.csv",
        index=False,
    )

    performance_rows = []
    for _, row in threshold_table.iterrows():
        metrics = threshold_metrics(
            external_labels,
            external_probabilities,
            float(row["threshold"]),
        )
        metrics["threshold_name"] = row["threshold_name"]
        performance_rows.append(metrics)
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(
        TABLE_DIR / "external_threshold_performance.csv",
        index=False,
    )

    bins = min(5, max(3, len(external_labels) // 4))
    observed, predicted = calibration_curve(
        external_labels,
        external_probabilities,
        n_bins=bins,
        strategy="quantile",
    )
    calibration = pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_fraction": observed,
        }
    )
    calibration.to_csv(TABLE_DIR / "external_calibration_curve.csv", index=False)

    plt.figure(figsize=(6.4, 5.6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.plot(
        predicted,
        observed,
        marker="o",
        color="#c44e52",
        linewidth=2,
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("External calibration: GSE267388")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "external_calibration_curve.png", dpi=300)
    plt.close()

    sample_coding = pd.DataFrame(
        {
            "dataset": ["GSE229925"] * len(clp.columns) + ["GSE267388"] * len(lps.columns),
            "sample_id": list(clp.columns) + list(lps.columns),
            "group": (
                ["Sham"] * 4
                + ["CLP"] * 11
                + ["LPS"] * len(lps_samples)
                + ["PBS"] * len(pbs_samples)
            ),
            "binary_label": [0] * 4 + [1] * 11 + [1] * len(lps_samples) + [0] * len(pbs_samples),
        }
    )
    sample_coding.to_csv(TABLE_DIR / "diagnostic_sample_coding.csv", index=False)

    gene_coding = pd.DataFrame(
        {
            "gene": HUB_GENES,
            "role": [
                "NF-kB inflammatory regulator",
                "V-ATPase lysosomal acidification",
                "Heme degradation and iron release",
                "Lysosomal biogenesis transcription factor",
                "V-ATPase lysosomal acidification",
                "Lysosomal biogenesis transcription factor",
            ],
            "direction_in_sepsis": ["up", "up", "up", "down", "down", "up"],
        }
    )
    gene_coding.to_csv(TABLE_DIR / "diagnostic_gene_coding.csv", index=False)

    data_dictionary = pd.DataFrame(
        [
            {
                "variable": "sample_id",
                "definition": "Unique sample identifier",
                "coding": "Character",
            },
            {
                "variable": "group",
                "definition": "Experimental group",
                "coding": "Sham/CLP/LPS/PBS for model datasets; Case/Control for external clinical data",
            },
            {
                "variable": "binary_label",
                "definition": "Binary outcome used only for model evaluation",
                "coding": "0 = non-sepsis control, 1 = sepsis or LPS",
            },
            {
                "variable": "gene expression",
                "definition": "Log2-transformed normalized expression",
                "coding": "One column per hub gene; genes must be named RELA, ATP6V1B2, HMOX1, MITF, ATP6V1D and TFE3",
            },
            {
                "variable": "predicted_probability",
                "definition": "Six-gene logistic model probability",
                "coding": "Continuous 0-1",
            },
        ]
    )
    data_dictionary.to_csv(
        TABLE_DIR / "diagnostic_data_dictionary.csv",
        index=False,
    )

    summary = {
        "training_cohort": "GSE229925",
        "external_cohort": "GSE267388",
        "clinical_direction_validation": "GSE79962",
        "clinical_sample_level_matrix_available": False,
        "thresholds": threshold_table.to_dict(orient="records"),
        "external_metrics": external_metrics.to_dict(orient="records"),
        "external_threshold_performance": performance.to_dict(orient="records"),
    }
    (RESULT_DIR / "external_validation_calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
