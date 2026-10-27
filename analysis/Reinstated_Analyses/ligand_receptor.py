from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "reinstated_analyses" / "outputs"
PB_DIR = OUT / "pseudobulk"
DATA = Path(
    os.environ.get("SEPSIS_DATA_DIR", ROOT / "data")
).expanduser().resolve()


def sample_group(sample: str) -> str:
    lower = sample.lower()
    if "_n-" in lower:
        return "Control"
    if "clp-12h" in lower:
        return "CLP_12h"
    if "clp-24h" in lower:
        return "CLP_24h"
    return "Unknown"


def main() -> None:
    response = requests.get(
        "https://omnipathdb.org/interactions",
        params={
            "datasets": "ligrecextra",
            "format": "tsv",
            "fields": "genesymbols",
        },
        timeout=90,
    )
    response.raise_for_status()
    omnipath_path = DATA / "omnipath_ligrecextra.tsv"
    omnipath_path.write_bytes(response.content)
    pairs = pd.read_csv(io.BytesIO(response.content), sep="\t")
    pairs = pairs[
        ["source_genesymbol", "target_genesymbol", "is_directed"]
    ].dropna()
    pairs = pairs[pairs["is_directed"].astype(str).str.lower().eq("true")]
    pairs["ligand"] = pairs["source_genesymbol"].astype(str).str.upper()
    pairs["receptor"] = pairs["target_genesymbol"].astype(str).str.upper()
    pairs = pairs[["ligand", "receptor"]].drop_duplicates()

    frames = {}
    for path in sorted(PB_DIR.glob("*.csv")):
        cell_type = path.stem
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str)
        frames[cell_type] = frame

    senders = [
        "Macrophage",
        "Monocyte",
        "Fibroblast",
        "Endothelial",
        "Pericyte_SMC",
    ]
    receivers = [
        "Cardiomyocyte",
        "Fibroblast",
        "Endothelial",
        "Macrophage",
    ]
    rows = []
    for sender in senders:
        if sender not in frames:
            continue
        sender_frame = frames[sender]
        for receiver in receivers:
            if receiver not in frames or receiver == sender:
                continue
            receiver_frame = frames[receiver]
            common_samples = sender_frame.index.intersection(receiver_frame.index)
            groups = pd.Series(
                [sample_group(s) for s in common_samples],
                index=common_samples,
            )
            ligand_present = [
                g for g in pairs["ligand"].unique() if g in sender_frame.columns
            ]
            receptor_present = [
                g for g in pairs["receptor"].unique() if g in receiver_frame.columns
            ]
            subset = pairs[
                pairs["ligand"].isin(ligand_present)
                & pairs["receptor"].isin(receptor_present)
            ]
            for row in subset.itertuples(index=False):
                ligand_values = sender_frame.loc[common_samples, row.ligand].to_numpy(float)
                receptor_values = receiver_frame.loc[common_samples, row.receptor].to_numpy(float)
                score = np.sqrt(
                    np.maximum(ligand_values, 0)
                    * np.maximum(receptor_values, 0)
                )
                series = pd.Series(score, index=common_samples)
                means = series.groupby(groups).mean()
                if not {"Control", "CLP_12h", "CLP_24h"}.issubset(means.index):
                    continue
                effect12 = means["CLP_12h"] - means["Control"]
                effect24 = means["CLP_24h"] - means["Control"]
                if max(means.max(), 0) < 0.05:
                    continue
                rows.append(
                    {
                        "sender": sender,
                        "receiver": receiver,
                        "ligand": row.ligand,
                        "receptor": row.receptor,
                        "control_score": means["Control"],
                        "clp12_score": means["CLP_12h"],
                        "clp24_score": means["CLP_24h"],
                        "effect_clp12_vs_control": effect12,
                        "effect_clp24_vs_control": effect24,
                        "mean_effect": (effect12 + effect24) / 2,
                        "consistent_increase": effect12 > 0 and effect24 > 0,
                    }
                )
    results = pd.DataFrame(rows).sort_values(
        ["consistent_increase", "mean_effect"],
        ascending=[False, False],
    )
    results.to_csv(OUT / "ligand_receptor_interactions.csv", index=False)
    results.head(200).to_csv(
        OUT / "ligand_receptor_top200.csv", index=False
    )
    print(results.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
