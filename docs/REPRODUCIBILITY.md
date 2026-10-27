# Reproducibility

## Software

- Python 3.12
- Package versions in `requirements.txt`
- CPU execution is sufficient; the single-cell steps require substantial memory

## Input layout

The default layout is:

```text
data/
  GSE229925_anno_genes.FPKM.txt.gz
  GSE267388_gene_expression.xls.gz
  GSE79962_original_supplement.xlsx
  GSE294985_Processed_data_FPKM.xlsx
  GSE207177_processed.h5ad
  GSE301781/
    GSM*.h5
source_data/
  top_candidate_genes.csv
```

Alternative paths can be set through `SEPSIS_DATA_DIR`,
`SEPSIS_SOURCE_DATA_DIR`, `GSE207177_H5AD`, and `GSE301781_DATA_DIR`.

## Main sequence

1. Bulk differential-expression and module analyses.
2. Single-cell clustering, annotation, module scores, and candidate localization.
3. Publication-level candidate and cross-model summaries.
4. External validation and exploratory calibration analyses.
5. Human whole-blood validation.
6. Permutation, decision-curve, pseudobulk, ligand-receptor, and network analyses.
7. GSE301781 independent cardiac CD45+ validation.

## Random seeds

- PCA: 42
- Leiden clustering: 42
- Module and marker scoring: 42
- Logistic regression: 42
- Permutation analyses: 20261018 or 20261019, as specified in each script

## Outputs

Main outputs are written to `results/tables`, `results/figures`, and
`results/summary`. Supplementary analysis outputs are written to
`analysis/reinstated_analyses/outputs`.

## Verification

Before submission, record:

- the repository commit hash;
- the public release tag;
- the exact command used for each script;
- the SHA-256 checksum of each downloaded input file;
- the Python and package versions used for the final run.
