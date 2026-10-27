# Input data

The original data are public and are not redistributed in this repository. Download
them from NCBI GEO and place the files in this directory using the names below.

## Required bulk and tabular files

| Dataset | Expected local file |
|---|---|
| GSE229925 | `GSE229925_anno_genes.FPKM.txt.gz` |
| GSE267388 | `GSE267388_gene_expression.xls.gz` |
| GSE79962 | `GSE79962_original_supplement.xlsx` |
| GSE294985 | `GSE294985_Processed_data_FPKM.xlsx` |
| GSE207177 | `GSE207177_processed.h5ad` |

## Single-cell validation data

For GSE301781, place the 12 filtered feature-barcode HDF5 files under:

```text
data/GSE301781/
```

File names must begin with the corresponding GSM accession, for example:

```text
GSM9089582_168574_filtered_feature_bc_matrix.h5
```

The manuscript defines six control and six CLP six-week samples in
`analysis/Reinstated_Analyses/independent_validation_gse301781.py`.

## Optional derived source data

The supplementary analysis that loads `top_candidate_genes.csv` expects:

```text
source_data/top_candidate_genes.csv
```

Place the file from the journal source-data package there, or set
`SEPSIS_SOURCE_DATA_DIR` to the directory containing it.

## Data-use boundary

The GEO source datasets remain subject to their original access and use conditions.
Do not upload controlled-access or third-party raw data to this repository.
