# Cell-source-attributed ferroptosis-lysosomal program in sepsis-associated myocardial injury

Analysis code for the manuscript:

> Cell-source-attributed candidate ferroptosis-lysosomal transcriptional program in
> sepsis-associated myocardial injury: a hypothesis-generating integrative multi-omic
> and single-cell analysis

Authors: Yilin Lu, Liang Liu, Yaqing An, Na Meng, Lin Yuan, Yu Gong, Jingyan Li,
Dongqi Yao, Hengbo Gao, Yingping Tian.

Corresponding author: Yingping Tian, The Second Hospital of Hebei Medical University,
Shijiazhuang, Hebei 050000, China. Email: tianyingping-jzh@163.com.

## Scope

This repository contains the analysis scripts used to generate the derived tables and
figures reported in the manuscript. It supports an exploratory, hypothesis-generating
secondary analysis of public data. It does not contain primary patient data, protected
health information, or restricted-access records.

The six candidate genes examined in the study are `RELA`, `ATP6V1B2`, `HMOX1`, `MITF`,
`ATP6V1D`, and `TFE3`. They are not presented as a validated diagnostic panel or as
evidence of a causal mechanism.

## Repository layout

```text
analysis/                       Main analysis scripts
analysis/Reinstated_Analyses/   Supplementary analyses
data/                           Data placement instructions; public data are not redistributed
docs/                           Reproducibility and submission notes
requirements.txt                Python package versions
environment.yml                 Conda environment definition
CITATION.cff                    Citation metadata
```

## Data

The study uses public data from GEO accessions `GSE229925`, `GSE267388`, `GSE79962`,
`GSE294985`, `GSE207177`, and `GSE301781`. Raw and processed source files should be
downloaded from GEO and placed under `data/` as described in
[`data/README.md`](data/README.md). The repository intentionally does not redistribute
the original GEO files.

Derived source tables accompanying the article are provided as journal supplementary
source data.

## Environment

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

An equivalent Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate sepsis-myocardium-ferroptosis-lysosome
```

## Reproducible run order

Place the input files under `data/`, then run from the repository root:

```bash
python analysis/01_bulk_analysis.py
python analysis/02_single_cell_analysis.py
python analysis/04_publication_analysis.py
python analysis/05_external_validation_calibration.py
python analysis/06_external_human_validation.py
python analysis/Reinstated_Analyses/reproduce_perturbation_and_model_checks.py
python analysis/Reinstated_Analyses/module_and_pseudobulk.py
python analysis/Reinstated_Analyses/ligand_receptor.py
python analysis/Reinstated_Analyses/linear_network_virtual_perturbation.py
python analysis/Reinstated_Analyses/gse301781_modules.py
python analysis/Reinstated_Analyses/independent_validation_gse301781.py
python analysis/Reinstated_Analyses/make_gse301781_figure.py
```

Scripts write generated files to `results/` or
`analysis/reinstated_analyses/outputs/`. These generated files are excluded from
version control.

## Path configuration

The default data directory is `data/` inside this repository. Alternative locations
can be supplied with environment variables:

```bash
export SEPSIS_DATA_DIR=/path/to/sepsis/data
export SEPSIS_SOURCE_DATA_DIR=/path/to/derived/source/data
export GSE207177_H5AD=/path/to/GSE207177_processed.h5ad
export GSE301781_DATA_DIR=/path/to/GSE301781
```

On Windows PowerShell, use `$env:SEPSIS_DATA_DIR = "..."` and corresponding
assignments for the other variables.

## Important analytical boundaries

- Machine-learning results are exploratory and are not clinical validation.
- Cell-level single-cell P values are not treated as independent biological replication.
- GSE294985 did not reproduce the fixed six-gene direction.
- GSE301781 is an independent cardiac CD45+ validation resource, not a whole-heart
  myocardial tissue validation cohort.
- The linear-network virtual perturbation is a sensitivity analysis, not a CRISPR or
  experimental perturbation.
- No MIMIC-IV or eICU-CRD data were used.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the
manuscript and the archived repository release when available.

## License

Analysis code is released under the MIT License. The license does not grant rights to
redistribute third-party GEO data, which remain subject to their original access and
use terms.

## Contact

Yilin Lu: lyl924x@163.com  
Yingping Tian: tianyingping-jzh@163.com
