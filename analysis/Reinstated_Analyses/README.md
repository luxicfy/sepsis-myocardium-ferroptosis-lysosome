# Reinstated supplementary analyses

Run these scripts after the main analysis and after placing the required files under
`data/` and `source_data/`.

```bash
python analysis/Reinstated_Analyses/reproduce_perturbation_and_model_checks.py
python analysis/Reinstated_Analyses/module_and_pseudobulk.py
python analysis/Reinstated_Analyses/ligand_receptor.py
python analysis/Reinstated_Analyses/linear_network_virtual_perturbation.py
python analysis/Reinstated_Analyses/gse301781_modules.py
python analysis/Reinstated_Analyses/independent_validation_gse301781.py
python analysis/Reinstated_Analyses/make_gse301781_figure.py
```

The linear-network analysis is an exploratory sensitivity analysis and must not be
described as experimental perturbation.
