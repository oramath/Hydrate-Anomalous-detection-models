# hydrate3w

Physics-informed early warning of gas hydrate plugging in subsea production lines, built on the public
Petrobras **3W Dataset 2.0.0** (CC BY 4.0, [doi:10.6084/m9.figshare.29205836.v1](https://doi.org/10.6084/m9.figshare.29205836.v1)).

This repository regenerates every figure and table of Article 1 from the raw Parquet files.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/get_data.sh data            # about 1.8 GB, prints the export line below
export H3W_DATA=$PWD/data/3W/dataset
./run_all.sh                              # about 1.5 h on 2 CPU cores
```

Outputs land in `outputs/figures` (PNG at 300 dpi and vector PDF) and `outputs/tables` (CSV and Markdown).

## What each script produces

| Script | Output | Manuscript item |
| --- | --- | --- |
| `s00_build_cache.py` | `outputs/cache/minute/*.parquet`, `qc_records.csv`, `instance_index.csv` | QC and minute features (Sec. 2.2) |
| `s10_tables.py` | `tab02_variables`, `tab03_hydrate_instances`, `tab04_correlations`, `tab05_models`, `tabS1_features` | Tables 2 to 5, Table S1 |
| `fig01_schematic.py` | `fig01_schematic` | Fig. 1, production system and sensors |
| `fig02_composition.py` | `fig02_composition` | Fig. 2, dataset composition |
| `fig03_sensor_quality.py` | `fig03_sensor_quality` | Fig. 3, sensor QC of hydrate events |
| `fig04_event_anatomy_all.py` | `fig04_event_anatomy_all` | Fig. 4, label and state anatomy |
| `fig05_phase_diagram.py` | `fig05_phase_diagram` | Fig. 5, hydrate P-T diagram and correlation spread |
| `fig06_subcooling_distributions.py` | `fig06_subcooling_distributions`, `tab_subcooling_by_group` | Fig. 6, subcooling across the dataset |
| `fig07_event_physics.py` | `fig07_event_physics` | Fig. 7, one event in full (pass another instance id as argument) |
| `fig08_cooldown.py` | `fig08_cooldown`, `tab_cooldown_fits` | Fig. 8, Newtonian cool-down and time-to-hydrate |
| `fig09_state_by_class.py` | `fig09_state_by_class`, `tabS2_state_hours_by_class` | Fig. 9 and Table S2, label-leakage evidence |
| `fig10_workflow.py` | `fig10_workflow` | Fig. 10, workflow |
| `s20_lowo.py` | `outputs/cache/oof_scores.parquet` | leave-one-well-out scores, M0, M2, M3 |
| `s21_tcn.py` | adds M4 to `oof_scores.parquet` | temporal CNN (needs torch) |
| `s30_metrics.py` | `tabR1_event_outcomes`, `tabR2_model_summary`, `tab_operating_curves`, `fig11_operating_curves`, `fig12_lead_times` | Results |
| `s31_robustness.py` | `tabR3_robustness`, `fig13_robustness` | Robustness to correlation and gas gravity |

## Package layout

```
hydrate3w/config.py     every constant: paths, seabed temperature, gas-gravity band, QC ranges, dose threshold
hydrate3w/io.py         Parquet loading and per-sensor QC (plausible range, frozen runs)
hydrate3w/thermo.py     hydrate equilibrium correlations, subcooling, exposure dose, Newton cool-down, time-to-hydrate
hydrate3w/features.py   minute-level physics and statistical features (no state labels, no valve states)
hydrate3w/models.py     physics rule, gradient boosting, temporal CNN
hydrate3w/evaluate.py   alarms, lead time, false-alarm rate
hydrate3w/style.py      one colour system: fixed colours per operational state, label phase, correlation and model
```

## Design rules that keep results honest

* Every score comes from a model that never saw the well it scores.
* State labels and the nine valve-state variables never enter a model: in 3W 2.0.0 non-Open states occur almost
  only inside hydrate instances (Fig. 9), so they would leak the answer.
* Simulated hydrate instances are excluded: OLGA modelled hydrate as a linearly closing valve.
* The physics-rule threshold (5 degC*h) is fixed a priori, and the full threshold sweep is reported.

## Verified environment

Python 3.13.16, numpy 2.5.3, pandas 3.0.5, pyarrow 26.0.0, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2,
torch 2.14.1 (CPU). Random seeds are fixed in every model.

## Licence and citation

Code: MIT. Data: 3W Dataset 2.0.0, CC BY 4.0, Petrobras. Cite Vargas et al. (2026), Scientific Data,
doi:10.1038/s41597-026-07225-z, when using the data.
