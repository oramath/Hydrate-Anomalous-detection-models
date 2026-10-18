#!/usr/bin/env bash
# Reproduce every figure and table of Article 1, in order.
# Usage:  H3W_DATA=/path/to/3W/dataset ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${H3W_DATA:?set H3W_DATA to the 3W 2.0.0 dataset folder (the one holding 0/ ... 9/ and dataset.ini)}"
export H3W_DATA

python3 scripts/s00_build_cache.py            # QC + minute features for 1,119 real instances (about 15 min on 2 cores)
python3 scripts/s10_tables.py                 # Tables 2, 3, 5, 6, 7
for f in fig01_schematic fig02_composition fig03_sensor_quality fig04_event_anatomy_all \
         fig05_phase_diagram fig06_subcooling_distributions fig07_event_physics fig08_cooldown \
         fig09_state_by_class fig10_workflow; do
  python3 "scripts/${f}.py"
done
python3 scripts/s20_lowo.py                   # leave-one-well-out scores, M0, M2, M3 (about 45 min on 2 cores)
python3 scripts/s21_tcn.py                    # M4 temporal CNN (needs torch, about 30 min on CPU)
python3 scripts/s30_metrics.py                # event metrics, Tables 8 and 9, Figs 11 and 12
python3 scripts/s31_robustness.py             # Table 10, Fig 13
echo "done: outputs/figures and outputs/tables"
