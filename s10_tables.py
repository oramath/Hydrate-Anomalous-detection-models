"""Step 1. Methods tables (CSV and Markdown).

  tab02_variables.csv            27 variables, meaning, unit, location, model role, validity in hydrate events
  tab03_hydrate_instances.csv    the 14 real production-line hydrate instances
  tab04_correlations.csv         hydrate equilibrium correlations used (Table 4)
  tabS1_features.csv             model features with physical meaning
  tab05_models.csv               model configurations and validation protocol (Table 5)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import thermo  # noqa: E402
from hydrate3w.config import (ALARM_PERSIST_MIN, DOSE_RESET_MIN, DOSE_THRESHOLD, GAMMA_REF,  # noqa: E402
                              PA_PER_BAR, SHUTIN_STATES, TAB, VARIABLES)
from hydrate3w.data import minute, qc, real, short  # noqa: E402


def md(df: pd.DataFrame, name: str):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False))


q = qc()
h8 = set(real(8).instance)
rows = []
for v, (desc, unit, qty, loc, role) in VARIABLES.items():
    s = q[(q.variable == v) & q.instance.isin(h8)]
    rows.append(dict(variable=v, description=desc, unit=unit, location=loc, role=role,
                     valid_in_class8=f"{(s.status == 'valid').sum()}/{len(s)}"))
md(pd.DataFrame(rows), "tab02_variables")

rows = []
for i in real(8).sort_values(["well", "start"]).instance:
    m = minute(i)
    c = m["class"]
    qi = q[q.instance == i].set_index("variable").status
    t_tr = c.index[c.eq(108)][0] if c.eq(108).any() else None
    t_pl = c.index[c.eq(8)][0] if c.eq(8).any() else None
    si = m.state.isin(SHUTIN_STATES)
    rows.append({
        "instance": short(i), "hours": round(len(m) / 60, 1),
        "normal_h": round(c.eq(0).sum() / 60, 1), "transient_h": round(c.eq(108).sum() / 60, 1),
        "plug_h": round(c.eq(8).sum() / 60, 1), "shut_in_h": round(si.sum() / 60, 1),
        "transient_to_plug_h": round((t_pl - t_tr).total_seconds() / 3600, 1) if (t_tr is not None and t_pl is not None) else np.nan,
        "state_at_plug": m.loc[t_pl, "state_name"] if t_pl is not None else "no plug label",
        "tree_P": qi.get("P-TPT"), "tree_T": qi.get("T-TPT"),
        "max_P_line_bar": round(m.P_line.max() / PA_PER_BAR, 0) if m.P_line.notna().any() else np.nan,
        "median_shutin_subcooling_C": round(m.loc[si, "sub"].median(), 1) if m.loc[si, "sub"].notna().any() else np.nan,
    })
md(pd.DataFrame(rows), "tab03_hydrate_instances")

P_ref = np.array([50, 100, 200]) * PA_PER_BAR
rows = [
    dict(correlation="Towler-Mokhatab (2005)", form="T[°F] = 13.47 ln P + 34.27 ln γ − 1.675 ln P ln γ − 20.35",
         inputs="P [psia], γ", role="central case"),
    dict(correlation="Motiee (1991)", form="T[°F] = −238.24 + 78.997 log P − 5.3525 (log P)² + 349.47 γ − 150.85 γ² − 27.604 γ log P",
         inputs="P [psia], γ", role="sensitivity"),
    dict(correlation="Hammerschmidt (1934)", form="T[°F] = 8.9 P^0.285", inputs="P [psia]", role="sensitivity, no γ"),
]
for r, name in zip(rows, thermo.CORRELATIONS):
    for p in P_ref:
        r[f"T_eq at {p / PA_PER_BAR:.0f} bar, γ=0.7 (°C)"] = round(float(thermo.teq(p, GAMMA_REF, name)), 1)
md(pd.DataFrame(rows), "tab04_correlations")

feat = [
    ("P_line", "bar", "line inlet pressure, first valid of P-TPT, PT-P, P-MON-SDV-P, P-MON-CKP", "thermodynamic state"),
    ("T_line", "°C", "line inlet temperature, T-TPT", "thermodynamic state"),
    ("T_eq", "°C", f"hydrate equilibrium temperature at P_line, Towler-Mokhatab, γ = {GAMMA_REF}", "phase boundary"),
    ("sub", "°C", "subcooling ΔT_sub = T_eq − T_line", "driving force for nucleation"),
    ("dose", "°C·h", f"∫ max(ΔT_sub, 0) dt, reset after {DOSE_RESET_MIN} min outside the zone", "cumulative hydrate exposure"),
    ("dT_dt", "°C/h", "30 min least-squares slope of T_line", "cool-down rate"),
    ("t_hyd_h", "h", "τ ln[(T − T_sea)/(T_eq − T_sea)], τ = −(T − T_sea)/(dT/dt)", "forecast time to hydrate entry"),
    ("dP_line", "bar", "P-TPT − P-MON-CKP", "line hydraulic resistance plus hydrostatic head"),
    ("dP_line_anom", "bar", "dP_line minus its trailing 6 h median", "restriction growth"),
    ("dP_tubing", "bar", "P-PDG − P-TPT", "tubing flow and inflow context"),
    ("raw x 4 statistics", "various", "12 process signals: minute mean, 10 min std, 30 min slope, 60 min difference",
     "data-driven baseline"),
]
md(pd.DataFrame(feat, columns=["feature", "unit", "definition", "physical meaning"]), "tabS1_features")

models = [
    ("Physics rule", "dose ≥ threshold", f"threshold {DOSE_THRESHOLD:g} °C·h, set a priori", "none"),
    ("GBM raw", "histogram gradient boosting on 48 raw-signal features", "250 trees, lr 0.05, 31 leaves, balanced weights", "LOWO, 40 wells"),
    ("GBM physics", "same model on raw plus 8 physics features", "as above", "LOWO, 40 wells"),
    ("TCN physics", "causal dilated CNN, 5 levels, 32 channels, 125 min receptive field", "AdamW lr 2e-3, 4 epochs, 6 h chunks", "well held out: 9 hydrate-well folds + 5 grouped folds"),
    ("State shortcut (audit)", "gradient boosting on one-hot operational state labels only", "as GBM", "LOWO, 40 wells"),
]
proto = pd.DataFrame(models, columns=["model", "structure", "settings", "validation"])
proto["alarm"] = f"score above threshold for {ALARM_PERSIST_MIN} consecutive minutes"
md(proto, "tab05_models")
print("tables written to", TAB)
