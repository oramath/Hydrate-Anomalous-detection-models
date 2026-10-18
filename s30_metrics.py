"""Step 3. Event-level metrics from out-of-fold scores, operating curves, and the results tables and figures.

Writes
  outputs/tables/tabR1_event_outcomes.csv   per hydrate event and model at the operating point
  outputs/tables/tabR2_model_summary.csv    detection, lead time and false alarms per model
  outputs/figures/fig11_operating_curves    event recall vs false alarms per 1,000 h
  outputs/figures/fig12_lead_times          lead time per event and model
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402
from hydrate3w.config import ALARM_PERSIST_MIN, CACHE, DOSE_THRESHOLD, TAB  # noqa: E402
from hydrate3w.data import short  # noqa: E402
from hydrate3w.evaluate import alarm_onsets, event_outcome, false_alarm_rate  # noqa: E402

style.apply()
S = style
oof = pd.read_parquet(CACHE / "oof_scores.parquet")
MODELS = [m for m in ("Physics rule", "GBM raw", "GBM physics", "TCN physics", "State shortcut") if m in oof]
OPERATING = {"Physics rule": DOSE_THRESHOLD}
GRID = {"Physics rule": [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]}
groups = {i: g.set_index("t") for i, g in oof.groupby("instance")}
hyd = [i for i in groups if i.startswith("8/")]


def negative_mask(g: pd.DataFrame) -> pd.Series:
    c = g["class"]
    neg = c.notna() & ~c.isin([108, 8])
    if c.isin([108]).any():
        neg &= g.index < c.index[c.eq(108)][0]
    return neg


NEG = {i: negative_mask(g) for i, g in groups.items()}
NEG_HOURS = sum(float(n.sum()) for n in NEG.values()) / 60


def evaluate(model: str, thr: float):
    rows = []
    fa = 0
    for i, g in groups.items():
        flag = g[model] >= thr
        on = alarm_onsets(flag, ALARM_PERSIST_MIN)
        fa += int(NEG[i].reindex(on).fillna(False).sum())
        if i in hyd:
            r = event_outcome(g, flag)
            r.update(instance=i, model=model, threshold=thr)
            rows.append(r)
    ev = pd.DataFrame(rows)
    return ev, false_alarm_rate(fa, NEG_HOURS), fa


def main():
    curves, ops = [], []
    for model in MODELS:
        grid = GRID.get(model, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
        for thr in grid:
            ev, far, n_fa = evaluate(model, thr)
            curves.append(dict(model=model, threshold=thr, recall=ev.detected.mean(), far_per_1000h=far,
                               false_alarms=n_fa, median_lead_h=ev.lead_to_plug_h.median()))
        ev, far, n_fa = evaluate(model, OPERATING.get(model, 0.5))
        ev["far_per_1000h"] = far
        ops.append(ev)
    curves = pd.DataFrame(curves)
    ops = pd.concat(ops)
    curves.to_csv(TAB / "tab_operating_curves.csv", index=False)
    ops.to_csv(TAB / "tabR1_event_outcomes.csv", index=False)
    summ = ops.groupby("model").agg(events=("instance", "size"), detected=("detected", "sum"),
                                    median_lead_h=("lead_to_plug_h", "median"),
                                    min_lead_h=("lead_to_plug_h", "min"),
                                    alarm_after_transient_h=("after_transient_h", "median"),
                                    far_per_1000h=("far_per_1000h", "first")).reindex(MODELS)
    summ["negative_hours"] = NEG_HOURS
    summ.to_csv(TAB / "tabR2_model_summary.csv")
    print(summ.round(2).to_string())

    # figure 11: operating curves
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for model in MODELS:
        c = curves[curves.model == model].sort_values("threshold")
        col = S.MODEL_COLORS[model]
        ax.plot(c.far_per_1000h + 0.01, c.recall, marker="o", color=col, lw=2.2, ms=6,
                ls="--" if model == "State shortcut" else "-", label=model)
        op = ops[ops.model == model]
        ax.scatter([op.far_per_1000h.iloc[0] + 0.01], [op.detected.mean()], s=180, facecolor="none",
                   edgecolor=col, lw=2.2, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("false alarms per 1,000 h of negative operation (log scale, +0.01)")
    ax.set_ylabel(f"share of the {len(hyd)} hydrate events warned before the plug label")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right")
    ax.set_title("Event recall against false-alarm rate, leave-one-well-out (circles: operating points)",
                 loc="left", fontsize=10.5)
    print(S.save(fig, "fig11_operating_curves"))

    # figure 12: lead time per event
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    inst = sorted(hyd)
    w = 0.8 / len(MODELS)
    for k, model in enumerate(MODELS):
        op = ops[ops.model == model].set_index("instance").reindex(inst)
        x = np.arange(len(inst)) + (k - (len(MODELS) - 1) / 2) * w
        lead = op.lead_to_plug_h.to_numpy()
        ax.bar(x, np.nan_to_num(lead), width=w, color=S.MODEL_COLORS[model], label=model, edgecolor="white")
        miss = ~op.detected.to_numpy(dtype=bool)
        ax.scatter(x[miss], np.full(miss.sum(), 2), marker="x", color=S.MODEL_COLORS[model], s=40, zorder=4)
    ax.set_xticks(range(len(inst)))
    ax.set_xticklabels([short(i) for i in inst], rotation=35, ha="right")
    ax.set_ylabel("warning lead time to first plug label (h)")
    ax.legend(ncol=len(MODELS), loc="upper left", fontsize=8)
    ax.set_title("Warning lead time per real hydrate event (x: no warning before the plug label)", loc="left")
    print(S.save(fig, "fig12_lead_times"))


if __name__ == "__main__":
    main()
