"""Step 3b. Robustness of the physics rule to the equilibrium correlation and gas gravity.

Recomputes the exposure dose for all nine correlation x gravity combinations on every
real instance and evaluates the dose rule at its fixed threshold. Writes
outputs/tables/tabR3_robustness.csv and outputs/figures/fig13_robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style, thermo  # noqa: E402
from hydrate3w.config import ALARM_PERSIST_MIN, DOSE_THRESHOLD, GAMMAS, TAB  # noqa: E402
from hydrate3w.data import minute, real  # noqa: E402
from hydrate3w.evaluate import alarm_onsets, event_outcome, false_alarm_rate  # noqa: E402

style.apply()
S = style
import os
rows = []
frames = {} if os.environ.get("H3W_PLOT_ONLY") else {r.instance: minute(r.instance) for r in real().itertuples()}


def neg_mask(m):
    c = m["class"]
    neg = c.notna() & ~c.isin([108, 8])
    if c.eq(108).any():
        neg &= m.index < c.index[c.eq(108)][0]
    return neg


neg = {i: neg_mask(m) for i, m in frames.items()}
neg_hours = sum(float(n.sum()) for n in neg.values()) / 60
for method in ([] if os.environ.get("H3W_PLOT_ONLY") else thermo.CORRELATIONS):
    for g in GAMMAS:
        col = f"sub_{method}_{g:.2f}"
        events, fa = [], 0
        for i, m in frames.items():
            dose = thermo.exposure_dose(m[col], dt_h=1 / 60)
            flag = dose >= DOSE_THRESHOLD
            on = alarm_onsets(flag, ALARM_PERSIST_MIN)
            fa += int(neg[i].reindex(on).fillna(False).sum())
            if i.startswith("8/"):
                events.append(event_outcome(m, flag))
        ev = pd.DataFrame(events)
        rows.append(dict(correlation=method, gamma=g, warned=int(ev.detected.sum()), events=len(ev),
                         median_lead_h=ev.lead_to_plug_h.median(), min_lead_h=ev.lead_to_plug_h.min(),
                         false_alarms=fa, far_per_1000h=false_alarm_rate(fa, neg_hours)))
        print(rows[-1], flush=True)
res = pd.read_csv(TAB / "tabR3_robustness.csv") if os.environ.get("H3W_PLOT_ONLY") else pd.DataFrame(rows)
res.to_csv(TAB / "tabR3_robustness.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw=dict(wspace=0.12))
for ax, key, title, cmap, fmt, vmax in ((axes[0], "median_lead_h", "Median warning lead time (h), scale from 0", S.SEQ_CMAP, "{:.1f}", 40),
                                        (axes[1], "far_per_1000h", "False alarms per 1,000 h, scale from 0", S.SEQ_CMAP, "{:.2f}", 3)):
    M = res.pivot(index="correlation", columns="gamma", values=key).reindex(list(thermo.CORRELATIONS))
    im = ax.imshow(M.to_numpy(dtype=float), cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for (a, b), v in np.ndenumerate(M.to_numpy(dtype=float)):
        w = res[(res.correlation == M.index[a]) & (res.gamma == M.columns[b])].iloc[0]
        ax.text(b, a, fmt.format(v) + f"\n{w.warned}/{w.events} warned", ha="center", va="center", fontsize=8.5,
                color="white" if v > 0.55 * vmax else S.INK)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([f"γ = {g:.2f}" for g in M.columns])
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index if ax is axes[0] else [])
    ax.grid(False)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle(f"Physics rule (dose ≥ {DOSE_THRESHOLD:g} °C·h) across nine equilibrium assumptions", x=0.02, ha="left",
             fontweight="bold", y=1.04)
print(S.save(fig, "fig13_robustness"))
