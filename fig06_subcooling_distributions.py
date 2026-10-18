"""Figure 6. Subcooling separates flowing operation from hydrate build-up across the whole real dataset."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402
from hydrate3w.config import SHUTIN_STATES, TAB  # noqa: E402
from hydrate3w.data import minute, real, valid_line_pt  # noqa: E402

style.apply()
S = style
groups = {}


def add(name, s):
    groups.setdefault(name, []).append(s.dropna().to_numpy())


for r in real().itertuples():
    if not valid_line_pt(r.instance):
        continue
    m = minute(r.instance)
    sub, c, st = m["sub"], m["class"], m["state"]
    if r.event == 0:
        add("Normal operation (class 0)", sub[c == 0])
    elif r.event in range(1, 8):
        add("Other events 1 to 7, all minutes", sub)
    elif r.event == 9:
        add("Service-line hydrate (class 9)", sub)
    elif r.event == 8:
        add("Class 8: normal-labelled flow", sub[c == 0])
        add("Class 8: transient, well open", sub[(c == 108) & (st == 0)])
        add("Class 8: transient, shut in", sub[(c == 108) & st.isin(SHUTIN_STATES)])
        add("Class 8: plug label", sub[c == 8])
order = ["Normal operation (class 0)", "Other events 1 to 7, all minutes", "Service-line hydrate (class 9)",
         "Class 8: normal-labelled flow", "Class 8: transient, well open", "Class 8: transient, shut in",
         "Class 8: plug label"]
cols = [S.AQUA, S.AQUA, S.AQUA, S.AQUA, S.AQUA, S.BLUE, S.PHASE_COLORS["Hydrate plug"]]  # Open, Shut-In, plug
data = [np.concatenate(groups[g]) if g in groups else np.array([np.nan]) for g in order]
summary = pd.DataFrame(dict(group=order, minutes=[len(d) for d in data],
                            hours=[len(d) / 60 for d in data],
                            median_sub=[np.nanmedian(d) for d in data],
                            share_positive=[np.nanmean(d > 0) for d in data]))
summary.to_csv(TAB / "tab_subcooling_by_group.csv", index=False)

fig, ax = plt.subplots(figsize=(11.5, 5.8))
ax.axvspan(0, 45, color=S.RED, alpha=0.06)
ax.axvline(0, color=S.INK, lw=1.2)
for k, (d, col) in enumerate(zip(data, cols)):
    d = d[np.isfinite(d)]
    if len(d) < 5:
        continue
    samp = d if len(d) < 20000 else np.random.default_rng(0).choice(d, 20000, replace=False)
    parts = ax.violinplot([samp], positions=[k], vert=False, widths=0.85, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(col)
        b.set_edgecolor(col)
        b.set_alpha(0.55)
    q1, q2, q3 = np.percentile(d, [25, 50, 75])
    ax.hlines(k, q1, q3, color=S.INK, lw=4)
    ax.scatter([q2], [k], color="white", edgecolor=S.INK, s=34, zorder=4)
    ax.text(47, k, f"{100 * np.mean(d > 0):.0f}% inside\n{len(d) / 60:,.0f} h", va="center", fontsize=8,
            color="#a32625" if np.mean(d > 0) > 0.5 else S.INK2)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.invert_yaxis()
ax.set_xlim(-135, 62)
ax.set_xlabel("subcooling at the subsea tree, ΔT_sub = T_eq − T (°C)    (Towler-Mokhatab, γ = 0.70)")
ax.text(2, -0.75, "inside hydrate stability zone →", color="#a32625", fontsize=9, fontweight="bold")
ax.text(-2, -0.75, "← hydrate-free", color=S.BLUE, fontsize=9, fontweight="bold", ha="right")
ax.grid(axis="y", visible=False)
ax.set_title("Flowing wells sit far outside the stability zone and shut-in hydrate build-up sits far inside it",
             loc="left")
print(S.save(fig, "fig06_subcooling_distributions"))
print(summary.round(2).to_string())
