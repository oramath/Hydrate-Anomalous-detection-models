"""Figure 2. Composition of 3W 2.0.0 and where the hydrate instances sit in time and across wells."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402
from hydrate3w.config import CLASS_NAMES  # noqa: E402
from hydrate3w.data import index  # noqa: E402

style.apply()
S = style
idx = index()
fig = plt.figure(figsize=(12, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.42)

# (a) instances by event type and source
ax = fig.add_subplot(gs[0])
events = list(range(10))
left = np.zeros(10)
for src in ("real", "simulated", "hand-drawn"):
    n = np.array([((idx.event == e) & (idx.source == src)).sum() for e in events])
    ax.barh(events, n, left=left, color=S.SOURCE_COLORS[src], height=0.72, label=src,
            edgecolor="white", linewidth=1.2)
    left += n
for e in events:
    ax.text(left[e] + 8, e, f"{int(left[e])}", va="center", fontsize=8.5, color=S.INK2)
ax.set_yticks(events)
ax.set_yticklabels([f"{e}  {CLASS_NAMES[e]}" for e in events])
for t, e in zip(ax.get_yticklabels(), events):
    if e in (8, 9):
        t.set_fontweight("bold")
        t.set_color("#a32625")
ax.axhspan(7.5, 9.5, color=S.RED, alpha=0.08, zorder=0)
ax.invert_yaxis()
ax.set_xlabel("instances")
ax.set_xlim(0, left.max() * 1.15)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", title="source")
ax.set_title(f"{len(idx):,} instances, 302 of them hydrate events", loc="left")
S.panel_label(ax, "a")

# (b) real instances by well and date
ax = fig.add_subplot(gs[1])
real = idx[idx.source == "real"].copy()
wells = sorted(real.well.unique(), key=lambda w: int(w.split("-")[1]))
ypos = {w: i for i, w in enumerate(wells)}
groups = [("Normal operation", real.event == 0, S.AQUA, 14, 0.55),
          ("Other undesirable events", real.event.between(1, 7), S.MUTED, 14, 0.7),
          ("Hydrate in service line (9)", real.event == 9, S.VIOLET, 46, 0.95),
          ("Hydrate in production line (8)", real.event == 8, S.RED, 70, 1.0)]
for lab, mask, col, size, a in groups:
    sub = real[mask]
    ax.scatter(sub.start, sub.well.map(ypos), s=size, color=col, alpha=a, label=f"{lab}, n={len(sub)}",
               edgecolor="white" if size > 20 else "none", linewidth=0.8, zorder=3 if size > 20 else 2)
ax.set_yticks(range(len(wells)))
ax.set_yticklabels([w.replace("WELL-000", "W").replace("WELL-00", "W") for w in wells], fontsize=6.8)
hyd_wells = set(real[real.event.isin([8, 9])].well)
for t, w in zip(ax.get_yticklabels(), wells):
    if w in hyd_wells:
        t.set_color("#a32625")
        t.set_fontweight("bold")
ax.set_ylabel("well (bold red: has a hydrate event)")
ax.set_xlabel("instance start date")
ax.legend(loc="lower right", fontsize=7.8, frameon=True, facecolor="white", edgecolor=S.GRID, markerscale=1.0)
ax.set_title(f"{len(real):,} real instances from {len(wells)} wells, 2011 to 2023", loc="left")
S.panel_label(ax, "b")
print(S.save(fig, "fig02_composition"))
