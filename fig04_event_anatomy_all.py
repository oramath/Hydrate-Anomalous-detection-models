"""Figure 4. Label and operational-state anatomy of all 14 real production-line hydrate events."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402
from hydrate3w.data import minute, real, short  # noqa: E402

style.apply()
S = style
inst = real(8).sort_values(["well", "start"]).instance.tolist()
phase_map = {0: "Normal", 108: "Transient", 8: "Hydrate plug"}
fig, ax = plt.subplots(figsize=(13.5, 7.2))
ymax = 0
for k, i in enumerate(inst):
    m = minute(i)
    t = (m.index - m.index[0]).total_seconds() / 3600
    dt = 1 / 60
    ph = m["class"].map(phase_map).fillna("Unlabelled")
    y = len(inst) - 1 - k
    # upper ribbon: label phase, lower ribbon: operational state
    for series, y0, h, cmap in ((ph, y + 0.05, 0.38, S.PHASE_COLORS), (m["state_name"], y - 0.38, 0.38, S.STATE_COLORS)):
        run = (series != series.shift()).cumsum()
        for _, seg in series.groupby(run):
            a, b = t[m.index.get_indexer([seg.index[0]])[0]], t[m.index.get_indexer([seg.index[-1]])[0]] + dt
            ax.add_patch(plt.Rectangle((a, y0), b - a, h, color=cmap.get(seg.iloc[0], S.MUTED), lw=0))
    first_plug = m.index[m["class"].eq(8)]
    if len(first_plug):
        tp = (first_plug[0] - m.index[0]).total_seconds() / 3600
        ax.plot([tp, tp], [y - 0.42, y + 0.47], color="white", lw=3.2)
        ax.plot([tp, tp], [y - 0.42, y + 0.47], color=S.INK, lw=1.4)
        ax.scatter([tp], [y + 0.52], marker="v", s=36, color=S.INK, zorder=5)
    ymax = max(ymax, t[-1])
ax.set_yticks(range(len(inst)))
ax.set_yticklabels([short(i) for i in inst][::-1])
ax.set_ylim(-0.7, len(inst) - 0.3)
ax.set_xlim(0, ymax * 1.01)
ax.set_xlabel("hours since start of instance")
ax.grid(axis="y", visible=False)
ph_h = [Patch(color=S.PHASE_COLORS[k], label=k) for k in ("Normal", "Transient", "Hydrate plug", "Unlabelled")]
st_order = ["Open", "Shut-In", "Closed With Gas", "Closed With Diesel", "Depressurization",
            "Flushing Diesel", "Flushing Gas", "Bullheading", "Restart"]
st_h = [Patch(color=S.STATE_COLORS[k], label=k) for k in st_order]
l1 = ax.legend(handles=ph_h, title="class label (upper strip)", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
ax.add_artist(l1)
ax.legend(handles=st_h + [plt.Line2D([], [], color=S.INK, marker="v", lw=1.4, label="first plug label")],
          title="operational state (lower strip)", loc="upper left", bbox_to_anchor=(1.01, 0.66), fontsize=8)
ax.set_title("Hydrate plugs are labelled after long shut-ins and are confirmed at restart or diesel flushing", loc="left")
print(S.save(fig, "fig04_event_anatomy_all"))
