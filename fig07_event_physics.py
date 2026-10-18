"""Figure 7. Physics of one real hydrate event: pressures, subcooling, exposure dose and the warning lead time."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style, thermo  # noqa: E402
from hydrate3w.config import ALARM_PERSIST_MIN, DOSE_THRESHOLD, GAMMAS, PA_PER_BAR  # noqa: E402
from hydrate3w.data import minute, short  # noqa: E402
from hydrate3w.evaluate import first_alarm  # noqa: E402

style.apply()
S = style
INST = sys.argv[1] if len(sys.argv) > 1 else "8/WELL-00019_20141117190526"
m = minute(INST)
t = (m.index - m.index[0]).total_seconds() / 3600
idx0 = m.index[0]
hrs = lambda ts: (ts - idx0).total_seconds() / 3600  # noqa: E731
t_tr = hrs(m.index[m["class"].eq(108)][0])
t_pl = hrs(m.index[m["class"].eq(8)][0])
alarm = first_alarm(m["dose"] >= DOSE_THRESHOLD, ALARM_PERSIST_MIN)
t_al = hrs(alarm) if alarm is not None else np.nan

fig, axes = plt.subplots(4, 1, figsize=(12, 9.6), sharex=True,
                         gridspec_kw=dict(height_ratios=[1.1, 1.3, 1.0, 0.45], hspace=0.12))
ax = axes[0]
for v, col, lab in (("P-PDG", S.VIOLET, "downhole, P-PDG"), ("P-TPT", S.BLUE, "subsea tree, P-TPT"),
                    ("P-MON-CKP", S.ORANGE, "topside, P-MON-CKP")):
    ax.plot(t, m[v] / PA_PER_BAR, color=col, label=lab)
ax.set_ylabel("pressure (bar)")
ax.legend(loc="upper right", ncol=3)
t_in = hrs(m.index[(m["sub"] > 0) & (m.index > m.index[m["class"].eq(108)][0])][0])
ax.set_title(f"{short(INST)}: the tree enters the hydrate zone {60 * (t_in - t_tr):.0f} min after shut-in "
             f"and stays inside for {t_pl - t_in:.0f} h before the plug label", loc="left")

ax = axes[1]
teq_lo = thermo.teq(m["P_line"], GAMMAS[0])
teq_hi = thermo.teq(m["P_line"], GAMMAS[-1])
ax.fill_between(t, teq_lo, teq_hi, color=S.VIOLET, alpha=0.25, lw=0, label="hydrate equilibrium T_eq, γ 0.6 to 0.8")
ax.plot(t, m["T_eq"], color=S.VIOLET, lw=1.2)
ax.plot(t, m["T_line"], color=S.RED, lw=1.8, label="line temperature, T-TPT")
inside = (m["T_line"] < m["T_eq"]).to_numpy()
ax.fill_between(t, m["T_line"], m["T_eq"], where=inside, color=S.RED, alpha=0.25, lw=0,
                label="subcooling ΔT_sub > 0")
ax.axhline(4.0, color=S.BLUE, lw=1, ls=":")
ax.text(t[-1], 5.0, "seabed 4 °C", ha="right", fontsize=8, color=S.BLUE)
ax.set_ylabel("temperature (°C)")
ax.legend(loc="upper right", ncol=3)

ax = axes[2]
ax.plot(t, m["dose"].clip(lower=0.1), color=S.RED, lw=2, label="exposure dose D")
ax.axhline(DOSE_THRESHOLD, color=S.INK2, ls="--", lw=1)
ax.text(1, DOSE_THRESHOLD * 1.25, f"alarm threshold {DOSE_THRESHOLD:g} °C·h", fontsize=8, color=S.INK2)
ax.set_yscale("log")
ax.set_ylim(0.1, max(10, m["dose"].max() * 2))
ax.set_ylabel("dose (°C·h)")
if np.isfinite(t_al):
    ax.annotate("", xy=(t_pl, DOSE_THRESHOLD * 4), xytext=(t_al, DOSE_THRESHOLD * 4),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=1.6))
    ax.text((t_al + t_pl) / 2, DOSE_THRESHOLD * 5.5, f"warning lead time {t_pl - t_al:.1f} h",
            ha="center", fontsize=9.5, fontweight="bold")

ax = axes[3]
ph = m["class"].map({0: "Normal", 108: "Transient", 8: "Hydrate plug"}).fillna("Unlabelled")
for series, y0, cmap in ((ph, 0.52, S.PHASE_COLORS), (m["state_name"], 0.0, S.STATE_COLORS)):
    run = (series != series.shift()).cumsum()
    for _, seg in series.groupby(run):
        a = t[m.index.get_indexer([seg.index[0]])[0]]
        b = t[m.index.get_indexer([seg.index[-1]])[0]] + 1 / 60
        ax.add_patch(plt.Rectangle((a, y0), b - a, 0.46, color=cmap.get(seg.iloc[0], S.MUTED), lw=0))
        if b - a > 0.06 * t[-1]:
            ax.text((a + b) / 2, y0 + 0.23, seg.iloc[0], ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")
ax.set_ylim(0, 1)
ax.set_yticks([0.23, 0.75])
ax.set_yticklabels(["state", "label"])
ax.grid(False)
ax.set_xlabel("hours since start of instance")

for a in axes:
    for x, col, ls in ((t_tr, S.YELLOW, "--"), (t_pl, S.RED, "-"), (t_al, S.INK, ":")):
        if np.isfinite(x):
            a.axvline(x, color=col, ls=ls, lw=1.4)
axes[0].text(t_tr, axes[0].get_ylim()[1], " transient label", color="#9a6a00", fontsize=8, va="top")
axes[0].text(t_pl, axes[0].get_ylim()[1], " first plug label", color="#a32625", fontsize=8, va="top")
if np.isfinite(t_al):
    axes[0].text(t_al, axes[0].get_ylim()[1] * 0.88, " physics alarm", color=S.INK, fontsize=8, va="top")
axes[-1].set_xlim(0, t[-1])
print(S.save(fig, "fig07_event_physics"))
