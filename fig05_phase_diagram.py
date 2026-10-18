"""Figure 5. Hydrate equilibrium curves with field P-T trajectories, and the robustness of subcooling to the correlation."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style, thermo  # noqa: E402
from hydrate3w.config import GAMMAS, PA_PER_BAR, SHUTIN_STATES, T_SEABED_C  # noqa: E402
from hydrate3w.data import minute, real, short, valid_line_pt  # noqa: E402

style.apply()
S = style
inst = [i for i in real(8).sort_values(["well", "start"]).instance if valid_line_pt(i)]
frames = {i: minute(i) for i in inst}

fig = plt.figure(figsize=(13.5, 7.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.25)
ax = fig.add_subplot(gs[0])

P = np.geomspace(5e5, 4e7, 300)
for name in thermo.CORRELATIONS:
    col = S.CORR_COLORS[name]
    lo, hi = thermo.teq(P, GAMMAS[0], name), thermo.teq(P, GAMMAS[-1], name)
    ax.fill_betweenx(P / PA_PER_BAR, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(thermo.teq(P, 0.70, name), P / PA_PER_BAR, color=col, lw=2.2,
            label=f"{name}" + ("" if name == "Hammerschmidt" else ", γ 0.6 to 0.8 band"))
ax.axvspan(T_SEABED_C - 1, T_SEABED_C + 1, color=S.BLUE, alpha=0.12)
ax.text(T_SEABED_C, 4.6, "seabed\n3 to 5 °C", ha="center", fontsize=8, color=S.BLUE, fontweight="bold")

pts = pd.concat([frames[i][["T_line", "P_line", "state_name"]] for i in inst]).dropna()
draw = ["Open", "Restart", "Unknown", "Flushing Diesel", "Flushing Gas", "Bullheading", "Depressurization",
        "Closed With Diesel", "Closed With Gas", "Shut-In"]          # flowing states first, shut-in on top
for z, st in enumerate(draw):
    sub = pts[pts.state_name == st]
    if len(sub) < 5:
        continue
    sub = sub.iloc[:: max(1, len(sub) // 600)]
    flowing = st in ("Open", "Restart", "Unknown")
    ax.scatter(sub.T_line, sub.P_line / PA_PER_BAR, s=6 if flowing else 12, color=S.STATE_COLORS[st],
               alpha=0.35 if flowing else 0.9, lw=0, label=f"field: {st} (n={len(pts[pts.state_name == st]):,} min)",
               zorder=3 + z)
ax.set_yscale("log")
ax.set_xlim(-2, 60)
ax.set_ylim(4, 400)
ax.set_xlabel("temperature at subsea tree, T-TPT (°C)")
ax.set_ylabel("pressure at subsea tree, P-TPT (bar)")
ax.text(1.0, 260, "HYDRATE\nSTABLE", color="#a32625", fontsize=13, fontweight="bold", alpha=0.85)
ax.text(33, 22, "HYDRATE-FREE", color=S.BLUE, fontsize=13, fontweight="bold", alpha=0.85)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=7.3, ncol=3, markerscale=2.2)
ax.set_title(f"Shut-in states sit deep inside the stability zone ({len(inst)} events)", loc="left", fontsize=10.5)
S.panel_label(ax, "a")

# (b) per-event shut-in subcooling, central case, with the spread over all correlations and gravities
ax = fig.add_subplot(gs[1])
combos = [c for c in frames[inst[0]].columns if c.startswith("sub_")]
for k, i in enumerate(inst):
    m = frames[i]
    si = m[m.state.isin(SHUTIN_STATES)]
    central = si["sub"].dropna()
    per_combo = si[combos].median()
    parts = ax.violinplot([central], positions=[k], widths=0.75, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(S.RED)
        b.set_alpha(0.35)
    ax.scatter([k], [central.median()], color=S.RED, s=40, zorder=4, edgecolor="white")
    ax.vlines(k, per_combo.min(), per_combo.max(), color=S.INK, lw=3, zorder=5)
ax.axhline(0, color=S.INK2, lw=1)
ax.text(len(inst) - 0.5, 0.6, "equilibrium", ha="right", fontsize=8, color=S.INK2)
ax.set_xticks(range(len(inst)))
ax.set_xticklabels([short(i) for i in inst], rotation=30, ha="right")
ax.set_ylabel("subcooling during shut-in, ΔT_sub (°C)")
ax.set_title("Shut-in subcooling dwarfs the correlation spread", loc="left", fontsize=10.5)
ax.plot([], [], color=S.INK, lw=3, label="range of medians over 3 correlations x 3 gravities")
ax.scatter([], [], color=S.RED, s=40, label="median, Towler-Mokhatab, γ = 0.70")
ax.fill_between([], [], color=S.RED, alpha=0.35, label="distribution, central case")
ax.legend(loc="lower left", fontsize=7.5)
S.panel_label(ax, "b")
print(S.save(fig, "fig05_phase_diagram"))
