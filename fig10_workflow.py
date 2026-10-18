"""Figure 10. Study workflow from raw 1 Hz records to event-level early-warning metrics."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402

style.apply()
S = style
fig, ax = plt.subplots(figsize=(13, 5.0))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.0)
ax.axis("off")


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=col, ec="none", alpha=0.95))
    ax.text(x + 0.12, y + h - 0.14, title, fontsize=9.5, fontweight="bold", color="white", va="top")
    ax.text(x + 0.12, y + h - 0.48, body, fontsize=7.9, color="white", va="top", linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 color=S.INK2, lw=1.6))


box(0.1, 2.75, 2.3, 2.05, "1  Data", "3W 2.0.0, Figshare\n1,119 real instances\n40 wells, 1 Hz\n27 variables\nclass and state labels",
    S.BLUE)
box(2.75, 2.75, 2.3, 2.05, "2  Quality control", "plausible-range check\nfrozen-run check (1 h)\nminute averaging\nline P hierarchy:\nP-TPT, PT-P, SDV, CKP", S.AQUA)
box(5.4, 2.75, 2.6, 2.05, "3  Physics layer", "T_eq(P, γ): 3 correlations\nγ = 0.60, 0.70, 0.80\nΔT_sub = T_eq − T\ndose D = ∫ max(ΔT_sub, 0) dt\nτ, time-to-hydrate\nline ΔP indices", S.RED)
box(8.35, 2.75, 2.2, 2.05, "4  Statistics layer", "12 process signals\n10 min std\n30 min slope\n60 min difference", S.VIOLET)
box(10.9, 2.75, 2.0, 2.05, "Excluded inputs", "state labels\nvalve states (ESTADO)\nwell identity\nsimulated instances", "#52514e")
box(0.1, 0.1, 3.3, 1.9, "5  Models", "M1 physics rule: D ≥ 5 °C·h\nM2 gradient boosting, raw\nM3 gradient boosting, physics\nM4 temporal CNN, physics\nM0 state shortcut (audit only)", S.ORANGE)
box(3.75, 0.1, 3.3, 1.9, "6  Validation", "leave-one-well-out over 40 wells\nall instances of a well held out\nTCN: 9 hydrate-well + 5 grouped folds\nalarm = 10 consecutive minutes", S.GREEN)
box(7.4, 0.1, 2.7, 1.9, "7  Event metrics", "detection before plug label\nlead time to first plug label\nfalse alarms per 1,000 h\nfull threshold sweep", S.YELLOW)
box(10.45, 0.1, 2.45, 1.9, "8  Robustness", "correlation x γ sweep\ndose threshold sweep\nper-well results\nphysics ablation (M2 vs M3)", S.MAGENTA)
for (x0, x1) in ((2.4, 2.75), (5.05, 5.4), (8.0, 8.35)):
    arrow(x0, 3.8, x1, 3.8)
arrow(6.7, 2.75, 2.0, 2.0)
arrow(9.45, 2.75, 2.6, 2.0)
for (x0, x1) in ((3.4, 3.75), (7.05, 7.4), (10.1, 10.45)):
    arrow(x0, 1.05, x1, 1.05)
ax.set_title("From raw 1 Hz records to event-level early-warning metrics", loc="left")
print(S.save(fig, "fig10_workflow"))
