"""Figure 1. Satellite subsea well, production and service lines, and the 3W sensor positions."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402

style.apply()
S = style
fig, ax = plt.subplots(figsize=(11, 6.6))
ax.set_xlim(0, 13)
ax.set_ylim(-0.4, 10.9)
ax.axis("off")

# sea, seabed, rock
sea = np.linspace(0, 1, 200)[:, None]
ax.imshow(sea, extent=(0, 13, 2.4, 8.6), cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "sea", ["#1f5fa8", "#4f94e0", "#cfe7fb"]), aspect="auto", origin="lower", zorder=0)
ax.add_patch(Rectangle((0, -0.4), 13, 2.8, color="#d9c7a3", zorder=0))
ax.add_patch(Rectangle((0, -0.4), 13, 0.9, color="#b99a6a", zorder=0))
ax.text(12.85, 8.75, "sea surface", ha="right", fontsize=8.5, color=S.INK2)
ax.text(12.85, 2.15, "seabed, about 4 °C", ha="right", fontsize=8.5, color=S.INK2)
ax.text(12.85, 0.05, "reservoir", ha="right", fontsize=8.5, color="white", fontweight="bold")

# platform
ax.add_patch(FancyBboxPatch((0.4, 8.6), 4.8, 0.9, boxstyle="round,pad=0.05", fc="#52514e", ec="none"))
ax.add_patch(Rectangle((0.7, 9.5), 4.2, 0.45, fc="#8f8e88", ec="none"))
ax.text(6.0, 9.2, "Production platform\n(topside)", ha="left", fontsize=9.5, fontweight="bold", color=S.INK)

# hydrate risk envelope: cold seabed flowline and lower riser
risk = Polygon([[2.35, 2.35], [2.35, 5.2], [3.05, 5.2], [3.05, 2.95], [9.6, 2.95], [9.6, 2.35]],
               closed=True, fc=S.RED, alpha=0.35, ec="white", lw=1.4, ls="--", zorder=1)
ax.add_patch(risk)
ax.text(6.0, 3.65, "hydrate stability zone during shut-in\n(high P, line cools to seabed temperature)",
        ha="center", fontsize=8.5, color="#a32625", fontweight="bold", zorder=5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.92))

# lines: production (blue) and service (orange)
prod_x, prod_y = [2.6, 2.6, 9.3, 9.3], [8.6, 2.6, 2.6, 2.75]
serv_x, serv_y = [2.85, 2.85, 9.55, 9.55], [8.6, 2.75, 2.75, 2.85]
ax.plot(prod_x, prod_y, color=S.BLUE, lw=5, solid_capstyle="round", zorder=3)
ax.plot(serv_x, serv_y, color=S.ORANGE, lw=3.5, solid_capstyle="round", zorder=3)
ax.text(1.5, 6.6, "production\nline / riser", color="white", fontsize=8.5, ha="center", fontweight="bold")
ax.text(3.75, 6.6, "service line\n(gas lift)", color="white", fontsize=8.5, ha="center", fontweight="bold")

# subsea tree and well
ax.add_patch(FancyBboxPatch((9.0, 2.4), 0.9, 1.0, boxstyle="round,pad=0.04", fc="#3c3b38", ec="none", zorder=4))
ax.text(9.45, 3.55, "subsea tree", ha="center", fontsize=8.5, color="white", fontweight="bold")
ax.plot([9.45, 9.45], [2.4, 0.15], color="#3c3b38", lw=6, zorder=3)
ax.plot([9.45, 9.45], [2.4, 0.15], color=S.BLUE, lw=2.5, zorder=4)

# sensors: (x, y, tag, quantity)
qcol = {"P": S.VIOLET, "T": S.RED, "Q": S.GREEN, "V": S.YELLOW}
sensors = [
    (1.0, 9.05, "P-JUS-CKP\nT-JUS-CKP", "P", "above"), (2.25, 9.05, "P-MON-CKP\nT-MON-CKP", "T", "above"),
    (3.45, 9.05, "PCK\nABER-CKP", "V", "above"), (4.6, 9.05, "GLCK, QGL\nP-JUS-CKGL", "Q", "above"),
    (2.0, 8.15, "SDV-P\nP-MON-SDV-P", "V", "left"), (10.4, 3.1, "TPT: P-TPT, T-TPT", "T", "right"),
    (10.4, 2.55, "W1, M1, XO, PXO, PT-P", "V", "right"), (10.2, 1.3, "DHSV", "V", "right"),
    (10.2, 0.5, "PDG: P-PDG, T-PDG", "T", "right"), (8.3, 1.75, "annulus: P-ANULAR", "P", "left"),
]
box = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88)
for x, y, tag, q, where in sensors:
    ax.scatter([x], [y], s=120, color=qcol[q], ec="white", lw=1.6, zorder=6)
    if where == "above":
        ax.text(x, 9.72 if False else 10.0 + 0.0, "", fontsize=1)
        ax.annotate(tag, (x, y), xytext=(x, 8.0 if False else 9.95), ha="center", va="bottom",
                    fontsize=7.4, zorder=7, bbox=box, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.6))
    elif where == "left":
        ax.text(x - 0.2, y, tag, fontsize=7.4, va="center", ha="right", zorder=7, bbox=box, fontweight="bold")
    else:
        ax.text(x + 0.2, y, tag, fontsize=7.4, va="center", ha="left", zorder=7, bbox=box, fontweight="bold")

# legend
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], color=S.BLUE, lw=5, label="production line"),
           Line2D([], [], color=S.ORANGE, lw=3.5, label="service line"),
           Line2D([], [], marker="o", ls="", color=qcol["P"], ms=9, label="pressure"),
           Line2D([], [], marker="o", ls="", color=qcol["T"], ms=9, label="pressure and temperature"),
           Line2D([], [], marker="o", ls="", color=qcol["Q"], ms=9, label="flow, gas lift"),
           Line2D([], [], marker="o", ls="", color=qcol["V"], ms=9, label="valve or choke"),
           plt.matplotlib.patches.Patch(fc=S.RED, alpha=0.25, ec=S.RED, ls="--", label="hydrate risk envelope")]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.80), ncol=2, fontsize=8,
          frameon=True, facecolor="white", edgecolor=S.GRID)
ax.set_title("Satellite subsea well layout behind the 3W 2.0.0 variables", loc="left", pad=6)
print(style.save(fig, "fig01_schematic"))
