"""Figure 3. Sensor quality of every real hydrate instance after automatic QC."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402
from hydrate3w.config import PROCESS_VARS  # noqa: E402
from hydrate3w.data import qc, real, short  # noqa: E402

style.apply()
S = style
q = qc()
order = ["valid", "frozen", "implausible", "missing"]
code = {k: i for i, k in enumerate(order)}
cmap = ListedColormap([S.QC_COLORS[k] for k in order])
key = ["P-PDG", "T-PDG", "P-TPT", "T-TPT", "PT-P", "P-MON-CKP", "T-MON-CKP", "P-MON-SDV-P",
       "P-JUS-CKP", "T-JUS-CKP", "ABER-CKP", "P-ANULAR", "P-JUS-CKGL", "P-MON-CKGL", "QGL",
       "ABER-CKGL", "QBS", "P-JUS-BS"]
assert set(key) == set(PROCESS_VARS)

fig, axes = plt.subplots(1, 2, figsize=(13, 9.2), gridspec_kw=dict(width_ratios=[14, 57], wspace=0.08))
for ax, ev, title in zip(axes, (8, 9), ("(a) production line", "(b) service line")):
    inst = real(ev).sort_values(["well", "start"]).instance.tolist()
    M = np.array([[code[q[(q.instance == i) & (q.variable == v)].status.iloc[0]] for i in inst] for v in key])
    ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(inst)))
    ax.set_xticklabels([short(i) for i in inst], rotation=90, fontsize=6.6)
    ax.set_xticks(np.arange(-0.5, len(inst)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(key)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    valid_frac = (M == 0).mean(axis=1)
    ax.set_title(f"{title} (n={len(inst)})", loc="left")
    if ev == 8:
        ax.set_yticks(range(len(key)))
        ax.set_yticklabels(key)
        for t in ax.get_yticklabels():
            if t.get_text() in ("P-TPT", "T-TPT"):
                t.set_fontweight("bold")
                t.set_color(S.RED)
    else:
        ax.set_yticks([])
        axr = ax.twinx()
        axr.set_ylim(ax.get_ylim())
        axr.set_yticks(range(len(key)))
        axr.set_yticklabels([f"{100 * f:.0f}%" for f in valid_frac], fontsize=7.5, color=S.INK2)
        axr.grid(False)
        axr.spines[:].set_visible(False)
        axr.set_ylabel("share of class 9 instances where the sensor is valid", fontsize=8.5)
fig.legend(handles=[Patch(fc=S.QC_COLORS[k], label=k) for k in order], loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 0.975))
fig.suptitle("The tree pressure and temperature that drive hydrate subcooling are valid in only a minority of real events",
             x=0.08, y=1.0, ha="left", fontsize=12, fontweight="bold")
print(S.save(fig, "fig03_sensor_quality"))
