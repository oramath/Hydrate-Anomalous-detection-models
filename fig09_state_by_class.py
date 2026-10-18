"""Figure 9 (and supplementary Table S2). Operational-state hours by event class: why state labels and valve states are excluded."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style  # noqa: E402
from hydrate3w.config import CLASS_NAMES, STATE_NAMES, TAB  # noqa: E402
from hydrate3w.data import minute, real  # noqa: E402

style.apply()
S = style
H = pd.DataFrame(0.0, index=range(10), columns=list(STATE_NAMES.values()) + ["Unknown"])
for r in real().itertuples():
    m = minute(r.instance)
    vc = m["state_name"].value_counts() / 60
    for k, v in vc.items():
        H.loc[r.event, k] += v
H.index = [f"{e}  {CLASS_NAMES[e]}" for e in H.index]
H.round(1).to_csv(TAB / "tabS2_state_hours_by_class.csv")

order = ["Open", "Shut-In", "Closed With Gas", "Closed With Diesel", "Depressurization", "Flushing Diesel",
         "Flushing Gas", "Bullheading", "Restart", "Unknown"]
M = H[order].to_numpy()
fig, ax = plt.subplots(figsize=(11.5, 5.4))
im = ax.imshow(np.where(M > 0, M, np.nan), cmap=S.SEQ_CMAP, norm=LogNorm(0.5, M.max()), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if M[i, j] > 0:
            ax.text(j, i, f"{M[i, j]:,.0f}" if M[i, j] >= 1 else "<1", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 60 else S.INK)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=35, ha="right")
for t in ax.get_xticklabels():
    t.set_color(S.STATE_COLORS.get(t.get_text(), S.INK))
    t.set_fontweight("bold")
ax.set_yticks(range(len(H)))
ax.set_yticklabels(H.index)
ax.get_yticklabels()[8].set_color("#a32625")
ax.get_yticklabels()[8].set_fontweight("bold")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("hours (log scale)")
ax.set_title("Non-Open states occur almost only inside hydrate instances, so state labels would leak the answer",
             loc="left")
print(S.save(fig, "fig09_state_by_class"))
print(H[order].round(1).to_string())
