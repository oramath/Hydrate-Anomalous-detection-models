"""Figure 8. Shut-in cool-down follows Newtonian decay to seabed temperature, which forecasts time to hydrate entry."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w import style, thermo  # noqa: E402
from hydrate3w.config import GAMMA_REF, SHUTIN_STATES, TAB  # noqa: E402
from hydrate3w.data import minute, real, short  # noqa: E402

style.apply()
S = style
from hydrate3w.data import qc  # noqa: E402
_q = qc()
_tok = set(_q[(_q.variable == "T-TPT") & (_q.status == "valid")].instance)
inst = [i for i in real(8).sort_values(["well", "start"]).instance if i in _tok]
palette = [S.BLUE, S.ORANGE, S.AQUA, S.YELLOW, S.MAGENTA, S.GREEN, S.VIOLET, S.RED]
rows, curves = [], []
for k, i in enumerate(inst):
    m = minute(i)
    si = m.state.isin(SHUTIN_STATES)
    run = (si != si.shift()).cumsum()
    for _, seg in m[si].groupby(run[si]):
        if len(seg) < 60:
            continue
        T = seg["T_line"].to_numpy()
        th = np.arange(len(seg)) / 60
        T_inf = float(np.clip(np.nanmin(T), 2.0, 6.0))
        if not np.isfinite(T[:5]).any() or np.nanmax(T[:5]) - T_inf < 8:
            continue
        fit = thermo.fit_cooldown(th, T, T_inf)
        if fit is None or fit["r2"] < 0.8 or fit["T0"] - T_inf < 8:
            continue
        p0 = seg["P_line"].iloc[:60].median()
        teq0 = float(thermo.teq(p0, GAMMA_REF))
        pred = float(thermo.time_to_hydrate(fit["T0"], teq0, fit["tau_h"], T_inf))
        inside = np.where(seg["sub"].to_numpy() > 0)[0]
        obs = th[inside[0]] if len(inside) else np.nan
        rows.append(dict(instance=i, label=short(i), start=seg.index[0], hours=len(seg) / 60, T0=fit["T0"],
                         T_inf=T_inf, tau_h=fit["tau_h"], r2=fit["r2"], T_eq0=teq0,
                         t_hyd_pred_h=pred, t_hyd_obs_h=obs, color=palette[k % 8]))
        curves.append((th, T, fit, T_inf, palette[k % 8], short(i)))
res = pd.DataFrame(rows)
res.drop(columns="color").to_csv(TAB / "tab_cooldown_fits.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1, 1], wspace=0.3))
ax = axes[0]
seen = set()
for th, T, fit, T_inf, col, lab in curves:
    theta = (T - T_inf) / (fit["T0"] - T_inf)
    keep = th <= 6
    ax.plot(th[keep], theta[keep], color=col, lw=1.2, alpha=0.8, label=lab if lab not in seen else None)
    ax.plot(th[keep], np.exp(-th[keep] / fit["tau_h"]), color=col, lw=1, ls="--", alpha=0.9)
    seen.add(lab)
ax.set_yscale("log")
ax.set_ylim(1e-2, 1.5)
ax.set_xlim(0, 6)
ax.set_xlabel("hours since shut-in")
ax.set_ylabel("θ = (T − T_sea) / (T0 − T_sea)")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Measured (solid) vs exp(−t/τ) (dashed)", loc="left")
S.panel_label(ax, "a")

ax = axes[1]
ax.bar(range(len(res)), res.tau_h, color=res.color, edgecolor="white")
for j, r in res.iterrows():
    ax.text(j, r.tau_h, f"R²={r.r2:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.set_xticks(range(len(res)))
ax.set_xticklabels([f"{r.label}\n#{j + 1}" for j, r in res.iterrows()], rotation=90, fontsize=6.8)
ax.set_ylabel("cool-down time constant τ (h)")
ax.set_ylim(0, res.tau_h.max() * 1.35)
ax.set_title(f"τ median {res.tau_h.median():.1f} h over {len(res)} shut-ins", loc="left")
S.panel_label(ax, "b")

ax = axes[2]
ok = res.dropna(subset=["t_hyd_obs_h"])
ax.scatter(ok.t_hyd_obs_h, ok.t_hyd_pred_h, c=ok.color, s=70, edgecolor="white", zorder=3)
lim = max(1.0, float(np.nanmax([ok.t_hyd_obs_h.max(), ok.t_hyd_pred_h.replace(np.inf, np.nan).max()])) * 1.15)
ax.plot([0, lim], [0, lim], color=S.INK2, lw=1, ls="--")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("observed time to enter hydrate zone (h)")
ax.set_ylabel("forecast  τ ln[(T0 − T_sea)/(T_eq − T_sea)] (h)")
mae = np.nanmean(np.abs(ok.t_hyd_obs_h - ok.t_hyd_pred_h))
ax.set_title(f"Forecast error {mae:.2f} h (mean absolute)", loc="left")
S.panel_label(ax, "c")
print(S.save(fig, "fig08_cooldown"))
print(res.drop(columns=["color", "start"]).round(2).to_string())
