"""Step 2. Leave-one-well-out out-of-fold scores for every model.

Every fold holds out all real instances of one well (any event class), trains on the
other wells, and scores the held-out well minute by minute. Writes
outputs/cache/oof_scores.parquet.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w.config import CACHE  # noqa: E402
from hydrate3w.data import minute, real  # noqa: E402
from hydrate3w.models import FEATURE_SETS, add_state_onehot, gbm, physics_rule_score  # noqa: E402

NEG_STRIDE = 10         # keep every 10th negative minute for training (all minutes are scored)


def design() -> pd.DataFrame:
    parts = []
    for r in real().itertuples():
        m = minute(r.instance)
        m = m.assign(instance=r.instance, well=r.well, event=r.event, t=m.index)
        parts.append(m.reset_index(drop=True))
    df = pd.concat(parts, ignore_index=True)
    return add_state_onehot(df)


def main():
    t0 = time.time()
    df = design()
    print(f"design matrix {df.shape}, {df.well.nunique()} wells, built in {time.time() - t0:.0f} s")
    labelled = df["class"].notna()
    out = df[["instance", "well", "event", "t", "class", "state", "y", "dose"]].copy()
    out["Physics rule"] = physics_rule_score(df)
    for name, cols in FEATURE_SETS.items():
        out[name] = np.nan
        for k, w in enumerate(sorted(df.well.unique())):
            test = df.well == w
            tr = labelled & ~test
            keep = tr & ((df.y == 1) | (np.arange(len(df)) % NEG_STRIDE == 0))
            clf = gbm(seed=k).fit(df.loc[keep, cols], df.loc[keep, "y"])
            out.loc[test, name] = clf.predict_proba(df.loc[test, cols])[:, 1]
            print(f"  {name} fold {k + 1:2d} ({w}) {time.time() - t0:.0f} s", flush=True)
        print(f"{name}: done in {time.time() - t0:.0f} s")
    out.to_parquet(CACHE / "oof_scores.parquet")
    print("wrote", CACHE / "oof_scores.parquet")


if __name__ == "__main__":
    main()
