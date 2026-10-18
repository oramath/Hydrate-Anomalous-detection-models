"""Step 0. Quality-control every real instance and cache minute-level feature frames.

Writes
  outputs/cache/minute/<event>_<file>.parquet   minute features per real instance
  outputs/tables/qc_records.csv                  per-instance, per-variable QC status
  outputs/tables/instance_index.csv              all 2,228 instances with label hours
"""
from __future__ import annotations

import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from types import SimpleNamespace as _NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hydrate3w.config import CACHE, TAB  # noqa: E402
from hydrate3w.features import minute_frame  # noqa: E402
from hydrate3w.io import list_instances, load  # noqa: E402

MIN_DIR = CACHE / "minute"
MIN_DIR.mkdir(parents=True, exist_ok=True)


def _label_hours(row) -> dict:
    row = _NS(**row)
    lab = load(row.path, columns=["class", "state"])
    c = lab["class"]
    ev = row.event
    return dict(instance=row.instance, hours=len(lab) / 3600,
                h_normal=(c == 0).sum() / 3600,
                h_transient=(c == 100 + ev).sum() / 3600 if ev else 0.0,
                h_event=(c == ev).sum() / 3600 if ev else 0.0,
                h_unlabelled=c.isna().sum() / 3600,
                n_states=int(lab["state"].dropna().nunique()))


def _process(row):
    row = _NS(**row)
    out = MIN_DIR / f"{row.event}_{Path(row.path).stem}.parquet"
    m, qc = minute_frame(row.path)
    m.to_parquet(out)
    qc.insert(0, "instance", row.instance)
    return qc


def main():
    idx = list_instances()
    with Pool(max(1, cpu_count())) as pool:
        hours = pool.map(_label_hours, [r._asdict() for r in idx.itertuples(index=False)])
    idx = idx.merge(pd.DataFrame(hours), on="instance")
    idx.to_csv(TAB / "instance_index.csv", index=False)
    real = idx[idx.source == "real"]
    with Pool(max(1, cpu_count())) as pool:
        qcs = pool.map(_process, [r._asdict() for r in real.itertuples(index=False)])
    pd.concat(qcs).to_csv(TAB / "qc_records.csv", index=False)
    print(f"cached {len(real)} real instances, indexed {len(idx)} instances")


if __name__ == "__main__":
    main()
