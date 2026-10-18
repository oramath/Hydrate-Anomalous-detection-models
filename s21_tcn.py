"""Step 2b (optional, needs torch). Temporal CNN on physics and process channels, well-held-out scoring.

Every scored minute comes from a model that never saw its well: one fold per hydrate
well (9 folds) plus 5 grouped folds over the remaining wells. Appends a "TCN physics"
column to outputs/cache/oof_scores.parquet.
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
from hydrate3w.features import PHYSICS_FEATURES, RAW_KEYS  # noqa: E402
from hydrate3w.models import make_tcn, tcn_available  # noqa: E402

CHANNELS = RAW_KEYS + PHYSICS_FEATURES
CHUNK, EPOCHS, BATCH = 360, 4, 32


def load_all():
    data = []
    for r in real().itertuples():
        m = minute(r.instance)
        data.append((r.instance, r.well, m[CHANNELS].to_numpy(np.float32), m["y"].to_numpy(np.float32),
                     m["class"].notna().to_numpy(), m.index))
    return data


def scale_fit(arrs):
    X = np.concatenate(arrs)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
    return mu, sd


def prep(X, mu, sd):
    Z = (X - mu) / sd
    mask = np.isnan(Z)
    Z[mask] = 0.0
    return np.clip(Z, -10, 10), (~mask).astype(np.float32)


def chunks(Z, M, y, lab):
    for s in range(0, len(y), CHUNK // 2):
        e = min(len(y), s + CHUNK)
        if e - s < 30:
            break
        yield np.concatenate([Z[s:e], M[s:e]], 1).T, y[s:e], lab[s:e]


def main():
    if not tcn_available():
        print("torch not installed: skipping the TCN model")
        return
    import torch

    torch.set_num_threads(2)
    data = load_all()
    wells = sorted({d[1] for d in data})
    pos_wells = sorted({d[1] for d in data if d[3].any()})
    rest = [w for w in wells if w not in pos_wells]
    folds = [[w] for w in pos_wells] + [rest[i::5] for i in range(5)]
    scores = {}
    t0 = time.time()
    for k, hold in enumerate(folds):
        train = [d for d in data if d[1] not in hold]
        mu, sd = scale_fit([d[2] for d in train])
        items = []
        for _, _, X, y, lab, _ in train:
            Z, M = prep(X.copy(), mu, sd)
            items += list(chunks(Z, M, y, lab))
        pos_w = sum(float(it[1].sum()) for it in items)
        neg_w = sum(float(((1 - it[1]) * it[2]).sum()) for it in items)
        model = make_tcn(2 * len(CHANNELS))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(neg_w / max(pos_w, 1)))
        rng = np.random.default_rng(k)
        for _ in range(EPOCHS):
            order = rng.permutation(len(items))
            model.train()
            for b in range(0, len(order), BATCH):
                batch = [items[i] for i in order[b:b + BATCH]]
                L = min(len(it[1]) for it in batch)
                x = torch.tensor(np.stack([it[0][:, :L] for it in batch]))
                yy = torch.tensor(np.stack([it[1][:L] for it in batch]))
                w = torch.tensor(np.stack([it[2][:L] for it in batch]).astype(np.float32))
                loss = (lossf(model(x), yy) * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            for inst, w, X, y, lab, idx in data:
                if w not in hold:
                    continue
                Z, M = prep(X.copy(), mu, sd)
                x = torch.tensor(np.concatenate([Z, M], 1).T[None])
                scores[inst] = pd.Series(torch.sigmoid(model(x))[0].numpy(), index=idx)
        print(f"fold {k + 1}/{len(folds)} ({','.join(hold)[:40]}) done at {time.time() - t0:.0f} s", flush=True)
    oof = pd.read_parquet(CACHE / "oof_scores.parquet")
    oof["TCN physics"] = np.nan
    for inst, s in scores.items():
        sel = oof.instance == inst
        oof.loc[sel, "TCN physics"] = s.reindex(pd.DatetimeIndex(oof.loc[sel, "t"])).to_numpy()
    oof.to_parquet(CACHE / "oof_scores.parquet")
    print("added TCN physics scores")


if __name__ == "__main__":
    main()
