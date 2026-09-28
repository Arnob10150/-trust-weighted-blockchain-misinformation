"""Evaluation protocols for the risk-scoring engine (v2, cached embeddings).

P1  in-domain    : ISOT, random stratified 80/20 split -- the convention in
                   this literature.
P2  scrubbed     : P1 with wire-service boilerplate removed. A negative
                   control: if the in-domain score were carried by a handful of
                   agency tokens, removing them would move it.
P3  cross-corpus : train ISOT -> test PolitiFact / BuzzFeed / GossipCop.
                   Unseen publishers, unseen period = leakage-controlled.
P4  pooled       : train on a publisher-disjoint pool, stratified so that the
                   held-out sources keep a usable class balance, and tested on
                   held-out publishers and on BuzzFeed.

Calibration is fitted three ways -- none, on a source-domain holdout, and on a
small slice of the target domain -- because the deployment question is whether
a threshold set on one distribution stays meaningful on another.
"""
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, EPOCHS, LR, SEED = 128, 8, 1e-3, 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DF = pd.read_csv(os.path.join(ART, "corpus.csv")).fillna({"text": ""})
EMB = np.load(os.path.join(ART, "emb.npy"), mmap_mode="r")
EMB_S = np.load(os.path.join(ART, "emb_scrub.npy"), mmap_mode="r")


class Idx(Dataset):
    def __init__(self, idx, src):
        self.idx = np.asarray(idx)
        self.src = src

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return (torch.from_numpy(np.asarray(self.src[j], dtype=np.float32)),
                torch.tensor(float(DF.label.iloc[j])))


class Head(nn.Module):
    """Projection -> BiLSTM -> dropout -> linear, on cached BERT states."""

    def __init__(self, proj=128, hidden=64, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(768, proj)
        self.lstm = nn.LSTM(proj, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        x = torch.relu(self.proj(x))
        _, (h, _) = self.lstm(x)
        return self.fc(self.drop(torch.cat((h[-2], h[-1]), 1))).squeeze(-1)


def train(idx, src, epochs=EPOCHS):
    m = Head().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    dl = DataLoader(Idx(idx, src), batch_size=BATCH, shuffle=True, num_workers=0)
    m.train()
    for ep in range(epochs):
        tot = n = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = lossf(m(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        print(f"    epoch {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)
    return m


@torch.no_grad()
def infer(m, idx, src):
    m.eval()
    dl = DataLoader(Idx(idx, src), batch_size=512, shuffle=False)
    zs, ys = [], []
    t0 = time.perf_counter()
    for x, y in dl:
        zs.append(m(x.to(DEV)).cpu())
        ys.append(y)
    if DEV == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / max(len(idx), 1)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy(), ms


def fit_T(z, y):
    """Single-parameter temperature, fitted by NLL on (z, y)."""
    zt = torch.tensor(z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    lossf = nn.BCEWithLogitsLoss()

    def cl():
        opt.zero_grad()
        loss = lossf(zt / logT.exp(), yt)
        loss.backward()
        return loss

    opt.step(cl)
    return float(logT.exp().item())


def ece(p, y, bins=15):
    e = np.linspace(0, 1, bins + 1)
    v, n = 0.0, len(p)
    for i in range(bins):
        m = (p > e[i]) & (p <= e[i + 1])
        if m.sum():
            v += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(v)


def metrics(z, y, ms, T=1.0, thr=0.5):
    p = 1 / (1 + np.exp(-z / T))
    pred = (p > thr).astype(int)
    y = y.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "pos_rate": float(y.mean()),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "ece": ece(p, y), "fpr": fp / max(fp + tn, 1),
        "latency_ms": ms, "T": T, "thr": thr,
    }


def rows(corpus):
    return DF.index[DF.corpus == corpus].to_numpy()


def stratified_publisher_holdout(pool_idx, frac=0.2, seed=SEED):
    """Hold out whole publishers, but stratify the choice by corpus and by the
    publisher's majority class, so the held-out split keeps a usable class
    balance instead of collapsing to one corpus's majority label."""
    sub = DF.loc[pool_idx]
    rng = np.random.RandomState(seed)
    hold = []
    key = sub.groupby("publisher").agg(corpus=("corpus", "first"),
                                       maj=("label", lambda s: int(s.mean() > 0.5)))
    for (_, _), g in key.groupby(["corpus", "maj"]):
        pubs = g.index.to_numpy()
        rng.shuffle(pubs)
        hold += list(pubs[: max(1, int(round(frac * len(pubs))))])
    hold = set(hold)
    mask = sub.publisher.isin(hold).to_numpy()
    return pool_idx[~mask], pool_idx[mask]


def main():
    R = {}
    isot, poli, buzz, goss = (rows("isot"), rows("politifact"),
                              rows("buzzfeed"), rows("gossipcop"))

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(isot)
    k = int(0.8 * len(perm))
    tr_i, te_i = perm[:k], perm[k:]

    # ---- P1 ------------------------------------------------------------
    print("[P1] ISOT random split")
    m1 = train(tr_i, EMB)
    z, y, ms = infer(m1, te_i, EMB)
    R["P1_isot_random"] = metrics(z, y, ms)
    z_src, y_src = z, y                      # source-domain calibration set

    # ---- P2 negative control -------------------------------------------
    print("[P2] ISOT, boilerplate scrubbed")
    m2 = train(tr_i, EMB_S)
    z, y, ms = infer(m2, te_i, EMB_S)
    R["P2_isot_scrubbed"] = metrics(z, y, ms)

    # ---- P3 transfer ----------------------------------------------------
    print("[P3] cross-corpus transfer from ISOT")
    T_src = fit_T(z_src, y_src)
    print(f"    source-fitted T={T_src:.3f}")
    R["_T_source"] = T_src
    for nm, idx in (("politifact", poli), ("buzzfeed", buzz), ("gossipcop", goss)):
        z, y, ms = infer(m1, idx, EMB)
        R[f"P3_isot_to_{nm}"] = metrics(z, y, ms)
        R[f"P3_isot_to_{nm}_calsrc"] = metrics(z, y, ms, T=T_src)
        # target-domain calibration: fit T and the threshold on 20% of the
        # target, score the remaining 80%
        rs = np.random.RandomState(SEED)
        pm = rs.permutation(len(idx))
        c = max(50, int(0.2 * len(idx)))
        ci, ei = pm[:c], pm[c:]
        T_t = fit_T(z[ci], y[ci])
        pc = 1 / (1 + np.exp(-z[ci] / T_t))
        grid = np.linspace(0.05, 0.95, 91)
        thr = float(grid[np.argmax([f1_score(y[ci].astype(int), (pc > t).astype(int),
                                             zero_division=0) for t in grid])])
        R[f"P3_isot_to_{nm}_caltgt"] = metrics(z[ei], y[ei], ms, T=T_t, thr=thr)

    # ---- P4 pooled, publisher-disjoint, stratified ----------------------
    print("[P4] pooled, stratified publisher-disjoint")
    pool = np.concatenate([isot, poli, goss])
    ptr, pte = stratified_publisher_holdout(pool)
    print(f"    train {len(ptr)} (fake {DF.label.iloc[ptr].mean():.3f}) | "
          f"test {len(pte)} (fake {DF.label.iloc[pte].mean():.3f})")
    R["_p4_split"] = {"n_train": int(len(ptr)), "n_test": int(len(pte)),
                      "train_pos": float(DF.label.iloc[ptr].mean()),
                      "test_pos": float(DF.label.iloc[pte].mean())}
    m4 = train(ptr, EMB)
    z4, y4, ms = infer(m4, pte, EMB)
    R["P4_pooled_heldout_publishers"] = metrics(z4, y4, ms)

    zb, yb, msb = infer(m4, buzz, EMB)
    R["P4_pooled_to_buzzfeed"] = metrics(zb, yb, msb)
    rs = np.random.RandomState(SEED)
    pm = rs.permutation(len(buzz))
    c = int(0.2 * len(buzz))
    ci, ei = pm[:c], pm[c:]
    T_b = fit_T(zb[ci], yb[ci])
    pc = 1 / (1 + np.exp(-zb[ci] / T_b))
    grid = np.linspace(0.05, 0.95, 91)
    thr = float(grid[np.argmax([f1_score(yb[ci].astype(int), (pc > t).astype(int),
                                         zero_division=0) for t in grid])])
    R["P4_pooled_to_buzzfeed_caltgt"] = metrics(zb[ei], yb[ei], msb, T=T_b, thr=thr)

    np.savez(os.path.join(ART, "scores.npz"),
             buzz_z=zb, buzz_y=yb, buzz_T=T_b, buzz_thr=thr,
             src_z=z_src, src_y=y_src, T_src=T_src,
             p4_z=z4, p4_y=y4)
    json.dump(R, open(os.path.join(ART, "results.json"), "w"), indent=2)
    for k2, v in R.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"{k2:38s} acc={v['accuracy']*100:5.1f} f1={v['f1']*100:5.1f} "
                  f"auc={v['auc']*100:5.1f} ece={v['ece']:.3f}")


if __name__ == "__main__":
    main()
