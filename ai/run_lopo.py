"""P4: the only well-posed source-disjoint protocol available.

Publisher-disjoint evaluation is meaningful only on a corpus whose sources are
NOT label-pure -- otherwise holding out a source holds out a label. By the
source-only diagnostic, BuzzFeed-Webis is the sole corpus of the four that
qualifies (0% pure sources; source-only accuracy equals the majority baseline).

We therefore run leave-one-publisher-out CV on BuzzFeed under two training
regimes: BuzzFeed-internal, and pooled external corpora (which is additionally
cross-corpus). Both are compared against the majority-class baseline.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import run_experiments as X
from run_experiments import DF, EMB, train, infer, fit_T, ece, metrics, ART


def lopo(train_pool_idx, buzz_idx, tag, epochs=8):
    """Leave-one-publisher-out over BuzzFeed."""
    pubs = sorted(DF.publisher.iloc[buzz_idx].unique())
    zs, ys, per = [], [], []
    for p in pubs:
        te = buzz_idx[(DF.publisher.iloc[buzz_idx] == p).to_numpy()]
        bz_tr = buzz_idx[(DF.publisher.iloc[buzz_idx] != p).to_numpy()]
        tr = np.concatenate([train_pool_idx, bz_tr]) if len(train_pool_idx) \
            else bz_tr
        m = train(tr, EMB, epochs=epochs)
        z, y, _ = infer(m, te, EMB)
        zs.append(z)
        ys.append(y)
        yy = y.astype(int)
        rec = {"publisher": p.replace("buzzfeed:", ""), "n": int(len(y)),
               "pos_rate": float(yy.mean())}
        if len(set(yy)) > 1:
            rec["auc"] = float(roc_auc_score(yy, z))
        per.append(rec)
        print(f"  [{tag}] {p:22s} n={len(y):4d} pos={yy.mean():.2f} "
              f"auc={rec.get('auc', float('nan')):.3f}", flush=True)

    z = np.concatenate(zs)
    y = np.concatenate(ys).astype(int)
    # pooled-fold metrics; threshold at the training prior, not a blind 0.5
    p = 1 / (1 + np.exp(-z))
    out = {
        "n": int(len(y)), "pos_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, (p > 0.5).astype(int))),
        "f1": float(f1_score(y, (p > 0.5).astype(int), zero_division=0)),
        "auc": float(roc_auc_score(y, p)),
        "ece": ece(p, y),
        "majority_baseline": float(max(y.mean(), 1 - y.mean())),
        "per_publisher": per,
        "mean_publisher_auc": float(np.mean([r["auc"] for r in per if "auc" in r])),
        "std_publisher_auc": float(np.std([r["auc"] for r in per if "auc" in r])),
    }
    return out


def main():
    buzz = X.rows("buzzfeed")
    ext = np.concatenate([X.rows("isot"), X.rows("politifact"), X.rows("gossipcop")])
    R = json.load(open(os.path.join(ART, "results.json")))

    print("[P4a] BuzzFeed-internal leave-one-publisher-out")
    R["P4_buzzfeed_lopo_internal"] = lopo(np.array([], dtype=int), buzz, "internal")

    print("[P4b] pooled-external + BuzzFeed, leave-one-publisher-out")
    R["P4_buzzfeed_lopo_pooled"] = lopo(ext, buzz, "pooled")

    json.dump(R, open(os.path.join(ART, "results.json"), "w"), indent=2)
    for k in ("P4_buzzfeed_lopo_internal", "P4_buzzfeed_lopo_pooled"):
        v = R[k]
        print(f"\n{k}: acc={v['accuracy']*100:.2f} (majority "
              f"{v['majority_baseline']*100:.2f}) auc={v['auc']*100:.2f} "
              f"mean-pub-auc={v['mean_publisher_auc']*100:.2f}"
              f"±{v['std_publisher_auc']*100:.2f}")


if __name__ == "__main__":
    main()
