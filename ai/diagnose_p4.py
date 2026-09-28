"""Break the P4 held-out result down by corpus to find out whether the
below-chance AUC is a real transfer effect or an artifact of the split."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")
DF = pd.read_csv(os.path.join(ART, "corpus.csv")).fillna({"text": ""})
S = np.load(os.path.join(ART, "scores.npz"))
R = json.load(open(os.path.join(ART, "results.json")))

import run_experiments as X

isot = X.rows("isot")
poli = X.rows("politifact")
goss = X.rows("gossipcop")
pool = np.concatenate([isot, poli, goss])
ptr, pte = X.stratified_publisher_holdout(pool)

z, y = S["p4_z"], S["p4_y"]
assert len(z) == len(pte), (len(z), len(pte))
sub = DF.loc[pte]

print(f"{'corpus':12s} {'n':>7s} {'pos%':>6s} {'acc':>7s} {'auc':>7s}")
for c, g in sub.groupby("corpus"):
    m = (sub.corpus == c).to_numpy()
    yy = y[m].astype(int)
    if len(set(yy)) < 2:
        print(f"{c:12s} {m.sum():7d} {yy.mean()*100:6.1f}  single-class")
        continue
    p = 1 / (1 + np.exp(-z[m]))
    print(f"{c:12s} {m.sum():7d} {yy.mean()*100:6.1f} "
          f"{accuracy_score(yy,(p>0.5).astype(int))*100:7.2f} "
          f"{roc_auc_score(yy,p)*100:7.2f}")

print("\nheld-out publishers per corpus:")
print(sub.groupby("corpus").publisher.nunique())
print("\ntrain composition:")
print(DF.loc[ptr].groupby(["corpus", "label"]).size().unstack(fill_value=0))
print("\ntest composition:")
print(sub.groupby(["corpus", "label"]).size().unstack(fill_value=0))

# Is ISOT's subject field simply a relabelling of the class?
print("\nISOT publisher x label:")
print(DF[DF.corpus == "isot"].groupby(["publisher", "label"]).size().unstack(fill_value=0))
