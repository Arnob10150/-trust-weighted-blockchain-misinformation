"""Print the identical-condition comparison from cached predictions
(artifacts/camera_ready_preds.npz) without re-training anything."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_camera_ready as R

raw, _, _ = R.raw_frames()
isot = raw[raw.corpus == "isot"].copy(); isot["text"] = isot["raw"].map(R.clean)
isot = isot[isot.text.str.split().str.len() >= 4].drop_duplicates(subset=["text"]).reset_index(drop=True)
buzz = raw[raw.corpus == "buzzfeed"].copy(); buzz["text"] = buzz["raw"].map(R.clean)
buzz = buzz[buzz.text.str.split().str.len() >= 4].drop_duplicates(subset=["text"]).reset_index(drop=True)
n = len(isot); perm = np.random.RandomState(R.SEED).permutation(n); k = int(.8 * n)
te = perm[k:]
y, yb = isot.label.values, buzz.label.values
preds = dict(np.load(R.PRED))
print(f"majority P1 {max(y[te].mean(), 1 - y[te].mean())*100:.1f}  transfer {max(yb.mean(), 1 - yb.mean())*100:.1f}")
for key in sorted({kk.rsplit("|", 2)[0] for kk in preds}):
    seeds = [kk.split("|")[2] for kk in preds if kk.startswith(key + "|") and kk.endswith("|te")]
    rows = [(R.ev(y[te], preds[f"{key}|{s}|te"]), R.ev(yb, preds[f"{key}|{s}|bz"])) for s in seeds]
    f = lambda part, m: (np.mean([r[part][m] for r in rows]) * 100, np.std([r[part][m] for r in rows]) * 100)
    print(f"{key:36s} n={len(seeds)} P1 acc {f(0,'acc')[0]:5.1f}±{f(0,'acc')[1]:.1f} AUC {f(0,'auc')[0]:5.1f} | "
          f"TF acc {f(1,'acc')[0]:5.1f}±{f(1,'acc')[1]:.1f} AUC {f(1,'auc')[0]:5.1f}±{f(1,'auc')[1]:.1f} F1m {f(1,'f1m')[0]:5.1f}")
