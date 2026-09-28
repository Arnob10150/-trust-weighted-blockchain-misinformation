"""How much of each corpus's label is predictable from its SOURCE alone?

A 'source-only' baseline assigns every item its publisher's majority label. It
reads no text at all. Where that baseline approaches 100%, the corpus admits no
source-disjoint split -- a text model can reach the same accuracy by learning
publisher style, and no preprocessing step can remove the shortcut.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")
DF = pd.read_csv(os.path.join(ART, "corpus.csv"), usecols=["corpus", "label", "publisher"])

rows = []
for c, g in DF.groupby("corpus"):
    maj = g.groupby("publisher").label.transform(lambda s: int(s.mean() > 0.5))
    acc = (maj == g.label).mean()
    pure = g.groupby("publisher").label.nunique().eq(1).mean()
    base = max(g.label.mean(), 1 - g.label.mean())
    rows.append({
        "corpus": c, "items": len(g), "sources": g.publisher.nunique(),
        "majority_baseline": base, "source_only_acc": acc,
        "pct_pure_sources": pure,
    })

t = pd.DataFrame(rows).sort_values("source_only_acc", ascending=False)
pd.set_option("display.width", 140)
print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

out = os.path.join(ART, "source_leakage.csv")
t.to_csv(out, index=False)
print("\nwrote", out)
