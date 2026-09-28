"""Precompute frozen-BERT token embeddings once for the whole corpus.

The encoder is frozen, so its output is identical on every epoch of every
protocol. Caching it turns each subsequent training run into a BiLSTM-only fit.
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(os.path.dirname(HERE), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXLEN, BATCH = 64, 256

SCRUB = ["reuters", "(reuters)", "washington", "associated press"]


def scrub(s):
    out = s
    for t in SCRUB:
        for v in (t, t.upper(), t.title()):
            out = out.replace(v, " ")
    return " ".join(out.split())


def run(texts, path):
    if os.path.exists(path):
        print("exists, skipping:", path)
        return
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEV).eval().half()
    n = len(texts)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                   shape=(n, MAXLEN, 768))
    msk = np.zeros((n, MAXLEN), dtype=np.uint8)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            b = [str(t) for t in texts[i:i + BATCH]]
            e = tok(b, truncation=True, padding="max_length",
                    max_length=MAXLEN, return_tensors="pt")
            ids = e["input_ids"].to(DEV)
            am = e["attention_mask"].to(DEV)
            h = bert(ids, attention_mask=am).last_hidden_state
            mm[i:i + len(b)] = h.cpu().numpy().astype(np.float16)
            msk[i:i + len(b)] = am.cpu().numpy().astype(np.uint8)
            if (i // BATCH) % 40 == 0:
                print(f"  {i}/{n}", flush=True)
    mm.flush()
    np.save(path.replace(".npy", "_mask.npy"), msk)
    print("wrote", path)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(ART, "corpus.csv")).fillna({"text": ""})
    run(df["text"].tolist(), os.path.join(ART, "emb.npy"))
    run([scrub(str(t)) for t in df["text"]], os.path.join(ART, "emb_scrub.npy"))
