"""Baselines and BiLSTM ablation for reviewer response (R5).

Runs on the two protocols fully reproducible from local data: P1 (ISOT random
80/20, the in-domain convention) and transfer (train ISOT -> test BuzzFeed-Webis,
the honest out-of-source setting). The ISOT block is rebuilt exactly as in
data_prep so the seed-42 split matches run_experiments' P1. Every model is
scored on the identical split; the BERT+BiLSTM row is the paper's own P1 /
P3-buzzfeed result, reused verbatim for consistency.

Models: majority, source-only (publisher identity), TF-IDF+LogReg (lexical),
frozen-BERT[mean-pool]+LogReg (BERT-only ablation, no recurrence).
"""
import os, re, html, json
import numpy as np, pandas as pd
from xml.etree import ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from transformers import BertModel, BertTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); ART = os.path.join(ROOT, "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
URL_RE = re.compile(r"https?://\S+|www\.\S+"); WS_RE = re.compile(r"\s+")


def clean(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s); s = URL_RE.sub(" ", s); s = WS_RE.sub(" ", s)
    return s.strip()


def load_isot():
    base = os.path.join(DATA, "isot", "News_Dataset"); frames = []
    for fn, lab in (("Fake.csv", 1), ("True.csv", 0)):
        d = pd.read_csv(os.path.join(base, fn)).rename(columns={"title": "text", "text": "body"})
        d["label"] = lab; d["corpus"] = "isot"
        d["publisher"] = "isot:" + d["subject"].astype(str).str.lower()
        frames.append(d[["text", "label", "corpus", "publisher"]])
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.split().str.len() >= 4]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


VERACITY = {"mostly true": 0, "mostly false": 1, "mixture of true and false": 1}


def load_buzzfeed():
    ov = pd.read_csv(os.path.join(DATA, "buzzfeed_overview.csv"))
    ov = ov[ov["veracity"].isin(VERACITY)].copy(); ov["label"] = ov["veracity"].map(VERACITY)
    rows = []; adir = os.path.join(DATA, "buzzfeed", "articles")
    for _, r in ov.iterrows():
        p = os.path.join(adir, r["XML"])
        if not os.path.exists(p):
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        rows.append({"text": clean(root.findtext("title") or ""),
                     "label": int(r["label"]), "corpus": "buzzfeed",
                     "publisher": "buzzfeed:" + str(r["portal"])})
    df = pd.DataFrame(rows)
    df = df[df["text"].str.split().str.len() >= 4]
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


@torch.no_grad()
def bert_pool(texts, bs=256):
    """Mean-pooled frozen bert-base-uncased embedding, [N,768]."""
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEV).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=64,
                return_tensors="pt").to(DEV)
        h = bert(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        print(f"  bert {min(i+bs,len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(out)


def ev(y, p):
    y = np.asarray(y).astype(int)
    return (accuracy_score(y, (np.asarray(p) > 0.5).astype(int)),
            roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    isot = load_isot(); buzz = load_buzzfeed()
    print(f"ISOT {len(isot)} (fake {isot.label.mean():.3f}) | "
          f"BuzzFeed {len(buzz)} (fake {buzz.label.mean():.3f})")
    n = len(isot); perm = np.random.RandomState(SEED).permutation(n)
    k = int(0.8 * n); tr, te = perm[:k], perm[k:]
    ytr = isot.label.values[tr]; yte = isot.label.values[te]; yb = buzz.label.values
    maj = int(ytr.mean() > 0.5)
    R = {"n_isot": n, "n_buzz": len(buzz), "isot_pos": float(isot.label.mean()),
         "buzz_pos": float(buzz.label.mean()), "models": {}}

    # majority
    R["models"]["Majority"] = {
        "p1_acc": accuracy_score(yte, np.full(len(yte), maj)), "p1_auc": 0.5,
        "tf_acc": max(yb.mean(), 1 - yb.mean()), "tf_auc": 0.5}

    # source-only: per-publisher training-majority label; unseen -> global maj
    pub_maj = isot.iloc[tr].groupby("publisher").label.mean().gt(0.5).astype(int)
    def src_pred(df_rows):
        return df_rows.publisher.map(pub_maj).fillna(maj).values
    p1s = src_pred(isot.iloc[te])
    R["models"]["Source-only"] = {
        "p1_acc": accuracy_score(yte, p1s),
        "p1_auc": roc_auc_score(yte, p1s) if len(set(yte)) > 1 else float("nan"),
        "tf_acc": accuracy_score(yb, np.full(len(yb), maj)), "tf_auc": 0.5}

    # TF-IDF + LogReg
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    Xtr = vec.fit_transform(isot.text.values[tr])
    lr = LogisticRegression(max_iter=1000, C=4.0).fit(Xtr, ytr)
    p1 = lr.predict_proba(vec.transform(isot.text.values[te]))[:, 1]
    pb = lr.predict_proba(vec.transform(buzz.text.values))[:, 1]
    a1, u1 = ev(yte, p1); ab, ub = ev(yb, pb)
    R["models"]["TF-IDF + LogReg"] = {"p1_acc": a1, "p1_auc": u1, "tf_acc": ab, "tf_auc": ub}

    # frozen-BERT mean-pool + LogReg  (BERT-only, no recurrence)
    allt = list(isot.text.values) + list(buzz.text.values)
    E = bert_pool(allt); Ei, Eb = E[:n], E[n:]
    lrb = LogisticRegression(max_iter=2000, C=1.0).fit(Ei[tr], ytr)
    p1 = lrb.predict_proba(Ei[te])[:, 1]; pb = lrb.predict_proba(Eb)[:, 1]
    a1, u1 = ev(yte, p1); ab, ub = ev(yb, pb)
    R["models"]["BERT[pool] + LogReg"] = {"p1_acc": a1, "p1_auc": u1, "tf_acc": ab, "tf_auc": ub}

    # proposed BERT+BiLSTM: reuse the paper's own measured numbers
    res = json.load(open(os.path.join(ART, "results.json")))
    R["models"]["BERT + BiLSTM (ours)"] = {
        "p1_acc": res["P1_isot_random"]["accuracy"], "p1_auc": res["P1_isot_random"]["auc"],
        "tf_acc": res["P3_isot_to_buzzfeed"]["accuracy"], "tf_auc": res["P3_isot_to_buzzfeed"]["auc"]}

    json.dump(R, open(os.path.join(ART, "baselines.json"), "w"), indent=2)
    print(f"\n{'model':22s} {'P1 acc':>7} {'P1 AUC':>7} {'Tf acc':>7} {'Tf AUC':>7}")
    for k2, v in R["models"].items():
        print(f"{k2:22s} {v['p1_acc']*100:7.1f} {v['p1_auc']*100:7.1f} "
              f"{v['tf_acc']*100:7.1f} {v['tf_auc']*100:7.1f}")


if __name__ == "__main__":
    main()
