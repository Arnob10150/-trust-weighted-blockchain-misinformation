"""Camera-ready experiments answering the CSDE 2026 reviewer (Review-2).

1. Pipeline accounting   : per-corpus counts after every cleaning step, label
                           harmonization, exact and near-duplicate removal.
2. Shortcut probes       : how much of the ISOT label is recoverable from
                           writing style, timestamps, URLs/boilerplate, and
                           near-duplicate memorization -- none of which is
                           veracity.
3. Identical-condition   : lightweight, frozen-transformer, fine-tuned
   baseline comparison     transformer and our BERT-BiLSTM, all on the same
                           P1 split (seed 42), same 90/10 train/validation
                           carve-out, same 64-token headline input, three
                           seeds, on raw and on metadata-free headlines, and
                           under ISOT -> BuzzFeed transfer.

Writes artifacts/camera_ready.json. Runs in ~1 h on one RTX 3060.
"""
import os, re, html, json, time, math
import numpy as np, pandas as pd
from xml.etree import ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch, torch.nn as nn
from transformers import (AutoTokenizer, AutoModel,
                          AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data"); ART = os.path.join(ROOT, "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED, SEEDS, MAXLEN = 42, (42, 7, 1234), 64
OUT = os.path.join(ART, "camera_ready.json")
PRED = os.path.join(ART, "camera_ready_preds.npz")
URL_RE = re.compile(r"https?://\S+|www\.\S+"); WS_RE = re.compile(r"\s+")


def clean(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s); s = URL_RE.sub(" ", s); s = WS_RE.sub(" ", s)
    return s.strip()


TAG_RE = re.compile(r"[\(\[][^\)\]]{0,25}[\)\]]")
AGENCY_RE = re.compile(r"\b(reuters|associated press|ap|afp|breaking|exclusive|"
                       r"watch|video|videos|tweets?|images?|details)\b")


def metadata_free(s):
    """Keep only the claim's wording: lower-case, drop bracketed media tags,
    agency/format tokens, punctuation, digits-as-format and non-ASCII."""
    s = TAG_RE.sub(" ", s)
    s = s.encode("ascii", "ignore").decode().lower()
    s = AGENCY_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return WS_RE.sub(" ", s).strip()


# --------------------------------------------------------------------------
# 1. data with full accounting
# --------------------------------------------------------------------------
def domain_of(url):
    m = re.match(r"(?:https?://)?(?:www\.)?([^/]+)", str(url))
    return m.group(1).lower() if m else "unknown"


def raw_frames():
    base = os.path.join(DATA, "isot", "News_Dataset"); fr = []
    for fn, lab in (("Fake.csv", 1), ("True.csv", 0)):
        d = pd.read_csv(os.path.join(base, fn))
        fr.append(pd.DataFrame({"raw": d["title"], "body": d["text"], "date": d["date"],
                                "url": "", "label": lab, "corpus": "isot",
                                "publisher": "isot:" + d["subject"].astype(str).str.lower()}))
    fb = os.path.join(DATA, "fnn", "FakeNewsNet-master", "dataset")
    for fn, c, lab in (("politifact_fake.csv", "politifact", 1), ("politifact_real.csv", "politifact", 0),
                       ("gossipcop_fake.csv", "gossipcop", 1), ("gossipcop_real.csv", "gossipcop", 0)):
        d = pd.read_csv(os.path.join(fb, fn))
        fr.append(pd.DataFrame({"raw": d["title"], "body": "", "date": "", "url": d["news_url"].fillna(""),
                                "label": lab, "corpus": c,
                                "publisher": c + ":" + d["news_url"].map(domain_of)}))
    ov = pd.read_csv(os.path.join(DATA, "buzzfeed_overview.csv"))
    ov_all = len(ov)
    VER = {"mostly true": 0, "mostly false": 1, "mixture of true and false": 1}
    rows = []
    for _, r in ov[ov["veracity"].isin(VER)].iterrows():
        p = os.path.join(DATA, "buzzfeed", "articles", r["XML"])
        if not os.path.exists(p):
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        rows.append({"raw": root.findtext("title") or "", "body": "", "date": "",
                     "url": r["url"], "label": VER[r["veracity"]], "corpus": "buzzfeed",
                     "publisher": "buzzfeed:" + str(r["portal"])})
    fr.append(pd.DataFrame(rows))
    bz_veracity = ov["veracity"].value_counts().to_dict()
    return pd.concat(fr, ignore_index=True), ov_all, bz_veracity


def accounting(df, ov_all, bz_ver):
    A = {"buzzfeed_overview_rows": ov_all, "buzzfeed_veracity": bz_ver}
    order = ["isot", "gossipcop", "buzzfeed", "politifact"]
    df = df.copy(); df["text"] = df["raw"].map(clean)
    A["raw"] = df.corpus.value_counts().reindex(order).to_dict()
    short = df.text.str.split().str.len() < 4
    A["removed_short"] = df[short].corpus.value_counts().reindex(order, fill_value=0).to_dict()
    df = df[~short]
    within = df.duplicated(subset=["corpus", "text"])
    A["removed_dup_within"] = df[within].corpus.value_counts().reindex(order, fill_value=0).to_dict()
    df = df[~within]
    cross = df.duplicated(subset=["text"])            # first occurrence kept
    A["removed_dup_cross"] = df[cross].corpus.value_counts().reindex(order, fill_value=0).to_dict()
    # label-conflicting duplicates: same headline, different labels
    g = df.groupby("text").label.nunique()
    A["conflicting_label_headlines"] = int((g > 1).sum())
    df = df[~cross]
    A["final"] = df.corpus.value_counts().reindex(order).to_dict()
    A["final_total"] = int(len(df))
    return A


# --------------------------------------------------------------------------
# 2. shortcut probes on the P1 split
# --------------------------------------------------------------------------
def style_feats(t):
    w = t.split(); n = max(len(t), 1); nw = max(len(w), 1)
    return [len(t), len(w), sum(c.isupper() for c in t) / n,
            sum(x.isupper() and len(x) > 1 for x in w) / nw,
            sum(x[:1].isupper() for x in w) / nw,
            t.count("!"), t.count("?"), t.count(":"), t.count('"') + t.count("'"),
            t.count("(") + t.count("["), sum(c.isdigit() for c in t),
            sum(ord(c) > 127 for c in t), int(bool(TAG_RE.search(t)))]


MONTHS = "January February March April May June July August September October November December".split()


def date_format(d):
    d = str(d)
    if re.fullmatch(r"(%s) \d{1,2}, \d{4} " % "|".join(MONTHS), d): return "full+space"
    if re.fullmatch(r"(%s) \d{1,2}, \d{4}" % "|".join(MONTHS), d): return "full"
    if re.fullmatch(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", d): return "abbrev"
    if re.fullmatch(r"\d{1,2}-[A-Za-z]{3}-\d{2}", d): return "dd-mon-yy"
    if d.startswith("http"): return "url"
    return "other"


def lookup_acc(ktr, ytr, kte, yte):
    tab = pd.Series(ytr).groupby(np.asarray(ktr)).mean().gt(0.5).astype(int)
    maj = int(np.mean(ytr) > 0.5)
    p = pd.Series(kte).map(tab).fillna(maj).values
    return float(accuracy_score(yte, p))


def probes(isot, tr, te, buzz):
    P = {}
    ytr, yte = isot.label.values[tr], isot.label.values[te]
    # (a) writing style only -- no words
    X = np.array([style_feats(t) for t in isot.text])
    hg = HistGradientBoostingClassifier(random_state=SEED).fit(X[tr], ytr)
    p = hg.predict_proba(X[te])[:, 1]
    P["style_only"] = {"acc": accuracy_score(yte, p > .5), "auc": roc_auc_score(yte, p)}
    upper = X[:, 3] > 0.5
    P["allcaps_headline_rate"] = {"fake": float(upper[isot.label.values == 1].mean()),
                                  "real": float(upper[isot.label.values == 0].mean())}
    tag = X[:, 12] > 0
    P["bracket_tag_rate"] = {"fake": float(tag[isot.label.values == 1].mean()),
                             "real": float(tag[isot.label.values == 0].mean())}
    # (b) timestamps: format string alone, and date value alone
    fmt = isot.date.map(date_format).values
    P["date_format_only_acc"] = lookup_acc(fmt[tr], ytr, fmt[te], yte)
    P["date_format_by_label"] = pd.crosstab(fmt, isot.label.values).to_dict()
    dt = pd.to_datetime(isot.date.str.strip(), errors="coerce", format="mixed")
    ok = dt.notna().values
    ordv = np.where(ok, (dt - pd.Timestamp("2015-01-01")).dt.days.fillna(-1).values, -1).reshape(-1, 1)
    hg2 = HistGradientBoostingClassifier(random_state=SEED).fit(ordv[tr], ytr)
    p = hg2.predict_proba(ordv[te])[:, 1]
    P["date_value_only"] = {"acc": accuracy_score(yte, p > .5), "auc": roc_auc_score(yte, p)}
    P["date_range"] = {
        "fake": [str(dt[isot.label == 1].min().date()), str(dt[isot.label == 1].max().date())],
        "real": [str(dt[isot.label == 0].min().date()), str(dt[isot.label == 0].max().date())]}
    # (c) URL / boilerplate cues in the (unscored) body
    body = isot.body.fillna("").astype(str)
    reut = body.str.contains(r"\(Reuters\)", regex=True).values.astype(int)
    P["body_reuters_tag_acc"] = lookup_acc(reut[tr], ytr, reut[te], yte)   # tag present -> real
    link = body.str.contains(r"https?://|pic\.twitter\.com|@\w+", regex=True).values.astype(int)
    P["body_link_rate"] = {"fake": float(link[isot.label.values == 1].mean()),
                           "real": float(link[isot.label.values == 0].mean())}
    # (d) near-duplicates: nearest training headline for each test headline
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr = vec.fit_transform(isot.text.values[tr]); Xte = vec.transform(isot.text.values[te])
    sims, nn_lab = np.zeros(len(te)), np.zeros(len(te), int)
    for i in range(0, len(te), 1000):
        S = (Xte[i:i + 1000] @ Xtr.T).toarray()
        j = S.argmax(1); sims[i:i + 1000] = S[np.arange(len(j)), j]; nn_lab[i:i + 1000] = ytr[j]
    P["near_dup_frac"] = {str(t): float((sims >= t).mean()) for t in (0.6, 0.7, 0.8, 0.9)}
    P["nn_label_copy_acc"] = float(accuracy_score(yte, nn_lab))
    P["nn_label_copy_acc_ge0.8"] = float(accuracy_score(yte[sims >= .8], nn_lab[sims >= .8])) if (sims >= .8).any() else None
    Xb = vec.transform(buzz.text.values); sb = np.zeros(len(buzz))
    for i in range(0, len(buzz), 1000):
        sb[i:i + 1000] = (Xb[i:i + 1000] @ Xtr.T).toarray().max(1)
    P["buzz_near_dup_frac_0.8"] = float((sb >= .8).mean())
    # URL tokens alone on GossipCop (the URL path, not just the domain)
    return P, sims


def gossip_url_probe(raw):
    g = raw[raw.corpus == "gossipcop"].copy()
    g = g[g.url.str.len() > 0].reset_index(drop=True)
    rs = np.random.RandomState(SEED); pm = rs.permutation(len(g)); k = int(.8 * len(g))
    toks = g.url.str.replace(r"[^A-Za-z0-9]+", " ", regex=True).str.lower()
    v = TfidfVectorizer(min_df=2).fit(toks.values[pm[:k]])
    lr = LogisticRegression(max_iter=2000).fit(v.transform(toks.values[pm[:k]]), g.label.values[pm[:k]])
    p = lr.predict_proba(v.transform(toks.values[pm[k:]]))[:, 1]
    y = g.label.values[pm[k:]]
    return {"acc": accuracy_score(y, p > .5), "auc": roc_auc_score(y, p),
            "majority": float(max(y.mean(), 1 - y.mean()))}


# --------------------------------------------------------------------------
# 3. models under identical conditions
# --------------------------------------------------------------------------
def ev(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p)
    return {"acc": float(accuracy_score(y, p > .5)), "auc": float(roc_auc_score(y, p)),
            "f1m": float(f1_score(y, p > .5, average="macro"))}


def tfidf_lr(tr_t, ytr, tests, char=False):
    v = (TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200000, sublinear_tf=True)
         if char else TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000))
    lr = LogisticRegression(max_iter=2000, C=4.0).fit(v.fit_transform(tr_t), ytr)
    return [lr.predict_proba(v.transform(t))[:, 1] for t in tests]


@torch.no_grad()
def pooled(name, texts, bs=256):
    tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).to(DEV).eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
        h = m(**b).last_hidden_state; mk = b["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mk).sum(1) / mk.sum(1).clamp(min=1)).float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.vstack(out)


class BiLSTMHead(nn.Module):
    """Exactly the paper's head: 768->128 ReLU projection, 64-unit BiLSTM,
    dropout 0.3, linear logit.  BERT stays frozen."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 128); self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(torch.relu(self.proj(x)))
        return self.fc(self.drop(torch.cat((h[-2], h[-1]), 1))).squeeze(-1)


def bert_states(texts, tok, bert, bs=256):
    out = torch.empty((len(texts), MAXLEN, 768), dtype=torch.float16)   # preallocated, no concat copy
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            out[i:i + bs] = bert(**b).last_hidden_state.half().cpu()
    return out


def run_bilstm(H, tr, ytr, va, yva, tests, seed, epochs=8):
    """tests: list of (states, row-index or None). Indexes H per batch so the
    cached states are never copied."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = BiLSTMHead().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
    yt = torch.tensor(ytr, dtype=torch.float32); trt = torch.as_tensor(tr)
    best, state = 1e9, None
    for ep in range(epochs):
        m.train(); pm = torch.randperm(len(yt))
        for i in range(0, len(pm), 128):
            j = pm[i:i + 128]; opt.zero_grad()
            loss = lf(m(H[trt[j]].float().to(DEV)), yt[j].to(DEV)); loss.backward(); opt.step()
        vl = float(lf(torch.tensor(pred_head(m, H, va, logit=True)), torch.tensor(yva, dtype=torch.float32)))
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return [pred_head(m, Hs, ix) for Hs, ix in tests]


@torch.no_grad()
def pred_head(m, H, idx=None, logit=False):
    m.eval(); idx = torch.arange(len(H)) if idx is None else torch.as_tensor(idx)
    z = torch.cat([m(H[idx[i:i + 1024]].float().to(DEV)).cpu() for i in range(0, len(idx), 1024)]).numpy()
    return z if logit else 1 / (1 + np.exp(-z))


def finetune(name, tr_t, ytr, va_t, yva, tests, seed, epochs=3, lr=2e-5, bs=32):
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(DEV)
    enc = lambda t: tok(list(t), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    Etr = enc(tr_t); yt = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(yt) / bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(); lf = nn.BCEWithLogitsLoss()
    best, state = 1e9, None
    for ep in range(epochs):
        m.train(); pm = torch.randperm(len(yt))
        for i in range(0, len(pm), bs):
            j = pm[i:i + bs]; opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                z = m(input_ids=Etr["input_ids"][j].to(DEV), attention_mask=Etr["attention_mask"][j].to(DEV)).logits.squeeze(-1)
            loss = lf(z.float(), yt[j].to(DEV)); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        zv = ft_logits(m, enc(va_t))
        vl = float(lf(torch.tensor(zv), torch.tensor(yva, dtype=torch.float32)))
        print(f"      {name} seed{seed} ep{ep+1} val_loss={vl:.4f}", flush=True)
        if vl < best:
            best, state = vl, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    outs = [1 / (1 + np.exp(-ft_logits(m, enc(t)))) for t in tests]
    del m; torch.cuda.empty_cache()
    return outs


@torch.no_grad()
def ft_logits(m, E, bs=512):
    m.eval(); zs = []
    for i in range(0, len(E["input_ids"]), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            zs.append(m(input_ids=E["input_ids"][i:i + bs].to(DEV),
                        attention_mask=E["attention_mask"][i:i + bs].to(DEV)).logits.squeeze(-1).float().cpu())
    return torch.cat(zs).numpy()


def main():
    R = json.load(open(OUT)) if os.path.exists(OUT) else {}
    raw, ov_all, bz_ver = raw_frames()
    R["accounting"] = accounting(raw, ov_all, bz_ver)
    print(json.dumps(R["accounting"], indent=1))

    # ISOT and BuzzFeed exactly as in the paper's P1/P3 (same cleaning, same seed-42 split)
    isot = raw[raw.corpus == "isot"].copy(); isot["text"] = isot["raw"].map(clean)
    isot = isot[isot.text.str.split().str.len() >= 4].drop_duplicates(subset=["text"]).reset_index(drop=True)
    buzz = raw[raw.corpus == "buzzfeed"].copy(); buzz["text"] = buzz["raw"].map(clean)
    buzz = buzz[buzz.text.str.split().str.len() >= 4].drop_duplicates(subset=["text"]).reset_index(drop=True)
    n = len(isot); perm = np.random.RandomState(SEED).permutation(n); k = int(.8 * n)
    tr_all, te = perm[:k], perm[k:]
    nva = int(.1 * len(tr_all)); va, tr = tr_all[:nva], tr_all[nva:]
    print(f"ISOT {n}: train {len(tr)} val {len(va)} test {len(te)} | BuzzFeed {len(buzz)}")

    P, sims = probes(isot, tr_all, te, buzz)
    P["gossipcop_url_tokens"] = gossip_url_probe(raw)
    R["probes"] = P
    print(json.dumps(P, indent=1, default=str))
    json.dump(R, open(OUT, "w"), indent=2, default=str)

    y = isot.label.values; yb = buzz.label.values
    novel = sims < 0.8
    R["split"] = {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)), "buzz": int(len(buzz)),
                  "test_novel_frac": float(novel.mean())}
    preds = dict(np.load(PRED)) if os.path.exists(PRED) else {}

    for cond in ("raw", "mfree"):
        f = (lambda s: s) if cond == "raw" else metadata_free
        T = np.array([f(t) for t in isot.text]); TB = np.array([f(t) for t in buzz.text])
        jobs = []
        # lightweight, deterministic
        for nm, ch in (("TF-IDF word + LR", False), ("TF-IDF char + LR", True)):
            key = f"{cond}|{nm}|0"
            if key + "|te" not in preds:
                pt, pb = tfidf_lr(T[tr_all], y[tr_all], [T[te], TB], char=ch)
                preds[key + "|te"], preds[key + "|bz"] = pt, pb
        # frozen encoders + LR
        for nm, hf in (("MiniLM-L6 (frozen) + LR", "sentence-transformers/all-MiniLM-L6-v2"),
                       ("BERT (frozen) + LR", "bert-base-uncased")):
            key = f"{cond}|{nm}|0"
            if key + "|te" not in preds:
                E = pooled(hf, list(T) + list(TB)); Ei, Eb = E[:n], E[n:]
                lr = LogisticRegression(max_iter=3000, C=1.0).fit(Ei[tr_all], y[tr_all])
                preds[key + "|te"], preds[key + "|bz"] = lr.predict_proba(Ei[te])[:, 1], lr.predict_proba(Eb)[:, 1]
        np.savez(PRED, **preds)
        # ours: frozen BERT + BiLSTM
        if any(f"{cond}|BERT + Bi-LSTM (ours)|{s}|te" not in preds for s in SEEDS):
            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            bert = AutoModel.from_pretrained("bert-base-uncased").to(DEV).eval().half()
            H = bert_states(list(T), tok, bert); HB = bert_states(list(TB), tok, bert)
            del bert; torch.cuda.empty_cache()
            for s in SEEDS:
                t0 = time.time()
                pt, pb = run_bilstm(H, tr, y[tr], va, y[va], [(H, te), (HB, None)], s)
                preds[f"{cond}|BERT + Bi-LSTM (ours)|{s}|te"], preds[f"{cond}|BERT + Bi-LSTM (ours)|{s}|bz"] = pt, pb
                print(f"   ours seed {s} {time.time()-t0:.0f}s", flush=True)
            del H, HB
            np.savez(PRED, **preds)
        # fine-tuned transformers
        for nm, hf in (("DistilBERT (fine-tuned)", "distilbert-base-uncased"),
                       ("RoBERTa-base (fine-tuned)", "roberta-base")):
            for s in SEEDS:
                key = f"{cond}|{nm}|{s}"
                if key + "|te" in preds:
                    continue
                t0 = time.time()
                pt, pb = finetune(hf, T[tr], y[tr], T[va], y[va], [T[te], TB], s)
                preds[key + "|te"], preds[key + "|bz"] = pt, pb
                np.savez(PRED, **preds)
                print(f"   {nm} seed {s} {time.time()-t0:.0f}s", flush=True)

    # summarise
    table = {}
    for key in sorted({k.rsplit("|", 2)[0] for k in preds}):
        cond, nm = key.split("|")
        rs = [k.split("|")[2] for k in preds if k.startswith(key + "|") and k.endswith("|te")]
        M = {"p1": [], "p1_novel": [], "tf": []}
        for s in rs:
            pt, pb = preds[f"{key}|{s}|te"], preds[f"{key}|{s}|bz"]
            M["p1"].append(ev(y[te], pt)); M["p1_novel"].append(ev(y[te][novel], pt[novel])); M["tf"].append(ev(yb, pb))
        agg = {}
        for part, L in M.items():
            for met in ("acc", "auc", "f1m"):
                v = np.array([d[met] for d in L])
                agg[f"{part}_{met}"] = float(v.mean()); agg[f"{part}_{met}_sd"] = float(v.std())
        agg["n_seeds"] = len(rs)
        table.setdefault(cond, {})[nm] = agg
    R["models"] = table
    R["majority"] = {"p1": float(max(y[te].mean(), 1 - y[te].mean())), "tf": float(max(yb.mean(), 1 - yb.mean()))}
    json.dump(R, open(OUT, "w"), indent=2, default=str)
    for cond, T_ in table.items():
        print(f"\n== {cond}")
        for nm, a in T_.items():
            print(f"{nm:28s} P1 acc {a['p1_acc']*100:5.1f}±{a['p1_acc_sd']*100:.1f} AUC {a['p1_auc']*100:5.1f} | "
                  f"novel acc {a['p1_novel_acc']*100:5.1f} | TF acc {a['tf_acc']*100:5.1f} AUC {a['tf_auc']*100:5.1f} F1m {a['tf_f1m']*100:5.1f}")


if __name__ == "__main__":
    main()
