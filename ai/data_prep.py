"""Build a unified fake-news corpus from ISOT, FakeNewsNet and BuzzFeed-Webis.

Emits a single parquet/csv frame with columns:
    text     - headline (common modality across all three corpora)
    body     - full article body where available ('' otherwise)
    label    - 1 = fake / unreliable, 0 = real / reliable
    corpus   - isot | politifact | gossipcop | buzzfeed
    publisher- source-level id used for source-disjoint splitting
"""
import os
import re
import glob
import html
import pandas as pd
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "artifacts")
os.makedirs(OUT, exist_ok=True)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def clean(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = URL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s)
    return s.strip()


def domain_of(url):
    m = re.match(r"https?://(?:www\.)?([^/]+)", str(url))
    return m.group(1).lower() if m else "unknown"


def load_isot():
    base = os.path.join(DATA, "isot", "News_Dataset")
    frames = []
    for fn, lab in (("Fake.csv", 1), ("True.csv", 0)):
        d = pd.read_csv(os.path.join(base, fn))
        d = d.rename(columns={"title": "text", "text": "body"})
        d["label"] = lab
        d["corpus"] = "isot"
        # ISOT has no publisher column; `subject` is the closest source-level
        # grouping (each subject is dominated by one outlet/desk).
        d["publisher"] = "isot:" + d["subject"].astype(str).str.lower()
        frames.append(d[["text", "body", "label", "corpus", "publisher"]])
    return pd.concat(frames, ignore_index=True)


def load_fakenewsnet():
    base = os.path.join(DATA, "fnn", "FakeNewsNet-master", "dataset")
    frames = []
    for fn, corpus, lab in (
        ("politifact_fake.csv", "politifact", 1),
        ("politifact_real.csv", "politifact", 0),
        ("gossipcop_fake.csv", "gossipcop", 1),
        ("gossipcop_real.csv", "gossipcop", 0),
    ):
        d = pd.read_csv(os.path.join(base, fn))
        out = pd.DataFrame({
            "text": d["title"],
            "body": "",                       # bodies require the Twitter crawler
            "label": lab,
            "corpus": corpus,
            "publisher": corpus + ":" + d["news_url"].map(domain_of),
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


VERACITY = {"mostly true": 0, "mostly false": 1, "mixture of true and false": 1}


def load_buzzfeed():
    ov = pd.read_csv(os.path.join(DATA, "buzzfeed_overview.csv"))
    ov = ov[ov["veracity"].isin(VERACITY)].copy()
    ov["label"] = ov["veracity"].map(VERACITY)
    rows = []
    adir = os.path.join(DATA, "buzzfeed", "articles")
    for _, r in ov.iterrows():
        p = os.path.join(adir, r["XML"])
        if not os.path.exists(p):
            continue
        try:
            root = ET.parse(p).getroot()
        except ET.ParseError:
            continue
        title = (root.findtext("title") or "")
        paras = [e.text or "" for e in root.findall("mainText")]
        if not paras:
            paras = [root.findtext("mainText") or ""]
        rows.append({
            "text": title,
            "body": " ".join(paras),
            "label": int(r["label"]),
            "corpus": "buzzfeed",
            "publisher": "buzzfeed:" + str(r["portal"]),
        })
    return pd.DataFrame(rows)


def build():
    parts = [load_isot(), load_fakenewsnet(), load_buzzfeed()]
    df = pd.concat(parts, ignore_index=True)
    df["text"] = df["text"].map(clean)
    df["body"] = df["body"].map(clean)
    df = df[df["text"].str.split().str.len() >= 4]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "corpus.csv"), index=False)
    print(df.groupby(["corpus", "label"]).size().unstack(fill_value=0))
    print("\ntotal:", len(df))
    print("\npublishers per corpus:")
    print(df.groupby("corpus")["publisher"].nunique())
    return df


if __name__ == "__main__":
    build()
