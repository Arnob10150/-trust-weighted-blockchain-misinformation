# Trust-Weighted Blockchain Framework: AI Risk Scoring and Bounded Sybil-Resistant Stake Economics

Code for *A Trust-Weighted Blockchain Framework Coupling AI-Driven Risk Scoring with Bounded
Sybil-Resistant Stake Economics* (IEEE CSDE 2026).

Arnob Aich Anurag, Shamiul Islam, MD.Nasif Rafidi, MD. Sadman Hossain, Faria Sultana, Fahad Khandoker, Shohana Khondoker

## What is here

```
ai/          risk scorer (frozen BERT + Bi-LSTM), evaluation protocols, leakage probes, baselines
chain/       ERC-1155 publication/escrow contract (Solidity) and Hardhat gas/burst benchmark
analysis/    economic sensitivity analysis of the bond curve
artifacts/   measured results (JSON/CSV) reported in the paper
```

| Script | Purpose |
|---|---|
| `ai/data_prep.py` | merges ISOT, FakeNewsNet (PolitiFact, GossipCop) and BuzzFeed-Webis; harmonizes labels; cleans and de-duplicates headlines |
| `ai/source_leakage.py` | source-only diagnostic (how much of the label the source field alone predicts) |
| `ai/cache_embeddings.py` | caches frozen BERT token states (needs a CUDA GPU) |
| `ai/run_experiments.py` | protocols P1–P3 and temperature calibration |
| `ai/run_lopo.py` | P4: leave-one-publisher-out cross-validation |
| `ai/run_baselines.py` | TF-IDF / BERT baseline ladder |
| `ai/run_camera_ready.py` | shortcut probes (style, timestamps, URLs, near-duplicates) and the identical-condition comparison of 7 detectors × 3 seeds, raw vs. metadata-free |
| `ai/summarize_camera_ready.py` | prints the comparison table from cached predictions |
| `chain/scripts/bench.js` | deploys the contract, measures gas, batching and the burst-detection path |
| `analysis/econ_analysis.py` | bond-curve and cost-to-corrupt sensitivity sweep |

## Key results

- The source field alone labels ISOT at 100%; writing style alone reaches 99.8%, the timestamp format 100%, and URL tokens alone reach 98.4% on GossipCop.
- In-domain accuracy (P1): 97.6% for BERT–Bi-LSTM and above 99.9% for fine-tuned RoBERTa. Under transfer to BuzzFeed every model falls below the 81.2% majority baseline.
- The bounded bond curve compresses the novice-to-elite ratio from 10,001× to 14.0×. `mintContent()` costs 169,638 gas, and batching 32 items cuts per-item gas by 35.5%.

## Data (not included)

Place the public datasets as follows:

```
data/isot/News_Dataset/{Fake,True}.csv              ISOT Fake News Dataset (Univ. of Victoria)
data/fnn/FakeNewsNet-master/dataset/*.csv           https://github.com/KaiDMML/FakeNewsNet
data/buzzfeed/articles/*.xml, data/buzzfeed_overview.csv   BuzzFeed-Webis Fake News Corpus 2016 (Zenodo)
```

## Reproduce

```bash
pip install torch transformers scikit-learn pandas numpy
python ai/data_prep.py && python ai/source_leakage.py
python ai/cache_embeddings.py            # GPU, ~6 GB of cached states
python ai/run_experiments.py && python ai/run_lopo.py
python ai/run_camera_ready.py            # ~1.5 h on one RTX 3060
cd chain && npm install && npx hardhat compile && npx hardhat run scripts/bench.js
```

All randomness is seeded (42; the comparison also uses seeds 7 and 1234).

## License

Code is released for academic use. Third-party datasets remain under their original licenses.
