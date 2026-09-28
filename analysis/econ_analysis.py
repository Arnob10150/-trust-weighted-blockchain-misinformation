"""Economic-parameter sensitivity (R8) and end-to-end system simulation (R6).

R8: sweep the four hand-set economic parameters (risk multiplier, reputation
cap rho, decay half-life, verifier reward split) and report the security-
relevant response of each, showing the qualitative properties are stable.

R6: an end-to-end run of the pipeline on BuzzFeed -- the honest corpus -- where
the classifier is near-chance. Items are routed publish / jury / quarantine by
the calibrated score; the jury adjudicates the escalated band. We trace the
false-publication rate against the human-review budget and contrast it with the
"trust the classifier" oracle policy, closing with the economic cost a residual
false publication incurs.
"""
import os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); ART = os.path.join(ROOT, "artifacts")
CH = json.load(open(os.path.join(ART, "chain.json")))
S = np.load(os.path.join(ART, "scores.npz"))
OUT = {}

# ---------------------------------------------------------------- R8 sensitivity
BASE, RHO, POT = CH["bond_pub_eth"] * 10, 14, CH["slashing"]["pot_eth"]  # B_base=1000
# risk multiplier: cost to sustain k=10 fresh identities scales linearly
c2c = lambda k, M: k * BASE * M
OUT["sens_risk"] = [{"M": M, "c2c10": c2c(10, M),
                     "ratio_to_M1": c2c(10, M) / c2c(10, 1)} for M in (1, 2, 3, 5)]
# reputation cap rho: newcomer/elite ratio == rho; elite bond = B_base/rho
OUT["sens_rho"] = [{"rho": r, "elite_bond": BASE / r, "entry_ratio": r,
                    "novice_bond": BASE} for r in (7, 14, 28)]
# decay half-life: influence pre-farmed then left idle erodes; report residual
# reputation after an idle gap of g days for several half-lives h (days)
def residual(P0, g, h):
    lam = math.log(2) / h
    return P0 * math.exp(-lam * g)
OUT["sens_decay"] = [{"half_life_d": h, "lambda_per_d": math.log(2) / h,
                      "resid_frac_90d": residual(1, 90, h)} for h in (30, 60, 120)]
# verifier reward split: per-verifier payout must exceed a verification cost c
OUT["sens_reward"] = [{"verifiers": v, "per_verifier_eth": POT / v}
                      for v in (1, 3, 5, 10)]

# ---------------------------------------------------------------- R6 end-to-end
# Honest corpus (BuzzFeed) scored by the pooled model. The framework routes the
# most-uncertain items to a reputation-weighted jury (which we credit with the
# true label) and auto-decides the rest at the F1-optimal threshold. We trace
# system decision accuracy AND false-publication rate against the review budget.
z, y = S["buzz_z"], S["buzz_y"].astype(int)
p = 1.0 / (1.0 + np.exp(-z))
base_rate = float(y.mean()); N = len(y)
maj = int(base_rate > 0.5)
# F1-optimal auto threshold from the score itself
grid = np.linspace(0.05, 0.95, 91)
from itertools import count
def f1(th):
    pred = (p > th).astype(int); tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return tp / (tp + 0.5 * (fp + fn) + 1e-9)
thr = float(grid[int(np.argmax([f1(t) for t in grid]))])
auto_pred = (p > thr).astype(int)
uncert = -np.abs(p - thr)                       # closest to boundary = escalate first
order = np.argsort(-uncert)                      # most uncertain first

clf_acc = float((auto_pred == y).mean())         # classifier alone (0% review)
curve = []
for budget in np.linspace(0, 1, 51):
    k = int(round(budget * N)); esc = set(order[:k].tolist())
    correct = sum((1 if i in esc else int(auto_pred[i] == y[i])) for i in range(N))
    # false publications: fake items auto-decided (not escalated) as real
    fp = sum(1 for i in range(N) if i not in esc and y[i] == 1 and auto_pred[i] == 0)
    curve.append((float(budget), correct / N, fp / N))
curve = np.array(curve)
OUT["e2e_curve"] = curve.tolist()
OUT["e2e_thr"] = thr; OUT["e2e_clf_acc"] = clf_acc
OUT["e2e_majority"] = max(base_rate, 1 - base_rate)
pts = {}
for budget in (0.0, 0.10, 0.30, 0.50):
    i = int(np.argmin(np.abs(curve[:, 0] - budget)))
    pts[f"{int(budget*100)}pct"] = dict(review=float(curve[i, 0]),
                                        acc=float(curve[i, 1]), false_pub=float(curve[i, 2]))
OUT["e2e_points"] = pts

# imperfect + adversarial jury: 5 voters, honest voter correct w.p. q, an
# adversarial fraction f always votes to pass fraud (label 0). Reputation-
# weighted majority (honest voters carry weight 1, adversaries weight 1 too, so
# this is a conservative equal-weight vote). Escalated item resolved by the
# jury's aggregate vote, not by the true label. Report accuracy at 30% budget.
rng = np.random.default_rng(42)
def jury_acc(budget, q, f, n_jurors=5, trials=200):
    k = int(round(budget * N)); esc = order[:k]
    accs = []
    for _ in range(trials):
        correct = 0
        for i in range(N):
            if i in set(esc.tolist()):
                n_adv = int(round(f * n_jurors))
                votes = []
                for j in range(n_jurors):
                    if j < n_adv:
                        votes.append(0)                      # adversary passes fraud
                    else:
                        truth = int(y[i])
                        votes.append(truth if rng.random() < q else 1 - truth)
                verdict = int(sum(votes) > n_jurors / 2)
                correct += int(verdict == y[i])
            else:
                correct += int(auto_pred[i] == y[i])
        accs.append(correct / N)
    return float(np.mean(accs))
OUT["e2e_jury"] = {
    "q0.8_f0.2_30pct": jury_acc(0.30, 0.8, 0.2),
    "q0.9_f0.1_30pct": jury_acc(0.30, 0.9, 0.1),
    "q0.65_f0.2_30pct": jury_acc(0.30, 0.65, 0.2),
    "oracle_30pct": pts["30pct"]["acc"],
}
print("== jury (imperfect/adversarial) ==", OUT["e2e_jury"])

OUT["e2e_econ"] = dict(bond_forfeit_eth=CH["bond_pub_eth"],
                       verifier_share_eth=CH["slashing"]["recipients"][0]["received_eth"],
                       rep_cut_pct=20)
OUT["meta"] = dict(N=N, base_rate=base_rate)
json.dump(OUT, open(os.path.join(ART, "econ.json"), "w"), indent=2)

print("== R8 sensitivity ==")
print(" risk M:", [(d["M"], d["c2c10"], round(d["ratio_to_M1"], 1)) for d in OUT["sens_risk"]])
print(" rho   :", [(d["rho"], round(d["elite_bond"], 1), d["entry_ratio"]) for d in OUT["sens_rho"]])
print(" decay :", [(d["half_life_d"], round(d["resid_frac_90d"], 3)) for d in OUT["sens_decay"]])
print(" reward:", [(d["verifiers"], round(d["per_verifier_eth"], 1)) for d in OUT["sens_reward"]])
print("== R6 end-to-end ==")
print(f" N={N} base_rate={base_rate:.3f} majority={max(base_rate,1-base_rate):.3f} thr={thr:.2f}")
print(f" classifier alone (0% review) acc={clf_acc:.3f}")
for k, v in pts.items():
    print(f" review {k:5s}: acc={v['acc']:.3f} false_pub={v['false_pub']:.3f}")
print("wrote artifacts/econ.json")
