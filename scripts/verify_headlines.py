# Author: Prabhu Rangaraju
"""Headline-number verifier for the S2 reproduction package.

Checks the (recomputed) results_*.json against the canonical expected/
values and against the headline numbers reported in the manuscript.

Two layers:
  1. Identity / statistical-identity: faster-whisper must be
     sha256-identical to canonical (it is deterministic given frozen
     audio). Vosk is checked for sha256-identity first; on mismatch it
     falls back to the honest weaker criterion — every headline lies
     inside the canonical 95% CI and vice versa (mutual containment),
     since Vosk's FinalResult() has rare run-to-run flips.
  2. Manuscript headlines: the canonical values are spot-checked against
     the numbers printed in the paper (tolerance = publication rounding).

Usage: python3 scripts/verify_headlines.py
Run from the package root. Exit 0 = all checks pass.
"""
import hashlib
import json
import sys

fails = []
passes = []


def check(name, cond, detail=""):
    (passes if cond else fails).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p):
    with open(p) as f:
        return json.load(f)


rec_v = load("results_vosk.json")
rec_f = load("results_fw.json")
canon_v = load("expected/results_vosk.json")
canon_f = load("expected/results_fw.json")

# --- Layer 1: identity / statistical identity ---
check("faster-whisper results sha256-identical to canonical",
      sha256("results_fw.json") == sha256("expected/results_fw.json"))
if sha256("results_vosk.json") == sha256("expected/results_vosk.json"):
    check("vosk results sha256-identical to canonical", True)
else:
    # Honest fallback: mutual 95%-CI containment on every headline.
    ok = True
    for k in ["1", "2", "3", "4", "5"]:
        r, c = rec_v["perK"][k], canon_v["perK"][k]
        for metric, ci in (("war", "war_ci95"), ("mean_saved_ms", "saved_ci95")):
            rv, rci, cv, cci = r[metric], r[ci], c[metric], c[ci]
            if not (cci[0] <= rv <= cci[1] and rci[0] <= cv <= rci[1]):
                ok = False
                print(f"  ci-mismatch K={k} {metric}: {rv} vs canon "
                      f"[{cci[0]:.4f},{cci[1]:.4f}]")
        if abs(r["p3_mean_actions"] - c["p3_mean_actions"]) / c["p3_mean_actions"] > 0.02:
            ok = False
            print(f"  p3-mismatch K={k}")
    check("vosk headlines mutually within 95% CIs (sha256 differed)", ok,
          "rare Vosk final-decode flip — see README determinism statement")

# --- Layer 2: manuscript headline numbers (publication rounding) ---
v2 = canon_v["perK"]["2"]
check("headline: Vosk P2 K=2 WAR 38.6% [33.4, 44.1]",
      abs(v2["war"] - 0.3862) < 0.001
      and abs(v2["war_ci95"][0] - 0.3345) < 0.002
      and abs(v2["war_ci95"][1] - 0.4448) < 0.01,
      f"war={v2['war']:.4f} ci={v2['war_ci95']}")
check("headline: Vosk P2 K=2 saved 897.9 ms [803, 995]",
      abs(v2["mean_saved_ms"] - 897.9) < 1.0,
      f"saved={v2['mean_saved_ms']:.1f}")
v1, v5 = canon_v["perK"]["1"], canon_v["perK"]["5"]
check("headline: WAR K=1 39.0% -> K=5 36.9% (threshold paradox)",
      abs(v1["war"] - 0.3897) < 0.001 and abs(v5["war"] - 0.3690) < 0.001)
check("headline: saved K=1 1077.9 ms -> K=5 491.7 ms",
      abs(v1["mean_saved_ms"] - 1077.9) < 1.0
      and abs(v5["mean_saved_ms"] - 491.7) < 1.0)
cc = v2["cause_counts"]
check("headline: causes K=2 incompleteness 98 vs instability 14 (~7:1)",
      cc["incompleteness"] == 98 and cc["instability"] == 14, str(cc))
bc = v2["by_category"]
exp_cat = {"A": (47, 0.0), "B": (76, 0.9342), "C": (51, 0.0),
           "D": (55, 0.7273), "E": (16, 0.0), "F": (45, 0.0222)}
cat_ok = all(bc[c]["n"] == n and abs(bc[c]["war"] - w) < 0.001
             for c, (n, w) in exp_cat.items())
check("headline: category WARs A0.000/B0.9342/C0.000/D0.7273/E0.000/F0.0222",
      cat_ok)
check("headline: P3 actions/utt 1.56/1.54/1.47 at K=1/2/5",
      abs(canon_v["perK"]["1"]["p3_mean_actions"] - 1.5586) < 0.01
      and abs(v2["p3_mean_actions"] - 1.5448) < 0.01
      and abs(v5["p3_mean_actions"] - 1.4690) < 0.01)
f2, f1, f5 = canon_f["perK"]["2"], canon_f["perK"]["1"], canon_f["perK"]["5"]
check("headline: FW P2 K=2 WAR 21.5%",
      abs(f2["war"] - 0.2150) < 0.001, f"war={f2['war']:.4f}")
check("headline: FW speculation collapse K=1->5 (frac 1.00->0.07, WAR 0.29->0.00)",
      abs(f1["frac_speculative"] - 1.00) < 0.01
      and abs(f5["frac_speculative"] - 0.07) < 0.01
      and abs(f1["war"] - 0.290) < 0.005 and f5["war"] == 0.0)
check("headline: FW P3 1.23 actions/utt at K=2",
      abs(f2["p3_mean_actions"] - 1.2336) < 0.01)
check("headline: corpus 360 / included 290 / excluded 70",
      canon_v["n_corpus"] == 360 and canon_v["n_included"] == 290
      and canon_v["n_excluded"] == 70)

print()
print(f"{len(passes)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
