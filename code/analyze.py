"""Analyze streaming-ASR policy outcomes.
Input: {engine}_events.jsonl + corpus.jsonl
Output: results_{engine}.json with metrics per K, per category, CIs.
Paired bootstrap (B=2000) over utterances for 95% CIs of WAR differences.
"""
import json, sys
import numpy as np
from parser_policy import parse, action_eq, simulate, is_word_prefix

ENGINE = sys.argv[1] if len(sys.argv) > 1 else "vosk"
KS = [1, 2, 3, 4, 5]
rng = np.random.default_rng(20261004)

corpus = {}
with open("corpus.jsonl") as f:
    for line in f:
        r = json.loads(line)
        corpus[r["id"]] = r
if ENGINE == "fw":
    # The faster-whisper run is a stratified 120-utterance subset check
    # (fw_subset.json); scope the corpus so n_corpus/n_excluded describe
    # the actual run rather than counting non-subset utterances as
    # excluded. Corpus order is preserved so the seeded bootstrap is
    # unaffected.
    subset_ids = set(json.load(open("fw_subset.json")))
    corpus = {uid: r for uid, r in corpus.items() if uid in subset_ids}

events = {}
with open(f"{ENGINE}_events.jsonl") as f:
    for line in f:
        e = json.loads(line)
        events.setdefault(e["id"], []).append(e)
for v in events.values():
    v.sort(key=lambda e: e["t_ms"])

results = {"engine": ENGINE, "n_corpus": len(corpus)}
included, excluded = [], []
# parser validation on finals
for uid, row in corpus.items():
    evs = events.get(uid, [])
    finals = [e for e in evs if e["kind"] == "final"]
    if not finals:
        excluded.append((uid, "no final event"))
        continue
    gt = {"intent": row["intent"], "slots": row["slots"]}
    p = parse(finals[-1]["text"])
    if p is None or not action_eq(p, gt):
        excluded.append((uid, f"final parse mismatch: {finals[-1]['text']!r} -> {p}"))
    else:
        included.append(uid)
results["n_included"] = len(included)
results["n_excluded"] = len(excluded)
results["excluded"] = excluded[:20]

def timeline(uid):
    # dedupe (parallel workers may rarely double-emit): first of each key wins
    seen = set()
    evs = []
    for e in events[uid]:
        k = (e["t_ms"], e["kind"], e["text"])
        if k not in seen:
            seen.add(k)
            evs.append(e)
    tl = [(e["t_ms"], e["text"]) for e in evs
          if e["kind"] in ("partial", "segment", "final") and e["text"].strip()]
    # ensure final present
    fin = [e for e in events[uid] if e["kind"] == "final"][0]
    if not tl or tl[-1][1] != fin["text"]:
        tl.append((fin["t_ms"], fin["text"]))
    return tl

# revision stats: non-prefix partials
rev_stats = {"n_with_revision": 0, "n_total": 0, "by_cat": {}}
for uid in included:
    row = corpus[uid]
    tl = timeline(uid)
    fin_text = tl[-1][1]
    rev = any(not is_word_prefix(txt, fin_text) for _, txt in tl[:-1])
    rev_stats["n_total"] += 1
    rev_stats["n_with_revision"] += rev
    c = row["category"]
    d = rev_stats["by_cat"].setdefault(c, {"n": 0, "rev": 0})
    d["n"] += 1
    d["rev"] += rev
results["revision_stats"] = rev_stats

def compute_perK(included_ids):
    perK = {}
    for K in KS:
        sims = {}
        for uid in included_ids:
            row = corpus[uid]
            gt = {"intent": row["intent"], "slots": row["slots"]}
            sims[uid] = simulate(timeline(uid), gt, K)
        wars, saveds, causes = [], [], []
        nact3, fw3, fc3 = [], [], []
        by_cat = {}
        for uid, s in sims.items():
            c = corpus[uid]["category"]
            d = by_cat.setdefault(c, {"n": 0, "wasted": 0, "saved": [],
                                      "wasted_list": [], "causes": []})
            d["n"] += 1
            w = s["p2"]["wasted"]
            wars.append(w); d["wasted"] += w; d["wasted_list"].append(w)
            sv = s["p2"]["saved_ms"] if s["p2"]["speculative"] else 0
            saveds.append(sv); d["saved"].append(sv)
            causes.append(s["p2"]["cause"]); d["causes"].append(s["p2"]["cause"])
            nact3.append(s["p3"]["n_actions"]); fw3.append(s["p3"]["first_wasted"])
            fc3.append(s["p3"]["final_correct"])
        wars = np.array(wars, float); saveds = np.array(saveds, float)
        # paired bootstrap CI for WAR
        B = 2000
        bs = rng.integers(0, len(wars), size=(B, len(wars)))
        war_boot = wars[bs].mean(axis=1)
        sv_boot = saveds[bs].mean(axis=1)
        def cat_block(c, d):
            wl = np.array(d["wasted_list"], float)
            bbs = rng.integers(0, len(wl), size=(B, len(wl)))
            wb = wl[bbs].mean(axis=1)
            return {"n": d["n"], "war": d["wasted"]/d["n"],
                    "war_ci95": [float(np.quantile(wb, .025)),
                                 float(np.quantile(wb, .975))],
                    "mean_saved_ms": float(np.mean(d["saved"])),
                    "cause_counts": {k: int(sum(1 for x in d["causes"] if x == k))
                                     for k in ("incompleteness", "instability")}}
        perK[K] = {
            "war": float(wars.mean()),
            "war_ci95": [float(np.quantile(war_boot, .025)), float(np.quantile(war_boot, .975))],
            "mean_saved_ms": float(saveds.mean()),
            "saved_ci95": [float(np.quantile(sv_boot, .025)), float(np.quantile(sv_boot, .975))],
            "median_saved_ms": float(np.median(saveds)),
            "frac_speculative": float(np.mean([s["p2"]["speculative"] for s in sims.values()])),
            "cause_counts": {k: int(sum(1 for c in causes if c == k)) for k in
                             ("incompleteness", "instability", None)},
            "p3_mean_actions": float(np.mean(nact3)),
            "p3_first_wasted_rate": float(np.mean(fw3)),
            "p3_final_correct_rate": float(np.mean(fc3)),
            "by_category": {c: cat_block(c, d) for c, d in by_cat.items()},
        }
    return perK

perK = compute_perK(included)
results["perK"] = perK
# fair engine comparison: Vosk restricted to the faster-whisper subset
try:
    with open("fw_subset.json") as f:
        fw_ids = json.load(f)
    sub = [u for u in included if u in set(fw_ids)]
    if ENGINE == "vosk" and sub:
        results["perK_fw_subset"] = compute_perK(sub)
        results["n_subset"] = len(sub)
except FileNotFoundError:
    pass

# field-level error profile at K=2 (which slot was wrong among wasted)
from collections import Counter
field_err = Counter()
for uid in included:
    row = corpus[uid]
    gt = {"intent": row["intent"], "slots": row["slots"]}
    s = simulate(timeline(uid), gt, 2)
    a = s["p2"]["action"]
    if s["p2"]["wasted"] and a is not None and a["intent"] == gt["intent"]:
        for k, v in gt["slots"].items():
            av = a["slots"].get(k)
            vs = sorted(v) if isinstance(v, list) else v
            avs = sorted(av) if isinstance(av, list) else av
            if avs != vs:
                field_err[k] += 1
    elif s["p2"]["wasted"]:
        field_err["<intent>"] += 1
results["field_error_profile_K2"] = dict(field_err)

# latency: audio durations
durs = []
with open("audio/synth_log.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] in corpus:
            durs.append(r["bytes"] / 32000.0)  # 16kHz * 2 bytes
results["mean_audio_s"] = float(np.mean(durs))

with open(f"results_{ENGINE}.json", "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps({k: v for k, v in results.items()
                  if k not in ("excluded", "perK")}, indent=1))
for K in KS:
    d = perK[K]
    print(f"K={K}: WAR={d['war']:.3f} [{d['war_ci95'][0]:.3f},{d['war_ci95'][1]:.3f}] "
          f"saved={d['mean_saved_ms']:.0f}ms spec_frac={d['frac_speculative']:.2f} "
          f"causes={d['cause_counts']} p3_actions={d['p3_mean_actions']:.2f}")
print("field errors K=2:", dict(field_err))
