# Author: Prabhu Rangaraju
"""Figures for S2 manuscript. Reads results_vosk.json / results_fw.json.
All numbers come from the JSON; no literals invented here.
Fig 1: threshold operating curves (WAR vs mean latency saved) per engine.
Fig 2: WAR by utterance category at K=2 (Vosk), with 95% bootstrap CIs.
Fig 3: cause decomposition (incompleteness vs instability) stacked bars by category.
Fig 4: engine comparison (Vosk vs faster-whisper growing buffer) on the subset.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})

vosk = json.load(open("results_vosk.json"))
try:
    fw = json.load(open("results_fw.json"))
except FileNotFoundError:
    fw = None

CAT_ORDER = ["A", "B", "C", "D", "E", "F"]
CAT_LABEL = {"A": "A baseline", "B": "B correction", "C": "C negation",
             "D": "D late qualifier", "E": "E entity-dense", "F": "F confusable"}

# ---- Fig 1: operating curves ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for res, label, marker in ((vosk, "Vosk streaming", "o"),
                           (fw, "faster-whisper growing-buffer", "s")):
    if res is None:
        continue
    ks = sorted(int(k) for k in res["perK"])
    war = [res["perK"][str(k)]["war"] for k in ks]
    sv = [res["perK"][str(k)]["mean_saved_ms"] for k in ks]
    lo = [res["perK"][str(k)]["war_ci95"][0] for k in ks]
    hi = [res["perK"][str(k)]["war_ci95"][1] for k in ks]
    ax.errorbar(sv, war, yerr=[np.array(war)-np.array(lo),
                               np.array(hi)-np.array(war)],
                marker=marker, capsize=3, label=label)
    for k, x, y in zip(ks, sv, war):
        ax.annotate(f"K={k}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
ax.set_xlabel("Mean latency saved vs. wait-for-final (ms)")
ax.set_ylabel("Wasted-action rate")
ax.set_title("Speculative-commit operating curves")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig1_operating_curves.png")
print("fig1 saved")

# ---- Fig 2: WAR by category, Vosk K=2 (with 95% CIs) ----
fig, ax = plt.subplots(figsize=(5.6, 3.4))
d = vosk["perK"]["2"]["by_category"]
xs = np.arange(len(CAT_ORDER))
wars = [d[c]["war"] for c in CAT_ORDER]
ns = [d[c]["n"] for c in CAT_ORDER]
lo = [d[c]["war_ci95"][0] for c in CAT_ORDER]
hi = [d[c]["war_ci95"][1] for c in CAT_ORDER]
ax.bar(xs, wars, yerr=[np.array(wars)-np.array(lo), np.array(hi)-np.array(wars)],
       capsize=3, tick_label=[CAT_LABEL[c] for c in CAT_ORDER])
for x, w, n in zip(xs, wars, ns):
    ax.text(x, w + 0.02, f"{w:.2f}\n(n={n})", ha="center", fontsize=7)
ax.set_ylabel("Wasted-action rate (P2, K=2)")
ax.set_title("Wasted-action rate by utterance category")
ax.set_ylim(0, 1.06)
plt.xticks(rotation=12, ha="right")
fig.tight_layout(); fig.savefig("figures/fig2_war_by_category.png")
print("fig2 saved")

# ---- Fig 3: cause decomposition by category (stacked) ----
fig, ax = plt.subplots(figsize=(5.8, 3.6))
d = vosk["perK"]["2"]["by_category"]
inc = [d[c]["cause_counts"]["incompleteness"] for c in CAT_ORDER]
ins = [d[c]["cause_counts"]["instability"] for c in CAT_ORDER]
xs = np.arange(len(CAT_ORDER))
ax.bar(xs, inc, label="incompleteness (correction not heard yet)")
ax.bar(xs, ins, bottom=inc, label="instability (ASR revision)")
ax.set_xticks(xs); ax.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER],
                                      rotation=12, ha="right")
ax.set_ylabel("Wasted speculative commits (count)")
ax.set_title("Cause of waste by category (Vosk, P2, K=2)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig3_causes.png")
print("fig3 saved")

# ---- Fig 4: engine comparison on the stratified subset ----
# Fair comparison: Vosk restricted to the same 120-utterance subset.
if fw is not None and "perK_fw_subset" in vosk:
    vsub = vosk["perK_fw_subset"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    ks_v = sorted(int(k) for k in vsub)
    ks_f = sorted(int(k) for k in fw["perK"])
    axes[0].plot([vsub[str(k)]["war"] for k in ks_v], marker="o",
                 label="Vosk")
    axes[0].plot([fw["perK"][str(k)]["war"] for k in ks_f], marker="s",
                 label="faster-whisper")
    axes[0].set_xticks(range(len(ks_v))); axes[0].set_xticklabels([f"K={k}" for k in ks_v])
    axes[0].set_ylabel("Wasted-action rate"); axes[0].legend(fontsize=8)
    axes[0].set_title("(a) WAR vs stability threshold")
    axes[1].plot([vsub[str(k)]["mean_saved_ms"] for k in ks_v], marker="o",
                 label="Vosk")
    axes[1].plot([fw["perK"][str(k)]["mean_saved_ms"] for k in ks_f], marker="s",
                 label="faster-whisper")
    axes[1].set_xticks(range(len(ks_v))); axes[1].set_xticklabels([f"K={k}" for k in ks_v])
    axes[1].set_ylabel("Mean latency saved (ms)"); axes[1].legend(fontsize=8)
    axes[1].set_title("(b) Latency saved")
    fig.suptitle("Engine comparison: 120-utterance stratified subset\n"
                 f"({vosk.get('n_subset','?')} Vosk / {fw['n_included']} FW analyzable)")
    fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig("figures/fig4_engine_compare.png")
    print("fig4 saved")
print("done")
