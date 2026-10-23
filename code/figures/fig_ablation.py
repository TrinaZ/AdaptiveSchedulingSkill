import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("../results/longrun_ablation.json"))
V = [("full", "Full loop"), ("no_diagnose", "No Diagnose\n(always generate)"), ("no_gate", "No gate"), ("no_calib", "No calibration"),
     ("eb_gate", "EB gate\ninstead of t"), ("reselect", "No generation\n(re-selection)"), ("no_monitor", "No Monitor")]
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(7.0, 2.4))
full = np.array([r["full"] for r in D])
for i, (k, lab) in enumerate(V):
    v = np.array([r[k] for r in D]); m = v.mean()
    ax.bar(i, m, 0.62, color="#008300" if k == "full" else "#9aa0a6", linewidth=0)
    ax.scatter(i + np.linspace(-0.18, 0.18, len(v)), v, s=7, color="#202124", zorder=3)
    worse = int((v > full + 1e-9).sum())
    ax.text(i, 128, f"{m:.1f}" + ("" if k == "full" else f"\nworse {worse}/5"), ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(range(len(V))); ax.set_xticklabels([l for _, l in V], fontsize=6.6)
ax.set_ylim(0, 160); ax.set_yticks(range(0, 121, 20)); ax.set_ylabel("final cumulative regret")
ax.annotate("one stream at 367 \u2191", xy=(6, 150), fontsize=6, color="#202124", ha="center")
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
ax.set_title("Ablation of the governed loop (knowledge + Govern, no LLM): bars = mean over 5 streams, dots = streams", loc="left", fontsize=7.8)
fig.tight_layout(); fig.savefig("../figures/fig_ablation.pdf"); fig.savefig("../figures/fig_ablation.png", dpi=200)
