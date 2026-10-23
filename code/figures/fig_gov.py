import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
D = json.load(open("../results/online_governance_v3.json"))
B, O, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.05), gridspec_kw=dict(width_ratios=[1, 1.15]))
SC = {"S1_runtime_shift": "runtime shift", "S2_resource_drop": "capacity drop", "S4_novel_task": "new task type"}
M = sorted([m for m in D["motivation"] if m["degradation"] > 0.05], key=lambda m: m["degradation"])
lab = [f"{m['objective']} / {SC.get(m['scene'], m['scene'])}" for m in M]; v = [100 * m["degradation"] for m in M]
y = np.arange(len(M)); a.barh(y, v, height=0.55, color=B, linewidth=0)
a.set_xscale("log"); a.set_yticks(y); a.set_yticklabels(lab)
for i, (m, x) in enumerate(zip(M, v)):
    a.text(x * 1.08, i, f"+{x:.0f}%  ({m['pre_change_best']}$\\to${m['post_change_best']})", va="center", color=INK, fontsize=6.8)
a.set_xlim(8, 9000); a.set_xlabel("cost increase of the pre-change best rule (log)")
a.set_title("(a) Library rules degrade after a change", loc="left", fontsize=8, color=INK)
by = defaultdict(list)
for r in D["rows"]: by[r["policy"]].append(r)
P = [("ungated", "no gate"), ("hoeff_c0.0", "point\nestimate"), ("hoeff_c1.0", "Hoeffding\n(Eq. 1)"), ("paired_EB", "paired EB\n(Eq. 2)")]
dep = [100 * np.mean([x["deployed"] for x in by[p]]) for p, _ in P]
vio = [100 * (sum(x["violation"] for x in by[p] if x["deployed"]) / max(1, sum(x["deployed"] for x in by[p]))) for p, _ in P]
x = np.arange(len(P)); w = 0.36
b1 = b.bar(x - w / 2 - 0.01, dep, w, color=B, linewidth=0, label="candidate deployed (% of 168 episodes)")
b2 = b.bar(x + w / 2 + 0.01, vio, w, color=O, linewidth=0, label="deployments violating the bound (%)")
for xs, vs in ((x - w / 2 - 0.01, dep), (x + w / 2 + 0.01, vio)):
    for xi, vi in zip(xs, vs): b.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=6.8, color=INK)
b.set_xticks(x); b.set_xticklabels([l for _, l in P]); b.set_ylim(0, 125); b.set_yticks([0,20,40,60,80,100]); b.set_ylabel("%")
b.legend(frameon=False, fontsize=6.6, loc="upper right"); b.set_title("(b) Deploy decision under four gates", loc="left", fontsize=8, color=INK)
for ax in (a, b):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x" if ax is a else "y", color="#e4e3df", linewidth=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.2); fig.savefig("../figures/fig_gov.pdf"); fig.savefig("../figures/fig_gov.png", dpi=200)
print(dep, vio)
