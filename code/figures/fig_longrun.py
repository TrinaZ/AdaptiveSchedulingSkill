import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open("../results/longrun.json"))["streams"]
SR = {x["seed"]: x for x in json.load(open("../results/longrun_reasoner.json"))["streams"]}
P = ["static", "reselect", "ungated", "skill"]
col = {"static": "#9aa0a6", "reselect": "#2a78d6", "ungated": "#eb6834", "skill": "#008300"}
lab = {"static": "static rule", "reselect": "re-selection only (no LLM)", "ungated": "Skill without gate", "skill": "Adaptive Scheduling Skill"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
T = len(S[0]["cum_regret"]["skill"]); t = np.arange(T)
for p in P:
    M = np.array([s["cum_regret"][p] for s in S]); m = M.mean(0)
    a.plot(t, m, color=col[p], lw=1.6 if p == "skill" else 1.1, ls="--" if p == "static" else "-", label=lab[p])
MR = np.array([SR[s["seed"]]["cum_regret"]["skill"] for s in S]).mean(0)
a.plot(t, MR, color="#7b3fa0", lw=1.6, label="Skill with deepseek-reasoner")
for k in range(1, 11):
    a.axvline(40 * k, color="#b9b7b1", lw=0.7, ls=(0, (3, 2)), zorder=0)
    a.plot([40 * k], [75], marker="v", ms=3.5, color="#52514e", clip_on=False, zorder=3)
a.set_ylim(0, 75); a.set_xlim(0, T)
a.annotate(f"static: {np.mean([s['cum_regret']['static'][-1] for s in S]):.0f} at end", xy=(T * 0.30, 71), fontsize=6.5, color="#52514e", va="top")
a.set_xlabel("traffic window (\u25bc and dashed lines: change points, every 40 windows)"); a.set_ylabel("cumulative normalized regret")
a.set_title("(a) Mean cumulative regret over 5 streams", loc="left", fontsize=8)
a.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(0.0, 0.92))
x = np.arange(len(S)); w = 0.16
for i, p in enumerate(P + ["reasoner"]):
    v = [(SR[s["seed"]]["cum_regret"]["skill"] if p == "reasoner" else s["cum_regret"][p])[-1] for s in S]
    b.bar(x + (i - 2) * w, v, w * 0.92, color="#7b3fa0" if p == "reasoner" else col[p], linewidth=0)
b.set_yscale("log"); b.set_xticks(x); b.set_xticklabels([f"stream {s['seed']+1}" for s in S])
b.set_ylabel("final regret (log)"); b.set_title("(b) Final regret per stream", loc="left", fontsize=8)
for ax in (a, b):
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
b.grid(axis="y", color="#e4e3df", lw=0.5, which="both"); b.set_axisbelow(True)
fig.tight_layout(w_pad=1.5); fig.savefig("../figures/fig_longrun.pdf"); fig.savefig("../figures/fig_longrun.png", dpi=200)
