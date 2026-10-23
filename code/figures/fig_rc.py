import json, math, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
D = json.load(open("../results/gate_risk_coverage.json")); R = D["rows"]
col = {"point": "#9aa0a6", "t": "#2a78d6", "eb": "#eb6834"}; lab = {"point": "point estimate", "t": "paired t", "eb": "paired EB"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35))
ns = sorted({r["n"] for r in R})
for g in ("point", "t", "eb"):
    for tau, ls in ((0.02, "-"), (0.05, "--")):
        L = sorted([r for r in R if r["gate"] == g and r["tau"] == tau], key=lambda r: r["n"])
        kw = dict(color=col[g], ls=ls, lw=1.4 if ls == "-" else 1.0, marker="o", ms=2.5)
        axs[0].plot([r["n"] for r in L], [100 * r["coverage"] for r in L], label=f"{lab[g]}" if tau == 0.02 else None, **kw)
        axs[1].plot([r["n"] for r in L], [100 * r["risk"] for r in L], **kw)
        if tau == 0.02:
            lo, hi = zip(*[wilson(r["violations"], r["deployed"]) for r in L]); x = [r["n"] * (1 + 0.04 * (g == "t") - 0.04 * (g == "eb")) for r in L]
            axs[1].errorbar(x, [100 * r["risk"] for r in L], yerr=[[100 * (r["risk"] - l) for r, l in zip(L, lo)], [100 * (h - r["risk"]) for r, h in zip(L, hi)]],
                            fmt="none", ecolor=col[g], elinewidth=0.6, capsize=1.5, alpha=0.8)
        axs[2].plot([r["n"] for r in L], [100 * r["mean_rel"] for r in L], **kw)
for ax, t, yl in zip(axs, ["(a) Coverage: candidates deployed", "(b) Risk: deployments violating (95% CI)", "(c) Mean cost vs best library"],
                     ["% of episodes", "% of deployments", "%"]):
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlabel("validation instances n")
    ax.set_title(t, loc="left", fontsize=7.8); ax.set_ylabel(yl)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
axs[2].axhline(0, color="#52514e", lw=0.6)
axs[0].plot([], [], color="#52514e", ls="-", label=r"$\tau$ = 2%"); axs[0].plot([], [], color="#52514e", ls="--", label=r"$\tau$ = 5%")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=6.6, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(w_pad=1.0, rect=(0, 0.08, 1, 1)); fig.savefig("../figures/fig_rc.pdf"); fig.savefig("../figures/fig_rc.png", dpi=200)
