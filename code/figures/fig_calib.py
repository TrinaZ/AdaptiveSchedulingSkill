import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("../results/gate_calibration.json")); R = [r for r in D["rows"] if r["gate"] != "ungated"]
col = {"t": "#2a78d6", "eb": "#eb6834"}; lab = {"t": "paired t", "eb": "paired EB"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, n in zip(axs, (200, 1000)):
    ds = sorted({r["delta"] for r in R})
    ax.plot(ds, [100 * d for d in ds], color="#52514e", lw=0.8, ls=":", label=r"nominal $\delta$ (bound)")
    for g in ("t", "eb"):
        L = sorted([r for r in R if r["n"] == n and r["gate"] == g], key=lambda r: r["delta"])
        x = [r["delta"] for r in L]
        ax.plot(x, [100 * r["viol_per_episode"] for r in L], color=col[g], marker="o", ms=3, lw=1.4, label=f"{lab[g]}: observed")
        ax.fill_between(x, [100 * r["random_viol_lo"] for r in L], [100 * r["random_viol_hi"] for r in L], color=col[g], alpha=0.13, lw=0,
                        label=f"random gate, coverage of {lab[g]} (95%)")
        ax.plot(x, [100 * r["random_viol_mean"] for r in L], color=col[g], lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlabel(r"nominal $\delta$ per episode")
    ax.set_title(f"({'ab'[n == 1000]}) {n} validation instances", loc="left", fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("violating deployments\n(% of episodes)")
axs[0].set_ylim(0, 30)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=6.6, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig("../figures/fig_calib.pdf"); fig.savefig("../figures/fig_calib.png", dpi=200)
