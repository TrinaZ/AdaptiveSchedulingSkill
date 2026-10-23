import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
code = open("online_sla_bench.py").read().split("\n# ---- fail fast")[0]
sys.argv = ["b", "--src", ".", "--out", "/dev/null"]; B = {}; exec(code, B)
mk, simulate, objective, LIB = B["make_instance"], B["simulate"], B["objective"], B["LIB"]
E = {e["episode"]: e for e in json.load(open("../results/prototype.json"))["episodes"]}
def series(k, rule, o):
    e = E[k]; base = 7_000_000 + 100_000 * k; (s0, _), (s1, _) = e["pre"], e["post"]
    out = []
    for t in range(100):
        rs, R = mk(s0 if t < 40 else s1, base + t, 60, 1.0); out.append(objective(o, rs, simulate(rs, R, rule)))
    return np.array(out)
def llm_rule(pkg):
    ns = {"math": __import__("math"), "max": max, "min": min, "abs": abs}; exec(pkg["rule_source"], ns); f = ns["priority"]; p = pkg["params"]
    return lambda t, s: float(f(t, s, **p))
sm = lambda x, w=5: np.convolve(x, np.ones(w) / w, mode="same")
INK, MUT, B1, O1, G1 = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.linewidth": 0.6})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55))
# (a) deployment
k = 76; e = E[k]; o = e["post"][1]; pkg = e["skill_package"]
old = series(k, LIB[e["old_rule"]], o); new = series(k, llm_rule(pkg), o)
best = min(LIB, key=lambda n: series(k, LIB[n], o)[e["alarm"] + 1:].mean()); bl = series(k, LIB[best], o)
a = axs[0]; t = np.arange(100); post = t > e["alarm"]; a.set_ylim(-3, 105)
a.plot(t, sm(old), color=MUT, lw=1.2, label=f"incumbent ({e['old_rule']})")
a.plot(t[post], sm(bl)[post], color=B1, lw=1.2, label=f"best library rule ({best})")
a.plot(t[post], sm(new)[post], color=O1, lw=1.6, label="deployed LLM rule")
a.axvline(40, color=MUT, ls=":", lw=0.8); a.axvline(e["alarm"], color=INK, ls="--", lw=0.8)
a.text(1, 103, f"window 40: capacity drop (dotted)\nwindow {e['alarm']}: alarm (dashed),\n  labeled structural; LLM rule passes\n  the gate, expert approval, deployed", va="top", fontsize=6.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
a.set_title(f"(a) Episode {k}: capacity drop, wait SLO", loc="left", fontsize=8); a.set_xlabel("traffic window"); a.set_ylabel("excess waiting (5-window mean)")
a.legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)
# (b) fallback
k = 84; e = E[k]; o = e["post"][1]; ev = e["skill_package"]["evidence"]; fb = ev["fallback"]
old = series(k, LIB[e["old_rule"]], o); fbs = series(k, LIB[fb], o)
b = axs[1]; post = t >= 40; b.set_ylim(-1, 33)
b.plot(t[post], sm(old)[post], color=MUT, lw=1.2, label=f"incumbent ({e['old_rule']})")
b.plot(t[post], sm(fbs)[post], color=B1, lw=1.6, label=f"fallback ({fb}), review flag")
b.axvline(40, color=INK, ls="--", lw=0.8)
b.text(1, 32.5, f"window 40: SLA rewritten (declared)\nLLM candidate on 1,000 validation instances:\nmean {ev['candidate_mean']:.2f} vs {fb} {ev['library_means'][fb]:.2f}\npaired t bound > 0: rejected", va="top", fontsize=6.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=5)
b.set_title(f"(b) Episode {k}: SLA changed to tardiness", loc="left", fontsize=8); b.set_xlabel("traffic window"); b.set_ylabel("weighted tardiness (5-window mean)")
b.legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2); b.set_xlim(0, 100)
for ax in axs:
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.5); fig.savefig("../figures/fig_cases.pdf"); fig.savefig("../figures/fig_cases.png", dpi=200)
print("old/new post means:", old[e['alarm']:].mean())
print(pkg["rule_source"][:600]); print(pkg["params"])
