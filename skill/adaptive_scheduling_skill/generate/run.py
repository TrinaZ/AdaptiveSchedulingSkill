"""CLI of the Generate skill: reads one JSON record, writes one JSON record (see ../schemas and SKILL.md)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rec = json.load(open(a.inp))
out = core.generate(rec)
json.dump(out, open(a.out, "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k in ("type", "alarm", "change_type", "adequate", "headroom", "decision", "llm_calls")}, default=float))
