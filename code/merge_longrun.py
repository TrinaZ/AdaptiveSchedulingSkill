"""merge_longrun.py -- merge per-stream long-run files: python3 merge_longrun.py out.json in_s0.json in_s1.json ..."""
import sys, json
out, ins = sys.argv[1], sys.argv[2:]
cfg, streams = None, {}
for f in ins:
    d = json.load(open(f)); cfg = cfg or d["config"]
    for s in d["streams"]: streams[s["seed"]] = s
json.dump(dict(config=cfg, streams=[streams[k] for k in sorted(streams)]), open(out, "w"))
print(f"merged {len(streams)} streams -> {out}")
