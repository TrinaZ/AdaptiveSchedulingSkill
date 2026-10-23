#!/usr/bin/env bash
# assemble_release.sh -- fill this release folder with the result files that live only in your working folder.
# Nothing is deleted or modified in the working folder; files are only COPIED into ./results and ./legacy_batch_regime.
#
# Usage (from the release root):   bash tools/assemble_release.sh /mnt/e/ForPhD/Code/创新点4/LAOSA_260826
set -euo pipefail
OLD="${1:?give the path of your working folder (the one with results/ and online_sla_bench.py)}"
REL="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REL"
copy() {  # copy matching files from OLD/results unless already present
  for f in "$OLD"/results/$1; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); [ -e "results/$b" ] || cp "$f" "results/$b"
  done
}
echo "== copying result files from $OLD/results"
copy 'online_full_*.json'                # main online run (7 objectives) + _llmdebug logs
copy 'eoh_*.json'                        # llm_eoh rows
copy 'backend_*.json'                    # Qwen-Max and deepseek-reasoner backends (Table S6)
copy 'window_transfer_300_rest.json'     # Table S13, tard/miss/premium
copy 'h7_*.json'                         # H7 / H7' (already included; copied only if missing)
copy 'template_leak.json'; copy 'h2h_heldout.json'

echo "== merging window-length files into results/window_transfer_300.json"
python3 - <<'EOF'
import json, os
parts = [p for p in ("results/window_transfer_300_mix_slo_wait.json", "results/window_transfer_300_rest.json") if os.path.exists(p)]
rows = [r for p in parts for r in json.load(open(p))]
if rows:
    json.dump(rows, open("results/window_transfer_300.json", "w"), indent=1)
    for p in parts: os.remove(p)
    print(f"   {len(rows)} rows from {len(parts)} files")
EOF
mv -f results/template_leak.json results/h2h_heldout.json legacy_batch_regime/ 2>/dev/null || true

echo "== copying legacy batch-pipeline code (post leak-fix versions) into legacy_batch_regime/"
for f in laosa.py head_to_head.py run_poms_experiments.py algorithm_archive.py llm_adapter.py; do
  [ -e "$OLD/$f" ] && cp "$OLD/$f" legacy_batch_regime/ || echo "   (missing in working folder: $f)"
done

echo "== checking that every result file the paper needs is present"
python3 - <<'EOF'
import glob
need = {
 "online_full_*.json (7 + 7 logs)": ("results/online_full_*.json", 14),
 "eoh_*.json (5 + 5 logs)": ("results/eoh_*.json", 10),
 "backend_qwen-max_*.json": ("results/backend_qwen-max_*.json", 5),
 "backend_ds-reasoner_*.json": ("results/backend_ds-reasoner_*.json", 5),
 "h7_nollm_*.json": ("results/h7_nollm_*.json", 3),
 "h7_llm_*.json (3 + 3 logs)": ("results/h7_llm_*.json", 6),
 "h7_backend_ds-reasoner_* (3 + 3 logs)": ("results/h7_backend_ds-reasoner_*.json", 6),
 "gp_sla.json, gp_h7.json": ("results/gp_*.json", 2),
 "longrun.json + reasoner/selection/kngov/robust": ("results/longrun*.json", 5),
 "online_governance_v3.json": ("results/online_governance_v3.json", 1),
 "gate_risk_coverage.json": ("results/gate_risk_coverage.json", 1),
 "prototype.json": ("results/prototype.json", 1),
 "adversarial_test.json": ("results/adversarial_test.json", 1),
 "horizon_gate.json": ("results/horizon_gate.json", 1),
 "window_transfer_300.json": ("results/window_transfer_300.json", 1),
 "diagnose_skip.json": ("results/diagnose_skip.json", 1),
}
ok = True
for name, (pat, n) in need.items():
    k = len(glob.glob(pat)); flag = "ok " if k >= n else "MISSING"
    if k < n: ok = False
    print(f"   {flag} {name}: {k}/{n}")
print("   all present" if ok else "   some files are missing: copy them into results/ by hand")
EOF

echo "== files in the working folder that are NOT part of the release -> NOT_INCLUDED.txt"
python3 - "$OLD" <<'EOF'
import os, sys
old = sys.argv[1]
have = set(os.listdir("results")) | set(os.listdir("code")) | set(os.listdir("legacy_batch_regime"))
lines = []
for d in ("", "results"):
    p = os.path.join(old, d)
    for f in sorted(os.listdir(p)):
        if os.path.isdir(os.path.join(p, f)): continue
        if f not in have: lines.append(os.path.join(d, f) if d else f)
open("NOT_INCLUDED.txt", "w").write("\n".join(lines) + "\n")
print(f"   {len(lines)} files listed (archive them privately; do not upload)")
EOF
echo "NOT_INCLUDED.txt" >> .gitignore

echo "== secret scan"
python3 tools/scan_secrets.py .
echo "Done. Review README.md, then: git init && git add . && git commit -m 'Release for TSC submission'"
