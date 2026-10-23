"""
fix_runtime_leak.py -- remove access to the hidden TRUE runtime from every
generator path (template seeds, head-to-head pools, LLM prompts).

Background (audit 2026-10-15): instances.py sets
    estimated_runtime = runtime * U(0.6, 1.4)
Baselines and the LLM prompt for S3 use estimated_runtime, but
  * laosa._EOH_TEMPLATES (tier-adaptive population seeds) and
  * head_to_head.TEMPLATES (random / greedy / EoH-style seeds)
contain `task.runtime`, and the semantic single-call prompts in laosa.py and
run_poms_experiments.py list `task.runtime` as an available field.
All 93/93 distributional and 160/160 structural Beats-Best wins in
results/wgs_matrix_8seeds.json use task.runtime; with runtime replaced by
estimated_runtime, held-out BB drops to 0% on both tiers.

Same conventions as the project's other patch scripts:
  * every patch is an exact (old, new) pair that must occur exactly once;
  * default is a DRY RUN; pass --apply to write;
  * a .bak_runtime_leak backup is written before any file is modified.

Usage (from the repo root):
    python3 fix_runtime_leak.py          # dry run, reports [OK]/[FAIL]
    python3 fix_runtime_leak.py --apply  # write files
"""
import shutil
import sys
from pathlib import Path

PATCHES = {
    "laosa.py": [
        ('    ("def priority(task): return task.runtime ** {p}",',
         '    ("def priority(task): return task.estimated_runtime ** {p}",'),
        ('    ("def priority(task): return task.weight ** {w} * task.runtime ** {p}",',
         '    ("def priority(task): return task.weight ** {w} * task.estimated_runtime ** {p}",'),
        ('    ("def priority(task): return task.runtime / max(task.weight ** {w}, 1e-6)",',
         '    ("def priority(task): return task.estimated_runtime / max(task.weight ** {w}, 1e-6)",'),
        ('    ("def priority(task): return task.weight * task.runtime",',
         '    ("def priority(task): return task.weight * task.estimated_runtime",'),
        ('            + "- task.runtime (float)" + NL\n',
         ''),
    ],
    "head_to_head.py": [
        ('    ("def priority(task): return task.runtime ** {p}",',
         '    ("def priority(task): return task.estimated_runtime ** {p}",'),
        ('    ("def priority(task): return task.weight ** {w} * task.runtime ** {p}",',
         '    ("def priority(task): return task.weight ** {w} * task.estimated_runtime ** {p}",'),
        ('    ("def priority(task): return task.runtime / max(task.weight ** {w}, 1e-6)",',
         '    ("def priority(task): return task.estimated_runtime / max(task.weight ** {w}, 1e-6)",'),
        ('    ("def priority(task): return task.runtime * task.weight",',
         '    ("def priority(task): return task.estimated_runtime * task.weight",'),
    ],
    "run_poms_experiments.py": [
        ('        + "- task.runtime (float)" + NL\n',
         ''),
    ],
}


def main(apply: bool) -> int:
    texts, ok = {}, True
    for fname, patches in PATCHES.items():
        p = Path(fname)
        if not p.exists():
            print(f"[FAIL] {fname}: file not found"); ok = False; continue
        raw = p.read_bytes()
        crlf = b"\r\n" in raw
        text = raw.decode("utf-8")
        for old, new in patches:
            o = old.replace("\n", "\r\n") if crlf else old
            n_found = text.count(o)
            tag = "[OK]  " if n_found == 1 else "[FAIL]"
            if n_found != 1: ok = False
            print(f"{tag} {fname}: found {n_found}x  {old.strip()[:70]}")
        texts[fname] = (text, crlf)
    if not ok:
        print("\nNothing written: at least one patch did not match exactly once.")
        return 1
    if not apply:
        print("\nDry run only. Re-run with --apply to write.")
        return 0
    for fname, patches in PATCHES.items():
        text, crlf = texts[fname]
        for old, new in patches:
            o = old.replace("\n", "\r\n") if crlf else old
            nw = new.replace("\n", "\r\n") if crlf else new
            text = text.replace(o, nw)
        shutil.copy2(fname, fname + ".bak_runtime_leak")
        Path(fname).write_bytes(text.encode("utf-8"))
        left = text.count("task.runtime")
        print(f"[WROTE] {fname}  (remaining 'task.runtime' occurrences: {left})")
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
