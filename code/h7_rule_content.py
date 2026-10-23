"""h7_rule_content.py -- keyword classification of LLM-written rules on the H7 objectives (descriptive).
Counts rules that use task_type, that contain lateness logic, and of those, rules that raise ("boost") or lower
("defer") the priority of late requests. Keyword-based; samples were checked by hand.
Usage: python3 h7_rule_content.py results/h7_llm_*_llmdebug.json results/h7_backend_ds-reasoner_*_llmdebug.json"""
import sys, json, re
late = lambda c: bool(re.search(r"(if|else)[^\n]*(slack|late|overdue|due)[^\n]*[<>=]", c)) or "lateness" in c
boost = lambda c: bool(re.search(r"(\+\s*|=\s*|return\s+)1e[3-9]|\*=\s*[1-9]|boost|prioriti[sz]e|urgen", c))
defer = lambda c: bool(re.search(r"(return\s+|=\s*)-\s*1e[3-9]|-\s*1e[3-9]\s*\+|deprioriti|give up|hopeless|abandon|sunk|lost cause|already (lost|missed)|no longer", c, re.I))
for f in sys.argv[1:]:
    D = json.load(open(f))
    C = [c["code"] for d in D if str(d.get("mode", "")).startswith("llm_skill") for c in d["candidates"] if isinstance(c.get("code"), str)]
    Lt = [c for c in C if late(c)]
    print(f"{f.split('/')[-1]:<45} rules {len(C):>4}  task_type {sum('task_type' in c for c in C):>4}  lateness logic {len(Lt):>4}  "
          f"boost {sum(map(boost, Lt)):>4}  defer {sum(map(defer, Lt)):>3}")
