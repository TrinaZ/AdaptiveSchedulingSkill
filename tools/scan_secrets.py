"""scan_secrets.py -- refuse to publish if anything looks like an API key.
Usage: python3 tools/scan_secrets.py [root]   (exit code 1 if a likely key is found)"""
import os, re, sys
PAT = re.compile(r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35}|ghp_[A-Za-z0-9]{36})")
ENVLINE = re.compile(r"(DEEPSEEK|DASHSCOPE|OPENAI|ZHIPUAI|MOONSHOT|ANTHROPIC|GOOGLE)_API_KEY\s*=\s*['\"]?([A-Za-z0-9_\-\.]+)")
PLACEHOLDER = re.compile(r"(sk-)?x+\.*|your.*|xxx.*|\.\.\.|<.*>", re.I)
root = sys.argv[1] if len(sys.argv) > 1 else "."
bad = 0
for d, _, fs in os.walk(root):
    if ".git" in d.split(os.sep): continue
    for f in fs:
        p = os.path.join(d, f)
        try: txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        for m in PAT.finditer(txt):
            if PLACEHOLDER.fullmatch(m.group(0)): continue
            print(f"LIKELY KEY  {p}: {m.group(0)[:7]}***"); bad += 1
        for m in ENVLINE.finditer(txt):
            if PLACEHOLDER.fullmatch(m.group(2)): continue
            print(f"KEY ASSIGNMENT  {p}: {m.group(0)[:22]}***"); bad += 1
print("secret scan:", "FAILED" if bad else "clean")
sys.exit(1 if bad else 0)
