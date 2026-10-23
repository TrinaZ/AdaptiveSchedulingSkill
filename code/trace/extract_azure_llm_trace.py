"""
extract_azure_llm_trace.py -- build the LLM-inference-serving replay input (H10) from the Azure LLM inference trace 2023
(Patel et al., "Splitwise", ISCA 2024; CC-BY; https://github.com/Azure/AzurePublicDataset, files
data/AzureLLMInferenceTrace_code.csv and data/AzureLLMInferenceTrace_conv.csv).

Real: request arrival times, context (input) tokens and generated (output) tokens of two production LLM inference
services (coding assistant 'code', conversation 'conv'), 16 Nov 2023, about one hour.
Synthesized (stated in the paper): the service time of a request on one serving slot follows a linear latency model,
    t = T0 + A * context_tokens + B * generated_tokens   (prefill linear in input, decode linear in output),
and its KV-cache footprint is (context + generated) / 1000 thousand tokens, held for the whole service time.
Output length is unknown at arrival; the benchmark's runtime estimate keeps the prefill part exact and perturbs the decode
part (column 'mem' stores the prefill part, see make_llm_bench.py).

Output CSV (same columns as the Alibaba extract, so the replay code is shared):
  window, start [s], dur [s], cpu [KV-cache, thousand tokens], mem [prefill seconds], kind ('C' code | 'V' conv), job [request id]
Windows: the overlapping span of both services is split into two halves, window 0 and window 1.
Usage: python3 extract_azure_llm_trace.py AzureLLMInferenceTrace_code.csv AzureLLMInferenceTrace_conv.csv azure_llm_windows.csv
"""
import sys, csv
from datetime import datetime
T0, A, B = 0.05, 0.00015, 0.020      # seconds; latency-model constants (assumed, order of magnitude of a 70B model on one GPU slot)

def rows(path, svc):
    out = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            ts = datetime.strptime(r["TIMESTAMP"][:26], "%Y-%m-%d %H:%M:%S.%f").timestamp()
            ctx, gen = int(r["ContextTokens"]), int(r["GeneratedTokens"])
            out.append((ts, ctx, gen, svc, f"{svc}{i}"))
    return out

code, conv = rows(sys.argv[1], "C"), rows(sys.argv[2], "V")
lo = max(code[0][0], conv[0][0]); hi = min(code[-1][0], conv[-1][0]); mid = (lo + hi) / 2
allr = sorted([x for x in code + conv if lo <= x[0] <= hi])
with open(sys.argv[3], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["window", "start", "dur", "cpu", "mem", "kind", "job"])
    for ts, ctx, gen, svc, job in allr:
        pre = T0 + A * ctx
        w.writerow([0 if ts < mid else 1, f"{ts - lo:.4f}", f"{pre + B * gen:.4f}", f"{(ctx + gen) / 1000:.4f}", f"{pre:.4f}", svc, job])
n0 = sum(1 for x in allr if x[0] < mid)
print(f"span {hi - lo:.0f} s; window 0: {n0} requests, window 1: {len(allr) - n0}; written {sys.argv[3]}")
