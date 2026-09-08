#!/usr/bin/env python3
# ============================================================================
# make_longbench_table.py -- emit the LongBench LaTeX table.  (2026-08-02)
#
# Rebuild-on-demand: cells land continuously, so this regenerates the table from
# whatever is scored rather than freezing a snapshot by hand.
#
# THREE THINGS THE TABLE MUST STATE, because each was a real error we hit:
#  1. ctx per model. LongBench truncates to the MODEL's max length, not a fixed
#     number. Gemma-2-2B is n_ctx_train=8192; running it at 16384 overflowed its
#     trained context and produced word salad scoring 6.60 F1 under three
#     different policies -- a context bug that reads like a policy result.
#  2. Which SnapKV. The paper retunes window/kernel per benchmark (NIAH 16/5,
#     LongBench 32/7) and the FasterDecoding repo default is 64/5. We report all
#     three separately instead of letting one row stand for "SnapKV".
#  3. Retained cache in CELLS, not MiB. SnapKV cannot use quantized KV on this
#     engine (a per-head evictor needs FA-off; llama.cpp requires flash-attention
#     for quantized V), so it runs f16 at 1.88x the bytes per cell. Comparing MiB
#     would conflate the realizability gap with the cache format.
# ============================================================================
import json, re, os, glob, sys, statistics as st

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lb_cuda"
GOLD = "/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/longbench/gold.json"
sys.path.insert(0, "/home/mislam22/EndurKV_workspace/EndurKV/scripts")
from longbench_score import qa_f1_score, clean

BPE = {"f16": 2.0, "q8_0": 1.0625}
GEOM = {"llama1b": (16, 8, 64), "phi3": (32, 32, 96),
        "gemma2b": (26, 4, 256), "bonsai8b": (36, 8, 128)}
LABEL = {"vanilla": "vanilla (full cache)", "mukv": r"\textbf{$\mu$KV}",
         "snapkv": "SnapKV (w16/k5)", "snapkv_lb": "SnapKV",
         "snapkv_repo": "SnapKV (w64/k5, repo)", "adakv": "Ada-KV", "h2o": "H2O",
         "tova": "TOVA", "streamingllm": "StreamingLLM"}
# ONE SnapKV row, always the configuration SnapKV itself publishes for the
# benchmark in question -- w32/k7 for LongBench. The paper must not contain three
# different things all labelled "SnapKV"; the rule across every table is: use the
# baseline's own published setting for that benchmark, and the FasterDecoding
# default only where the baseline publishes none (e.g. WikiText). w16/k5 and
# w64/k5 were also measured here and span 0.8 F1, so the choice is not load-bearing;
# that goes in the caption, not in extra rows.
ORDER = ["vanilla", "mukv", "snapkv_lb", "adakv", "tova", "h2o", "streamingllm"]
SENSITIVITY = ["snapkv", "snapkv_repo"]
MODELS = [("llama1b", "Llama-1B"), ("gemma2b", "Gemma-2B"),
          ("phi3", "Phi-3"), ("bonsai8b", "Bonsai-8B")]
# 2026-08-04: extended from 2 to 5 F1-scored LongBench tasks, spanning single-doc QA,
# multi-doc QA and few-shot rather than 2 of the suite's 16. Summarization needs
# ROUGE-L and is out of scope. A cell is only averaged when EVERY task is present,
# so a partially-run policy shows "---" rather than an average over a different
# task mix than its baseline (that mismatch is what produced the 110% retention bug).
TASKS = [("hotpotqa", "HotpotQA"), ("2wikimqa", "2WikiMQA"),
         ("multifieldqa_en", "MultiFieldQA"), ("qasper", "Qasper"), ("triviaqa", "TriviaQA")]


def load(p):
    s = open(p).read()
    s = re.sub(r':\s*-?nan\b', ': NaN', s); s = re.sub(r':\s*-?inf\b', ': Infinity', s)
    return json.loads(s)


gold = json.load(open(GOLD))
f1 = {}     # (model,policy,task) -> [scores]
cells = {}  # (model,policy) -> [retained cell counts]
ctxs = {}   # model -> set of ctx seen
prompt_tok = {}  # model -> prompt lengths (kept for reference)
ratio = {}       # (model,policy) -> per-cell retained/prompt ratios
for d in sorted(os.listdir(RUNS)):
    parts = d.split("__")
    if len(parts) != 4:
        continue
    m, pol, task, pid = parts
    g = os.path.join(RUNS, d, "gen.txt")
    mp0 = os.path.join(RUNS, d, "meta.json")
    # skip crashed cells: empty gen.txt + no meta.json (see longbench_score.py)
    if not os.path.exists(g) or not os.path.exists(mp0) or task not in gold or pid not in gold[task]:
        continue
    f1.setdefault((m, pol, task), []).append(
        qa_f1_score(clean(open(g, errors="replace").read(), False), gold[task][pid]["answers"]))
    mp = os.path.join(RUNS, d, "meta.json")
    if os.path.exists(mp) and m in GEOM:
        try:
            j = load(mp)
        except Exception:
            continue
        ctxs.setdefault(m, set()).add(j.get("ctx_size"))
        L, H, D = GEOM[m]
        # LongBench passes no --cache-type, so every policy here is f16: the F1
        # comparison is quantization-matched and cells are directly comparable.
        nc = j["retained_kv_bytes"] / (L * H * D * 2 * 2.0)
        cells.setdefault((m, pol), []).append(nc)
        # FIXED 2026-08-02: the retained-% must be averaged PER CELL. Dividing a
        # policy's mean cell count by a model-wide mean prompt length compares
        # different task mixes -- policies that had only HotpotQA cells (longer
        # prompts) came out >100%, and vanilla, which is 100% by definition, read
        # as 92.1%. Ratio first, then average.
        ratio.setdefault((m, pol), []).append(nc / max(1, j["n_prompt_tokens"]))

print("% auto-generated by scripts/make_longbench_table.py -- do not hand-edit")
print(r"\begin{table*}[tb]")
print(r"\caption{LongBench F1 (official token-F1, max over reference answers) on the RTX 4500 Ada. "
      r"Every policy runs f16 K/V here, so F1 and retained cache are quantization-matched: "
      r"SnapKV cannot use quantized KV on this engine at all, since a per-head evictor needs FA-off "
      r"and llama.cpp requires flash-attention for a quantized V. \textbf{Cells} is retained cache in "
      r"KV cells (format-independent) as a percentage of the prompt. "
      r"SnapKV runs its published LongBench configuration (window 32, avgpool-7); its NIAH "
      r"setting (16/5) and the FasterDecoding default (64/5) were also measured and span "
      r"only 0.8 F1 on Llama-1B, so the comparison does not turn on that choice. "
      r"Gemma-2-2B runs at ctx 8192, its trained context; the other models at 16384.}")
print(r"\label{tab:longbench}")
print(r"\centering\small")
print(r"\setlength{\tabcolsep}{5pt}")
print(r"\begin{tabular}{l l %s r r}" % ("r " * len(TASKS)))
print(r"\toprule")
print(r"\textbf{Model} & \textbf{Policy} & " +
      " & ".join(r"\textbf{%s}" % t[1] for t in TASKS) +
      r" & \textbf{Avg} & \textbf{Cells kept} \\")
print(r"\midrule")
for key, disp in MODELS:
    rows = [p for p in ORDER if any((key, p, t[0]) in f1 for t in TASKS)]
    if not rows:
        continue
    ctx = sorted(ctxs.get(key, {0}))[0]
    print(r"\multicolumn{%d}{l}{\emph{%s\ \ (ctx %s)}} \\" % (len(TASKS) + 4, disp, ctx))
    van = None
    for p in rows:
        vals = [f1.get((key, p, t[0])) for t in TASKS]
        means = [100 * st.mean(v) if v else None for v in vals]
        avg = sum(means) / len(means) if all(m is not None for m in means) else None
        if p == "vanilla":
            van = avg
        c = cells.get((key, p))
        cpct = "---"
        if c and ratio.get((key, p)):
            # % of the prompt still physically resident. This is the realizability
            # number: muKV and StreamingLLM actually reach their budget, per-head
            # policies do not, and it is independent of KV cache format.
            cpct = "%.0f (%.1f\\%%)" % (st.mean(c), 100 * st.mean(ratio[(key, p)]))
        d = "" if avg is None or van is None or p == "vanilla" else " (%+.1f)" % (avg - van)
        print("  & %-26s & %s & %s%s & %s \\\\" % (
            LABEL.get(p, p),
            " & ".join("%.2f" % m if m is not None else "---" for m in means),
            "%.2f" % avg if avg is not None else "---", d, cpct))
    print(r"\midrule")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")
