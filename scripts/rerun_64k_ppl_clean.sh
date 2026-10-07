#!/bin/bash
# Re-runs the 64K perplexity cells (ppl_* only) of the three 64K campaigns on
# wiki_eval_disjoint_64k.txt, a slice disjoint from the 64K prompt. The gen_* cells do
# not read the eval text and stay cached. Campaigns run one after another because they
# share one GPU, and running them together would contend for VRAM.
set -u
cd /home/mislam22/EndurKV_workspace || exit 1
S=EndurKV/scripts
LOG=/tmp/rerun64kppl.log

echo "64K clean-slice PPL re-run started $(date -Is)" | tee -a $LOG

# Check that no 200- or 60-char window of the slice appears in the prompt.
python3 - <<'EOF' || { echo "DISJOINTNESS ASSERTION FAILED -- aborting"; exit 1; }
p = open("EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt", encoding="utf-8", errors="replace").read()
e = open("EndurKV/benchmarks/ppl/wiki_eval_disjoint_64k.txt", encoding="utf-8", errors="replace").read()
bad = 0
for W in (200, 60):
    hits = sum(1 for i in range(0, len(e) - W, W) if e[i:i+W] in p)
    print("  assert W=%d: %d overlapping windows" % (W, hits))
    bad += hits
raise SystemExit(1 if bad else 0)
EOF
echo "  slice verified disjoint" | tee -a $LOG

for sc in run_64k_compaction_matrix.sh run_64k_nolimit.sh run_64k_native_budgets.sh; do
  echo "$sc  $(date -Is)" | tee -a $LOG
  bash $S/$sc 2>&1 | tee -a $LOG
done
echo "ALL_64K_PPL_DONE $(date -Is)" | tee -a $LOG
