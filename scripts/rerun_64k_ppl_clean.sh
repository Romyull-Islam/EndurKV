#!/bin/bash
# ============================================================================
# rerun_64k_ppl_clean.sh -- re-measure every 64K perplexity cell against a slice
# that is actually disjoint from the prompt. (2026-08-13)
#
# WHY THIS EXISTS. All three 64K campaigns scored perplexity on
# wiki_eval_disjoint_long.txt. That slice was verified disjoint against the 12K PHONE
# prompt and the property was assumed to carry over. It does not. The 64K prompt is
# 251502 chars drawn from the same WikiText stream and reaches far enough to contain
# the slice verbatim: 70/119 200-char windows and 233/399 60-char windows are present
# in the prompt. So the model was being asked to re-emit text it could still see, which
# scores RECALL of retained cache, not prediction -- and an evictor is supposed to lose
# at recall. Every 64K PPL number produced before 2026-08-13 is void. The old cells are
# preserved (not deleted) under /tmp/rtx64k_*_VOID_contaminated_ppl so the void numbers
# remain auditable rather than silently replaced.
#
# WHAT RUNS. Only the ppl_* cells. The gen_* cells time prefill/decode/memory and do
# not touch the eval text, so they are left cached -- re-running them would cost hours
# of 57344-token prefills for numbers that cannot change.
#
# SEQUENTIAL, NOT PARALLEL: all three campaigns target the same RTX 4500 Ada, and a
# 64K f16 cache is ~2.1 GB before the FA-off baselines' attention scratch. Running them
# concurrently would contend for VRAM and confound the timings that share these dirs.
#
# NOTE ON THIS MACHINE (2026-08-13): nvidia-smi is broken -- the loaded kernel module is
# 580.159.03 while libcuda is 580.173.02, so NVML refuses to init. The CUDA driver API
# itself is fine (cuInit returns 0) and these runs are unaffected; only GPU telemetry is
# unavailable until the box reboots. None of these scripts call nvidia-smi.
# ============================================================================
set -u
cd /home/mislam22/EndurKV_workspace || exit 1
S=EndurKV/scripts
LOG=/tmp/rerun64kppl.log

echo "=== 64K clean-slice PPL re-run started $(date -Is) ===" | tee -a $LOG

# assert the slice really is disjoint before burning GPU hours on it -- this is the
# exact check whose absence voided the first campaign.
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
  echo "--- $sc  $(date -Is) ---" | tee -a $LOG
  bash $S/$sc 2>&1 | tee -a $LOG
done
echo "=== ALL_64K_PPL_DONE $(date -Is) ===" | tee -a $LOG
