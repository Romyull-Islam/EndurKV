#!/bin/bash
# ============================================================================
# rerun_64k_ppl_adhoc_arms.sh -- clean-slice PPL for the three 64K arms that were
# created outside a campaign script. (2026-08-13)
#
# WHY THESE THREE ARE SEPARATE. rerun_64k_ppl_clean.sh re-runs the arms each campaign
# script loops over. Three arms in /tmp/rtx64k_native were added ad hoc (via /tmp/fair.sh
# and a manual FA-on StreamingLLM run) and are therefore in no script's loop, so they
# kept their gen_ cell but lost their ppl_ cell to the contamination quarantine. Two of
# them carry the sharpest ablation in the campaign and cannot be left without a quality
# number:
#   streamingllm_fa  -- StreamingLLM, same 2049 cells, flash-attention ON but NOT compacted
#   sllm_compact     -- StreamingLLM, same 2049 cells, flash-attention ON AND compacted
# Against the FA-off run of the same policy those three points separate the two mechanisms:
# FA-on alone reaches break-even (1.01x), compaction is what converts it to 2.61x. The
# third arm, mukv_matched, is muKV at the retention StreamingLLM actually achieves
# (K=2542 -> ~2048 cells), so the comparison is at equal cells rather than equal K.
#
# BUILD-SKEW WARNING, which belongs in the caption of any table using these rows.
# The plain `streamingllm` arm in this directory was measured BEFORE the change that lets
# positional policies keep flash-attention on, so it ran FA-off (llama_context reports
# flash_attn = disabled; prefill 93.9 s). streamingllm_fa and sllm_compact ran after
# (flash_attn = auto; prefill ~10.5 s). They are the same policy on different builds. That
# is exactly what makes the ablation legible, but the rows must not be presented as if one
# campaign produced all three.
# ============================================================================
set -u
cd /home/mislam22/EndurKV_workspace || exit 1
B=EndurKV/entropy_probe/build-pc-cuda/eviction_bench
P=EndurKV/benchmarks/ctx_sweep/llama1b_57344tok.txt
E=EndurKV/benchmarks/ppl/wiki_eval_disjoint_64k.txt   # 0/119 and 0/399 vs P, asserted 08-13
M=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
O=/tmp/rtx64k_native

# Flags recovered from each arm's own meta.json + its llama_context FA line, not guessed.
spec(){ case "$1" in
  streamingllm_fa) echo "--policy streamingllm --n-sink 4 --k-nominal 2048 --fa-on-evict" ;;
  sllm_compact)    echo "--policy streamingllm --n-sink 4 --k-nominal 2048 --fa-on-evict --compact-inplace" ;;
  mukv_matched)    echo "$MU --k-nominal 2542 --compact-inplace" ;;
esac; }

for POL in streamingllm_fa sllm_compact mukv_matched; do
  D=$O/ppl_$POL
  [ -f $D/meta.json ] && { echo "  [ppl/$POL] cached"; continue; }
  mkdir -p $D
  timeout 3600 $B --prompt $P --prompt-id ah_$POL --eval-mode ppl --eval-text $E \
    --ctx-size 65536 --model $M --seed 42 --threads 8 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $(spec $POL) \
    --out-meta $D/meta.json --out-gen /dev/null --out-csv /dev/null >/dev/null 2>$D/err
  echo "  [ppl/$POL] $([ -f $D/meta.json ] && echo ok || echo "FAILED: $(tail -1 $D/err|cut -c1-60)")"
done
echo ADHOC_PPL_DONE
