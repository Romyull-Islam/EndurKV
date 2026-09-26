#!/bin/bash
# ============================================================================
# run_keydiff_phi3_gpu_n3.sh -- two more cooled rounds of KeyDiff on the Phi-3
# GPU, each with its own full-cache control. (2026-09-24)
#
# WHY. Table 1's KeyDiff row on Phi-3 (0.27x, 1.36 tok/s, 54,108 cells evicted
# by block re-scoring) is the only n=1 cell in the paper. Every other ratio is
# the median over three cooled rounds, each against the full cache of its own
# round. These two rounds bring the row to n=3 on the same footing.
#
# HOW. Same build (bin_vk_cur), model, prompt, flags, cooling gate, sampler and
# 4096 generated tokens as run_phi3_gpu_complete.sh, whose cell() function is
# reused unchanged: this file only defines the cell list. Tags carry the round
# so the originals (vanilla, keydiff2048) are never overwritten, and a finished
# cell is skipped, so the script can be re-run after a disconnect.
# Order: control, then KeyDiff, in each round, so the pair shares one thermal
# and battery state as closely as the gate allows.
# ============================================================================
set -u
SRC=/home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_phi3_gpu_complete.sh
# take everything from the original up to (not including) its cell list
eval "$(sed -n '/^set -u/,/^cell vanilla /p' "$SRC" | sed '$d')"
for r in 2 3; do
  cell vanilla_r$r      --policy vanilla
  cell keydiff2048_r$r  --policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null >/dev/null 2>&1
LOG "KEYDIFF PHI3 GPU N3 DONE -> $HOST"
