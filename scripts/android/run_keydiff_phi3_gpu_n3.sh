#!/bin/bash
# Rounds 2 and 3 of KeyDiff on the Phi-3 GPU, each paired with its own full-cache control,
# so the row is a median over three cooled rounds like the others.
# Reuses the setup and cell() from run_phi3_gpu_complete.sh. Tags carry the round so earlier
# cells are not overwritten, and finished cells are skipped. Control runs first in each round
# so the pair shares a thermal and battery state.
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
