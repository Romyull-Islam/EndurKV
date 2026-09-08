#!/bin/bash
# ============================================================================
# run_phone_watchdog_compaction_matrix.sh -- muKV on the phone GPU across the two
# levers that are ours to set: the thermal watchdog and the compaction mode.
# (2026-08-08, OnePlus 15 / Adreno 840)
#
# THE MATRIX. 7 cells:
#   vanilla                                   (NO watchdog -- baselines never get it)
#   muKV x {watchdog off, LOW, HIGH} x {round-trip, in-place compaction}
#
# WHY THESE TWO LEVERS TOGETHER. They are believed independent but have never been
# measured that way: the watchdog trades clock for thermal headroom, compaction trades
# peak memory for decode speed. Crossing them shows whether the watchdog's cost is the
# same under both compaction modes -- and in particular whether in-place compaction's
# lower memory peak buys thermal headroom the round-trip cannot.
#
# WATCHDOG VERSIONS -- the project's two REAL ladders, not invented ones. Both are the
# v5 daemon; they differ ONLY in trigger temperature (verified by diffing the two files:
# the sole differences are bat_tier/skin_tier constants and the log banner). Same tiers
# 1200/1050/967/902/826 MHz, same gpuss-junction backstop, zones resolved BY NAME:
#   v5LOW  battery 36.0/36.5/37.0/37.5/38.0   skin 39.5/40.0/40.5/41.0/41.5  (early)
#   v5HIGH battery 47.0/48.0/48.5/49.0/49.5   skin 50.0/51.0/51.5/52.0/52.5  (shipped)
# An earlier version of this script used made-up thresholds (shell 38.5/40.5, bat 35/37)
# that matched NEITHER ladder; those runs were discarded.
# The watchdog is REDUCE-ONLY and muKV-only. Vanilla runs on native GPU DVFS.
#
# QUALITY IS MEASURED ON 3 CELLS, NOT 7, AND THAT IS DELIBERATE. The watchdog changes
# only the GPU clock; it cannot change which cells are kept or any arithmetic, so PPL is
# invariant across the three watchdog settings by construction. Compaction CAN in
# principle differ (it changes attention tiling), so PPL is run for vanilla and for both
# compaction modes. Running all 7 would spend hours re-measuring a constant.
#
# COOLING GATE BEFORE EVERY CELL: DDR <= 35 C, battery <= 33 C, charging OFF while
# cooling. A failed gate SKIPS the cell rather than running it hot -- a timed or energy
# cell taken from a warm start is not comparable and is worse than a missing one.
# charging_restore runs on EXIT, not only on success: a killed script used to leave the
# phone with charging disabled.
#
# ENERGY comes from the USB rail via sample_sensors.sh:
#   E = SUM( usb_voltage_uv/1e6 * |usb_current_ua|/1e6 * dt ),  dt from monotonic_s,
#   capped at 5 s per sample so a stalled sampler cannot invent joules.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

HOSTOUT=/tmp/phone_wd_matrix; mkdir -p "$HOSTOUT"
TS=$(date +%Y%m%d_%H%M%S)
DEV=/data/local/tmp/endurkv/logs/wdmx_$TS
BIN=/data/local/tmp/ukv
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
WD_STOP=/data/local/tmp/gpu_wd.stop
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad

adb_safe_shell "mkdir -p $DEV" < /dev/null
# FIXED 2026-08-08: this used to prefer a SCRATCHPAD copy of the eval slice and fall
# back to the repo one. The scratchpad copy is contaminated -- 98.5% of its 60-char
# windows appear verbatim in the prompt -- so the PPL cells measured VERBATIM RECALL of
# retained text, not prediction. That hands the win to whichever policy evicted least:
# vanilla scored PPL 1.07 and muKV 13.35, which says nothing about language modelling.
# The repo slices are the checked-in, offset-documented ones (see benchmarks/ppl/README).
PROMPT_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/llama1b_12288tok.txt
EVAL_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ppl/wiki_eval_disjoint.txt

# ASSERT disjointness rather than trusting the filename. A slice named "disjoint" that
# is not disjoint is exactly how the above went unnoticed; this makes it loud instead.
python3 - "$PROMPT_SRC" "$EVAL_SRC" <<'PYEOF' || { echo "FATAL: eval slice overlaps the prompt -- refusing to run PPL cells"; exit 1; }
import sys
P=open(sys.argv[1],errors="replace").read(); E=open(sys.argv[2],errors="replace").read()
W=200; hits=sum(1 for i in range(0,len(E)-W,W) if E[i:i+W] in P)
print("  [ppl-slice] %d/%d %d-char windows of eval found in prompt"%(hits,len(range(0,len(E)-W,W)),W))
sys.exit(1 if hits else 0)
PYEOF

adb push "$PROMPT_SRC" "$DEV/prompt.txt" < /dev/null >/dev/null 2>&1
adb push "$EVAL_SRC"   "$DEV/eval.txt"   < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v5_real.sh \
         /data/local/tmp/gpu_watchdog_v5_real.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v5_low.sh \
         /data/local/tmp/gpu_wd_v5low.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh \
         /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1

# charging must come back on however this script ends, including a kill.
cleanup(){ adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --k-nominal 1024"

policy_flags(){ case "$1" in
  vanilla)  echo "--policy vanilla" ;;
  mukv_rt)  echo "$MU --force-defrag" ;;      # state round-trip: second context, 2x peak
  mukv_ip)  echo "$MU --compact-inplace" ;;   # chunked slide in place: no extra peak
esac; }

# the two shipped ladders, by file
wd_script(){ case "$1" in
  low)  echo "/data/local/tmp/gpu_wd_v5low.sh" ;;
  high) echo "/data/local/tmp/gpu_watchdog_v5_real.sh" ;;
  *)    echo "" ;;
esac; }

start_wd(){ local tag=$1 mode=$2
  [ "$mode" = "off" ] && return 0
  adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh $(wd_script $mode) /data/local/tmp/gpu_wd_$tag.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null; }
stop_wd(){ adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock'" < /dev/null; }

cell(){ # $1 tag  $2 policy-key  $3 watchdog-mode  $4 eval-mode
  local TAG=$1 POLK=$2 WD=$3 MODE=$4
  local D="$HOSTOUT/${MODE}_${TAG}"
  [ -f "$D/meta.json" ] && { echo "  [$MODE/$TAG] cached"; return; }
  mkdir -p "$D"

  echo "[$(date +%H:%M:%S)] cooling for $MODE/$TAG ..."
  local CG
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  case "$CG" in *"cool ddr="*) : ;;
    *) echo "  [SKIP-HOT] $MODE/$TAG -- cool gate failed, cell NOT run"; echo "$CG" > "$D/skipped"; return ;;
  esac

  local EX="--eval-mode gen --max-tokens 4096 --ignore-eos"
  [ "$MODE" = "ppl" ] && EX="--eval-mode ppl --eval-text $DEV/eval.txt"

  adb_safe_shell "su -c 'rm -f /data/local/tmp/sens_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sens_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  start_wd "$TAG" "$WD"

  echo "[$(date +%H:%M:%S)] running $MODE/$TAG (wd=$WD) ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $MODEL \
    --prompt $DEV/prompt.txt --prompt-id $TAG $EX --ctx-size 16384 --seed 42 --threads 4 \
    --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 $(policy_flags $POLK) \
    --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/${MODE}_$TAG.json --out-gen $DEV/${MODE}_$TAG.gen \
    --out-csv $DEV/${MODE}_$TAG.steps.csv > $DEV/${MODE}_$TAG.out 2> $DEV/${MODE}_$TAG.err" < /dev/null

  stop_wd
  adb_safe_shell "su -c 'pkill -f sample_sensors.sh'" < /dev/null

  adb pull "$DEV/${MODE}_$TAG.json" "$D/meta.json" < /dev/null >/dev/null 2>&1
  adb pull "$DEV/${MODE}_$TAG.err"  "$D/err"       < /dev/null >/dev/null 2>&1
  adb pull "/data/local/tmp/sens_$TAG.csv" "$D/sensors.csv" < /dev/null >/dev/null 2>&1
  [ "$WD" != "off" ] && adb pull "/data/local/tmp/gpu_wd_$TAG.log" "$D/watchdog.log" < /dev/null >/dev/null 2>&1

  if [ -f "$D/meta.json" ]; then
    python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/phone_cell_report.py "$D" "$MODE/$TAG"
  else
    echo "  [$MODE/$TAG] FAILED: $(tail -1 "$D/err" 2>/dev/null | cut -c1-70)"
  fi
}

# ---- gen cells: all 7 arms -------------------------------------------------
cell vanilla        vanilla off  gen
for WD in off low high; do
  cell "mukv_rt_$WD" mukv_rt "$WD" gen
  cell "mukv_ip_$WD" mukv_ip "$WD" gen
done

# ---- ppl cells: quality cannot depend on the watchdog (clock only) ---------
cell vanilla        vanilla off  ppl
cell mukv_rt_off    mukv_rt off  ppl
cell mukv_ip_off    mukv_ip off  ppl

echo "WD_MATRIX_DONE -> $HOSTOUT"
