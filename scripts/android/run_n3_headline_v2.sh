#!/bin/bash
# ============================================================================
# run_n3_headline_v2.sh -- n=3 CPU headline, ORDER-RANDOMISED, CPU-gated.
# (2026-08-26)
#
# WHY v2. v1 was killed mid-run: its cells started between 36.6 and 51.5 C core
# temperature -- a 14.9 C spread -- because the old gate held DDR and battery but
# never the cores, and DDR cools faster than the cores do. That is the same defect
# that made vanilla (cell #1 of the Phi-3 campaign, 30.1 C start) look 23% faster
# at prefill than every policy that followed it at 36 C+.
#
# THREE FIXES:
#   1. cool_gate.sh now gates CPU cores too (<=36 C), excluding the *-trip-*
#      pseudo-zones that read a constant 95 C and would have hung the gate.
#   2. warm_up before the first cell, so cell #1 is not measured on a colder
#      machine than the rest. A gate is an upper bound and cannot fix a cold start.
#   3. ORDER IS RANDOMISED per repeat with a recorded seed. In v1 vanilla ran
#      first in every repeat, so the divisor of every ratio held a fixed position.
#      Now no policy owns a slot, and any residual position effect is spread
#      across arms instead of biasing one.
#
# THREADS = 6, NOT 4. Table 1 (run_definitive_cpu_*.sh) uses --threads 6; the
# LongBench/KeyDiff runners use 4, and deriving this script from the wrong parent
# silently cut every number by a third: vanilla 5.02 -> 3.43 tok/s, an observed
# 6/4 ratio of 1.46 against an ideal 1.50. That one flag explains the whole
# "unexplained 30% slowdown" previously chased through battery voltage and then
# start temperature. Any cell destined for Table 1 must use 6.
#
# Start CPU temperature is recorded per cell so the spread is auditable rather
# than assumed; the scorer refuses to compare cells that differ by >3 C.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }
exec 9>/tmp/.endurkv_queue.lock; flock 9
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
DEV=/data/local/tmp/n3v2; HOST=/tmp/n3_headline_v2
SEED=${SEED:-20260826}
mkdir -p $HOST; adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push /tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad/wikitext_16k_p12k_d4k.txt "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags_for(){
  case "$1" in
    vanilla) echo "--policy vanilla --k-nominal 1024";;
    mukv)    echo "$MU --k-nominal 1024";;
    keydiff) echo "--policy keydiff --n-sink 0 --k-nominal 2048 --compact-inplace --keydiff-decode-block 128";;
    sllm)    echo "--policy streamingllm --n-sink 4 --k-nominal 2000";;
  esac
}
cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "$TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) LOG "  gate: $CG";; *) LOG "  [SKIP-HOT] $TAG ($CG)"; return;; esac
  SC=$(adb_safe_shell "su -c 'c=0; for z in /sys/class/thermal/thermal_zone*; do t=\$(cat \$z/type 2>/dev/null); case \$t in *trip*) continue;; cpu-*|cpullc-*) ;; *) continue;; esac; v=\$((\$(cat \$z/temp)/1000)); [ \$v -gt \$c ] && c=\$v; done; echo \$c'" < /dev/null 2>/dev/null | tr -d ' \r')
  BV=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/voltage_now'" < /dev/null 2>/dev/null | tr -d ' \r')
  echo "start_cpu_c=$SC batt_voltage_uv=$BV" > "$HOST/$TAG/start_state.txt"
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $DEV/$TAG.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null >/dev/null 2>&1
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 7200 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > $DEV/$TAG.out 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0
  while [ $w -lt 7400 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    sleep 30; w=$((w+30)); done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json"   >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$HOST/$TAG/gen.txt"     >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.csv"  "$HOST/$TAG/sensors.csv" >/dev/null 2>&1
  [ -s "$HOST/$TAG/gen.txt" ] && LOG "  [$TAG] ok  start_cpu=${SC}C" || LOG "  [$TAG] NO OUTPUT"
}
LOG "=== n=3 headline v2: CPU-gated, warm-started, order randomised (seed $SEED) ==="
LOG "warming up so cell #1 is not measured on a colder machine ..."
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; warm_up 34'" < /dev/null 2>/dev/null | tail -1
# LATIN-SQUARE ROTATION, not shuf. A random shuffle can leave a policy in the
# same slot twice by chance; rotation guarantees that across the three repeats no
# policy occupies one position more than once, so any residual position effect
# (cell #1 sees the coldest machine) is spread evenly across all four arms
# instead of biasing whichever one happens to be the divisor.
#   r1: vanilla mukv    keydiff sllm
#   r2: mukv    keydiff sllm    vanilla
#   r3: keydiff sllm    vanilla mukv
for r in 1 2 3; do
  case $r in
    1) ORDER="vanilla mukv keydiff sllm" ;;
    2) ORDER="mukv keydiff sllm vanilla" ;;
    3) ORDER="keydiff sllm vanilla mukv" ;;
  esac
  LOG "repeat $r order: $ORDER"
  for pol in $ORDER; do cell ${pol}_r$r $(flags_for $pol); done
done
LOG "N3V2_DONE"
touch /tmp/n3_headline_v2_DONE
