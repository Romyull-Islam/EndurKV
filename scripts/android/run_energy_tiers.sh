#!/bin/bash
# ============================================================================
# run_energy_tiers.sh -- what does each energy-aware tier actually cost and buy?
# (2026-08-11, OnePlus 15 / Adreno 840)
#
# The energy-aware controller picks a k-pct tier from state of charge: 20% above 50% SoC,
# 10% between 20-50%, 5% below 20%. Those tiers were chosen from an RTX SPEEDUP curve.
# Nobody has measured what they cost or save ON THE PHONE, in energy or in quality, so the
# ladder is currently justified by the wrong device and the wrong metric. This measures the
# operating points directly.
#
# ENERGY IS RAIL + PACK. Integrating the USB rail alone undercounts by 4-36%: usb_online=1
# and status="Not charging" look like a cleanly rail-powered run, but the SoC's peak draw
# exceeds what the rail delivers and the battery silently makes up the difference. Measured
# on these very cells: vanilla drew 52 mAh from the pack, muKV 26, muKV-no-compaction 14.
# Rail-only therefore FLATTENS exactly the differences under test (it made three arms look
# 3% apart when they are 20-28% apart). Both channels are captured here and summed.
#
# COOLING: gate (DDR<=35 C, batt<=33 C) then settle until DDR stops falling, then re-check
# the gate. Crossing a threshold is not the same as being cooled -- an earlier Phi-3 pair
# that merely passed the gate differed 134% between repeats of the same configuration.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
E=/data/local/tmp/endurkv/logs/eval_disjoint.txt
DEV=/data/local/tmp/endurkv/logs/etiers_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/energy_tiers; mkdir -p $HOST
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ppl/wiki_eval_disjoint.txt $E < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R | awk '{print int($1/1000)}'); local _sb=$(echo $R | awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1
}

cell(){ # tag  policy-flags  mode
  local TAG=$1 FLAGS=$2 MODE=$3
  local D=$HOST/${MODE}_$TAG; [ -f "$D/meta.json" ] && { echo "  [$MODE/$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $MODE/$TAG ..."; settle || { echo "  [SKIP-HOT]"; return; }
  local EX="--eval-mode gen --max-tokens 4096 --ignore-eos"
  [ "$MODE" = ppl ] && EX="--eval-mode ppl --eval-text $E"
  adb_safe_shell "su -c 'rm -f /data/local/tmp/s_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/s_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $MODE/$TAG ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG $EX --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $FLAGS --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/${MODE}_$TAG.json --out-gen $DEV/${MODE}_$TAG.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/${MODE}_$TAG.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb pull $DEV/${MODE}_$TAG.json "$D/meta.json" < /dev/null >/dev/null 2>&1
  adb pull $DEV/${MODE}_$TAG.gen  "$D/gen.txt"   < /dev/null >/dev/null 2>&1
  adb pull "/data/local/tmp/s_$TAG.csv" "$D/sensors.csv" < /dev/null >/dev/null 2>&1
  python3 /home/mislam22/EndurKV_workspace/EndurKV/scripts/energy_cell_report.py "$D" "$MODE/$TAG" 2>/dev/null || echo "  [$MODE/$TAG] FAILED"
}

for m in gen ppl; do
  cell vanilla  "--policy vanilla"      $m
  cell pct20    "$MU --k-pct 20"        $m
  cell pct10    "$MU --k-pct 10"        $m
  cell pct5     "$MU --k-pct 5"         $m
done
echo ETIERS_DONE
