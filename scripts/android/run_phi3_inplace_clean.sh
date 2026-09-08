#!/bin/bash
# ============================================================================
# run_phi3_inplace_clean.sh -- Phi-3 in-place + v5LOW, measured properly. (2026-08-10)
#
# WHY A RE-RUN. The first attempt produced a 134% spread between two repeats of the
# IDENTICAL configuration: 13.43 tok/s then 5.75. Not noise -- thermal history. Cell 1
# ran after a 4-SECOND cool gate (the phone happened to be cold already); cell 2 ran
# after a 14-minute Phi-3 run with a 15-minute cool-down that was not enough. Phi-3 cells
# are ~25 min of sustained GPU load, so they heat far more than the ~5 min Llama-1B cells
# the standard gate was tuned against.
#
# THREE FIXES over the previous runner:
#  1. GATE -> SETTLE -> RE-GATE. The project gate (DDR<=35 C, batt<=33 C) is necessary but
#     not sufficient: crossing a threshold is not the same as being cooled, and a cell can
#     legally start anywhere in the band. A previous settle-only version let DDR drift back
#     UP to 36.3 C while waiting for stability, so the gate is now re-checked AFTER the
#     settle. Start temperatures are logged, because "all arms passed the gate" was true of
#     the pair that differed by 4.6 C and 36% in wall time.
#  2. sample_sensors.sh runs, so the energy column exists. The previous runner captured the
#     watchdog log but not the power trace, which is why mJ/token was blank.
#  3. gen.txt is kept, so the ! -density corruption check can run. Without it the cell is
#     unverifiable -- and on this backend five baselines turned out to be producing garbage.
#
# n=3. Two cells cannot distinguish an outlier from a trend, and with the median statistic
# n=2 degenerates to the mean, which is what made the earlier row report a tok/s no run
# achieved.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
OUT=/tmp/phone_gpu_16k
DEV=/data/local/tmp/endurkv/logs/phi3clean_$(date +%Y%m%d_%H%M%S)
WD_STOP=/data/local/tmp/gpu_wd.stop
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --k-nominal 1024"

adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/phi3_12288tok.txt $DEV/prompt.txt < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v5_low.sh /data/local/tmp/gpu_wd_v5low.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock; pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

settle(){ # gate -> wait for DDR to stop falling -> gate again
  for attempt in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c '
      prev=999; same=0
      for i in \$(seq 1 90); do
        d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100))
        diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff))
        if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi
        [ \$same -ge 3 ] && break
        prev=\$d; sleep 10
      done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    DDR=$(echo $R | awk '{print int($1/1000)}'); BAT=$(echo $R | awk '{print int($2/1000)}')
    echo "    post-settle: ddr=${DDR}C batt=${BAT}C"
    [ "${DDR:-99}" -le 35 ] && [ "${BAT:-99}" -le 33 ] && return 0
    echo "    drifted above the gate -- cooling again (attempt $attempt)"
  done
  return 1
}

cell(){ local TAG=$1
  local D=$OUT/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling+settling for $TAG ..."
  settle | tee "$D/thermal_start" || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/sens_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sens_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/gpu_wd_v5low.sh $DEV/${TAG}_wd.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M \
    --prompt $DEV/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos \
    --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $MU --compact-inplace --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null \
    > $DEV/$TAG.out 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock; pkill -f sample_sensors'" < /dev/null
  adb pull $DEV/$TAG.json "$D/meta.json"  < /dev/null >/dev/null 2>&1
  adb pull $DEV/$TAG.gen  "$D/gen.txt"    < /dev/null >/dev/null 2>&1
  adb pull $DEV/$TAG.err  "$D/err"        < /dev/null >/dev/null 2>&1
  adb pull "/data/local/tmp/sens_$TAG.csv" "$D/sensors.csv" < /dev/null >/dev/null 2>&1
  adb pull $DEV/${TAG}_wd.log "$D/watchdog.log" < /dev/null >/dev/null 2>&1
  python3 -c "
import json,re,os
f='$D/meta.json'
if not os.path.exists(f): print('  [$TAG] FAILED'); raise SystemExit
j=json.loads(re.sub(r':\s*-?inf\b',': Infinity',re.sub(r':\s*-?nan\b',': NaN',open(f).read())))
g='$D/gen.txt'; b=None
if os.path.exists(g):
    t=open(g,errors='replace').read(); b=100*t.count('!')/max(len(t),1)
print('  [%-18s] prefill=%6.1fs decode=%6.1fs wall=%7.1fs tps=%6.2f cells=%4.0f compact=%s bangs=%s'%(
 '$TAG', j['prefill_ms']/1000, j['decode_ms']/1000, j['total_ms']/1000, j.get('decode_tps') or 0,
 j['retained_kv_bytes']/(32*32*96*2*2.0), j.get('compaction_mode'),
 ('%.2f%%'%b) if b is not None else 'n/a'))"
  [ -f "$D/watchdog.log" ] && echo "      watchdog steps: $(grep -c 'clk=' "$D/watchdog.log")"
}
cell phi3_mukv_ip
cell phi3_mukv_ip_r2
cell phi3_mukv_ip_r3
echo PHI3_CLEAN_DONE
