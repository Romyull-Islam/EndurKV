#!/bin/bash
# GPU sustained-thermal run, Llama-3.2-1B. Cool once, then run back-to-back
# 4096-token generations with no cooling in between until the phone reaches
# thermal equilibrium. A cool gate before every run would keep the phone below
# every watchdog ladder, so the watchdog can only act in this regime.
# Logs GPU clock and temperatures at 2 Hz plus per-iteration tok/s.
# Arms: vanilla (no watchdog), muKV without watchdog, muKV with v5LOW
# (battery 36.0/36.5/37.0 C, skin 39.5/40.0/40.5 C) and muKV with v5HIGH
# (battery 47/48/48.5 C, skin 50/51/51.5 C), a CPU-calibrated ladder.
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

BIN=/data/local/tmp/ukv
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
HOST=/tmp/gpu_sustained; mkdir -p "$HOST"
DEV=/data/local/tmp/endurkv/logs/gpusus_$(date +%Y%m%d_%H%M%S)
WD_STOP=/data/local/tmp/gpu_wd.stop
ITERS=${ITERS:-8}          # 8 x ~4.5 min ~= 36 min per arm: past equilibrium
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --k-nominal 1024 --compact-inplace"

adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/llama1b_12288tok.txt $DEV/prompt.txt < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v5_real.sh /data/local/tmp/gpu_watchdog_v5_real.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v5_low.sh  /data/local/tmp/gpu_wd_v5low.sh < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1

cleanup(){ adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock; pkill -f sample_sensors; pkill -f gpuclk_log; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

arm(){ # $1 tag  $2 flags  $3 watchdog(none|v5high)
  local TAG=$1 FLAGS=$2 WD=$3
  local O="$HOST/$TAG"; [ -f "$O/iters.txt" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$O"

  # Cool once to the standard gate so every arm starts from the same baseline.
  echo "[$(date +%H:%M:%S)] $TAG: cooling to the standard gate (DDR<=35, batt<=33) ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; return ;; esac

  # GPU clock and temperature sampler, 2 Hz.
  adb_safe_shell "su -c 'rm -f $DEV/${TAG}_clk.csv; nohup sh -c \"echo \\$\\$ > $DEV/${TAG}_clk.pid; echo t_s,gpu_clk,gpu_max,batt_mC,skin_mC,ddr_mC,gpuss_mC > $DEV/${TAG}_clk.csv; while true; do
      c=\\\$(cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq 2>/dev/null || cat /sys/kernel/gpu/gpu_clock 2>/dev/null);
      m=\\\$(cat /sys/kernel/gpu/gpu_max_clock 2>/dev/null);
      b=\\\$(cat /sys/class/thermal/thermal_zone93/temp 2>/dev/null);
      d=\\\$(cat /sys/class/thermal/thermal_zone47/temp 2>/dev/null);
      g=\\\$(cat /sys/class/thermal/thermal_zone40/temp 2>/dev/null);
      s=0; for z in 55 56 57 62; do v=\\\$(cat /sys/class/thermal/thermal_zone\\\$z/temp 2>/dev/null); [ -n \\\"\\\$v\\\" ] && [ \\\"\\\$v\\\" -gt \\\"\\\$s\\\" ] && s=\\\$v; done;
      echo \\\"\\\$(date +%s),\\\$c,\\\$m,\\\$b,\\\$s,\\\$d,\\\$g\\\" >> $DEV/${TAG}_clk.csv; sleep 0.5; done\" >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/sample_sensors.sh --out $DEV/${TAG}_sens.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  case "$WD" in
    v5low)  adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/gpu_wd_v5low.sh $DEV/${TAG}_wd.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null ;;
    v5high) adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/gpu_watchdog_v5_real.sh $DEV/${TAG}_wd.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null ;;
  esac

  # No cooling between iterations, so heat builds up.
  : > "$O/iters.txt"
  for i in $(seq 1 $ITERS); do
    echo "[$(date +%H:%M:%S)]   $TAG iter $i/$ITERS"
    adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $MODEL \
      --prompt $DEV/prompt.txt --prompt-id ${TAG}_$i --eval-mode gen --max-tokens 4096 --ignore-eos \
      --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
      --cache-type-k f16 --cache-type-v f16 $FLAGS --n-batch 512 --n-ubatch 64 \
      --out-meta $DEV/${TAG}_$i.json --out-gen /dev/null --out-csv /dev/null \
      > /dev/null 2> $DEV/${TAG}_$i.err" < /dev/null
    adb pull $DEV/${TAG}_$i.json "$O/iter_$i.json" < /dev/null >/dev/null 2>&1
    python3 -c "
import json,re,os
f='$O/iter_$i.json'
if os.path.exists(f):
    s=re.sub(r':\s*-?nan\b',': NaN',open(f).read()); j=json.loads(re.sub(r':\s*-?inf\b',': Infinity',s))
    print('    iter %s tps=%.2f wall=%.0fs'%('$i', j.get('decode_tps') or 0, j['total_ms']/1000))" | tee -a "$O/iters.txt"
  done

  # Kill the clock sampler by PID. pkill -f does not match its command line, and a
  # leftover sampler would mix later arms into this arm's trace.
  adb_safe_shell "su -c 'touch $WD_STOP; pkill -f sample_sensors; [ -f $DEV/${TAG}_clk.pid ] && kill \$(cat $DEV/${TAG}_clk.pid) 2>/dev/null; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock'" < /dev/null
  adb pull $DEV/${TAG}_clk.csv  "$O/clk.csv"     < /dev/null >/dev/null 2>&1
  adb pull $DEV/${TAG}_sens.csv "$O/sensors.csv" < /dev/null >/dev/null 2>&1
  [ "$WD" != none ] && adb pull $DEV/${TAG}_wd.log "$O/watchdog.log" < /dev/null >/dev/null 2>&1
  echo "  [$TAG] done"
}

arm vanilla     "--policy vanilla" none
arm mukv_nowd   "$MU"              none
arm mukv_wdlow  "$MU"              v5low
arm mukv_wdhigh "$MU"              v5high
echo "GPU_SUSTAINED_DONE -> $HOST"
