#!/bin/bash
# ============================================================================
# run_gpu_sustained_thermal.sh -- the GPU counterpart to Fig. "bonsai thermal".
# (2026-08-09)
#
# WHY THIS EXISTS. The thermal figure in both drafts (Fig. 4 in the 8-page, Fig. 6 in
# the 2027 draft) shows the watchdog converting vanilla's throttle SAWTOOTH -- the prime
# clock repeatedly collapsing to 883 MHz from ~40 min onward -- into a flat plateau at
# 1267 MHz, with battery held just under the 50 C trigger. That is a real and strong
# result, and it is the watchdog's actual value: SUSTAINED, STABLE operation, not mean
# throughput. But it is a CPU measurement: the top panel is the prime (big-core) clock,
# those runs record no GPU offload, and Bonsai-8B cannot run on Adreno at all (there is
# no Q1_0 Vulkan shader). There is no GPU equivalent on disk -- the deployment run that
# should have produced one failed its pulls and left /tmp/deploy empty.
#
# WHY A COLD-START A/B CANNOT ANSWER THIS. Two independent campaigns (n=3 on 2026-07-30,
# n=1x6 on 2026-08-08) measured the GPU watchdog under the cool gate and both found
# exactly nothing: zero clock steps, and wall/energy differences inside a 3-4% noise
# band. That is not evidence the watchdog is useless -- it is a direct consequence of the
# protocol. The gate (DDR<=35 C, battery<=33 C) exists to make timing comparable, and it
# guarantees the phone starts far below any ladder: v5LOW triggers at 36 C battery,
# v5HIGH at 47 C, and a cold-start 4096-token generation peaks at 35 C. The gate and the
# watchdog test are mutually exclusive by construction.
#
# THE PROTOCOL HERE IS THE OPPOSITE ONE, deliberately. Cool ONCE at the start so every
# arm begins from the same baseline, then drive the GPU back-to-back with no gate between
# generations until it reaches thermal equilibrium. This is the regime the drafts call
# "sustained-hot", and the only one in which a reduce-only watchdog can act.
#
# WHAT IS MEASURED. GPU clock at 2 Hz alongside battery/skin/DDR/gpuss, so the GPU
# analogue of the prime-clock panel can be plotted: does vanilla's GPU clock develop the
# same sawtooth, and does the watchdog replace it with a plateau? Per-iteration tok/s is
# recorded too, because the claim to test is stability over time, not the mean -- an arm
# whose throughput decays across iterations is throttling even if its average looks fine.
#
# WHICH LADDER FOR THE GPU: v5LOW is the one under test. The two ladders were calibrated
# against different heat sources. v5HIGH (battery 47.0/48.0/48.5, skin 50.0/51.0/51.5) is
# anchored to the CPU deep-throttle trigger measured at 50 C battery, which is right for a
# sustained CPU workload like Bonsai-8B. A GPU workload heats the surface on a different
# path and to lower absolute numbers: the cold-start GPU runs peaked at 39.5 C skin and
# 35.4 C battery, so v5HIGH cannot fire on this workload however long it runs. v5LOW
# (battery 36.0/36.5/37.0, skin 39.5/40.0/40.5) sits exactly where this workload lives.
# v5HIGH is kept as a fourth arm to CONFIRM it stays dormant rather than assuming it.
#
# ARMS: vanilla (no watchdog, as always -- baselines never receive it), muKV without the
# watchdog, muKV with v5LOW, and muKV with v5HIGH. The no-watchdog muKV arm matters: muKV may reach equilibrium
# below the ladder on its own, in which case the watchdog is unnecessary on GPU rather
# than ineffective, and those are different findings.
# ============================================================================
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

  # Cool ONCE, to the project's gate, so all three arms start from the same baseline.
  echo "[$(date +%H:%M:%S)] $TAG: cooling to the standard gate (DDR<=35, batt<=33) ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; return ;; esac

  # GPU clock sampler: the panel the CPU figure has and this one needs.
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

  # Back-to-back, NO cooling between iterations -- this is what builds the heat.
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

  # FIXED 2026-08-09: teardown used pkill -f "cur_freq", which never matched the
  # sampler's command line. Every arm's sampler therefore kept running to the end of the
  # campaign, so each clock trace contained its own arm PLUS all later arms superimposed
  # -- vanilla's "trace" spanned 287 min instead of 47. All clock-distribution analysis
  # from that run was void (it showed a LOWER mean cap for the faster arm). Kill by PID.
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
