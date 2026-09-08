#!/bin/bash
# ============================================================================
# run_prefill_v5.sh -- prefill measured at a PINNED CPU clock.
# (2026-08-29)
#
# Why v1-v4 all failed, and why this is different:
#   v1  cool-gate every cell -> equal temperature, decayed WALT governor; prefill
#                               set by slot (slot1 ~500 s, slot3 217 s).
#   v2  warm once, no gates  -> equal governor, temperature drifted 38->49 C; the
#                               ramp landed on vanilla/sllm, vanilla ended n=2.
#   v3  cool then warm_up    -> warm_up reached 42 C but ran in a separate adb call
#                               from the launch; heat bled away in the gap by a
#                               variable amount. Warm starts gave 173/171/173 s
#                               (SD 1 s); cool starts gave 243/203/466 s (SD 116 s).
#                               The artifact had period 2 and the rotation period 3,
#                               so muKV drew cells 3,5,7 (all warm) vs vanilla's
#                               1,6,8 -- a 2x win manufactured by scheduling alone.
#   v4  fuse warm+launch     -> removed the adb gap, but the CPU sheds shallow heat
#                               within ~1 s of load stopping, so the measured start
#                               temperature still did not describe the run
#                               (discard 37 C after 31 warm iterations; vanilla_r1
#                               entered at 60 C and took 456 s).
#
# The lesson from all four: temperature was always a proxy. What actually sets
# prefill is the CLOCK. So v5 stops chasing temperature and pins the clock directly
# -- performance governor at a fixed scaling_max_freq on every core, via pin_dvfs.sh.
# The governor cannot decay and every cell runs at the same frequency. Precedent:
# the phone_fixedclock campaign, where three different policies landed within 0.1 s
# of each other (151.7 / 151.8 / 151.8 s).
#
# Rotated n=3. 64 decode tokens: this measures prefill. DVFS is restored at exit.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

exec 9>/tmp/.endurkv_queue.lock; flock 9
DEV=/data/local/tmp/pf5; HOST=/tmp/prefill_v5; PIN_KHZ=1632000
SCRATCH=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
CB=/data/local/tmp/endurkv/bin_cpu_kd
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
mkdir -p $HOST
adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$SCRATCH/wikitext_16k_p12k_d4k.txt" "$DEV/wt/prompt.txt" < /dev/null >/dev/null 2>&1

restore_dvfs(){
  LOG "restoring DVFS"
  adb_safe_shell "su -c 'TARGET_MHZ=$PIN_KHZ sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore'" < /dev/null >/dev/null 2>&1
}
trap restore_dvfs EXIT INT TERM

MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
flags(){ case "$1" in
  vanilla) echo "--policy vanilla --k-nominal 1024";;
  sllm)    echo "--policy streamingllm --n-sink 4 --k-nominal 2000";;
  mukv)    echo "$MU --k-nominal 1024";; esac; }

# read the big-core clock actually in force
curfreq(){ adb_safe_shell "su -c 'cat /sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq 2>/dev/null'" < /dev/null 2>/dev/null | tr -d ' \r'; }

cell(){
  local TAG=$1; shift
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "  $TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  # cool so the SoC never hardware-throttles below the pinned clock
  CG=$(adb_safe_shell "su -c 'COOL_CPU_MAX=40 . /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) : ;; *) LOG "  [SKIP-HOT] $TAG ($CG)"; return;; esac
  # re-assert the pin (a thermal HAL event can lower scaling_max_freq underneath us)
  adb_safe_shell "su -c 'TARGET_MHZ=$PIN_KHZ LITTLE_TARGET_MHZ=$PIN_KHZ sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin'" < /dev/null >/dev/null 2>&1
  local F0; F0=$(curfreq)
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 3600 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size 16384 \
    --n-batch 512 --ubatch-size 64 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  # sample the clock mid-run: proves the pin held for the measurement
  local w=0 FMID=""
  while [ $w -lt 3700 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    [ $w -eq 60 ] && FMID=$(curfreq)
    sleep 20; w=$((w+20)); done
  local F1; F1=$(curfreq)
  echo "pin=$PIN_KHZ f_start=$F0 f_mid=$FMID f_end=$F1 gate=$CG" > "$HOST/$TAG/clock.txt"
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json" >/dev/null 2>&1
  local PF
  PF=$(python3 -c "
import json,re
try:
  j=json.loads(re.sub(r':\s*-?nan\b',': NaN',open('$HOST/$TAG/meta.json').read())); print('%.0f'%(j['prefill_ms']/1000))
except Exception: print('--')" 2>/dev/null)
  LOG "  [$TAG] prefill=${PF}s  clock ${F0:-?}/${FMID:-?}/${F1:-?} kHz"
}

LOG "=== prefill v5: CPU pinned to ${PIN_KHZ} kHz, performance governor ==="
adb_safe_shell "su -c 'TARGET_MHZ=$PIN_KHZ LITTLE_TARGET_MHZ=$PIN_KHZ sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin'" < /dev/null 2>&1 | tail -3
cell discard_warmup $(flags vanilla)
rm -rf $HOST/discard_warmup
for r in 1 2 3; do
  case $r in 1) O="vanilla sllm mukv";; 2) O="sllm mukv vanilla";; 3) O="mukv vanilla sllm";; esac
  LOG "repeat $r order: $O"
  for a in $O; do cell ${a}_r$r $(flags $a); done
done
LOG "PREFILL_V5_DONE"
touch /tmp/prefill_v5_DONE
