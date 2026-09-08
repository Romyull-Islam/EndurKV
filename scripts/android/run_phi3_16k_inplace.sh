#!/bin/bash
# ============================================================================
# run_phi3_16k_inplace.sh -- fill the phone-GPU table's broken Phi-3 row. (2026-08-10)
#
# The table currently prints "does not fit at 16K -- see ctx 8192 below" for Phi-3 with
# compaction, and that note was honest: the phi3_mukv_dfg cells exist on disk with NO
# meta.json, i.e. they were attempted and died. But mukv_dfg is --force-defrag, the STATE
# ROUND-TRIP, which restores into a SECOND context: 2 x 6144 MiB of KV + 2.4 GB of weights
# against 15.47 GB of unified RAM. Reproduced twice with logcat attached -- exactly one of
# the two KV allocations succeeds and Android SIGKILLs six Zygote processes. The same
# failure occurs on a 24 GB RTX ("second context alloc failed"), so it belongs to the
# MECHANISM, not to phone memory.
#
# In-place compaction needs only ONE cache (6144 + 2400 = 8.5 GB) and has already been
# shown to run this exact configuration: 874 cells retained, compacted in 6.2 s. This
# script produces the timed n=3 cells the table needs, so a measured row can replace the
# note.
#
# WATCHDOG v5LOW IS ON, and only on muKV -- baselines never receive it. v5LOW (battery
# 36.0/36.5/37.0, skin 39.5/40.0/40.5) is the ladder that actually engages on GPU work;
# v5HIGH's 47 C battery anchor was measured firing ZERO times across 36 min of sustained
# GPU load because it is calibrated to the CPU deep-throttle trigger.
#
# Cool gate before EVERY cell (DDR<=35 C, batt<=33 C, charging off while cooling); a
# failed gate skips the cell rather than running it hot.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv
M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
OUT=/tmp/phone_gpu_16k
DEV=/data/local/tmp/endurkv/logs/phi3ip16k_$(date +%Y%m%d_%H%M%S)
WD_STOP=/data/local/tmp/gpu_wd.stop
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --k-nominal 1024"

adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/benchmarks/ctx_sweep/phi3_12288tok.txt $DEV/prompt.txt < /dev/null >/dev/null 2>&1
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/gpu_watchdog_v5_low.sh /data/local/tmp/gpu_wd_v5low.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM

cell(){ local TAG=$1
  local D=$OUT/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; return ;; esac
  adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/gpu_wd_v5low.sh $DEV/${TAG}_wd.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG (in-place + v5LOW) ..."
  adb_safe_shell "cd $BIN && LD_LIBRARY_PATH=$BIN ./eviction_bench --model $M \
    --prompt $DEV/prompt.txt --prompt-id $TAG --eval-mode gen --max-tokens 4096 --ignore-eos \
    --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $MU --compact-inplace --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$TAG.json --out-gen /dev/null --out-csv /dev/null \
    > $DEV/$TAG.out 2> $DEV/$TAG.err" < /dev/null
  adb_safe_shell "su -c 'touch $WD_STOP; sleep 1; echo 1200 > /sys/kernel/gpu/gpu_max_clock'" < /dev/null
  adb pull $DEV/$TAG.json "$D/meta.json" < /dev/null >/dev/null 2>&1
  adb pull $DEV/$TAG.err  "$D/err"       < /dev/null >/dev/null 2>&1
  adb pull $DEV/${TAG}_wd.log "$D/watchdog.log" < /dev/null >/dev/null 2>&1
  if [ -f "$D/meta.json" ]; then
    python3 -c "
import json,re
s=re.sub(r':\s*-?nan\b',': NaN',open('$D/meta.json').read()); j=json.loads(re.sub(r':\s*-?inf\b',': Infinity',s))
print('  [%-18s] prefill=%6.1fs tps=%6.2f cells=%6.0f compact=%s steps=%s'%('$TAG',
 j['prefill_ms']/1000, j.get('decode_tps') or 0, j['retained_kv_bytes']/(32*32*96*2*2.0),
 j.get('compaction_mode'), j.get('n_decode_steps')))"
    grep -c "clk=" "$D/watchdog.log" 2>/dev/null | sed 's/^/      watchdog steps: /'
  else echo "  [$TAG] FAILED: $(tail -1 "$D/err" 2>/dev/null | cut -c1-70)"; fi
}
cell phi3_mukv_ip
cell phi3_mukv_ip_r2
cell phi3_mukv_ip_r3
echo PHI3_IP_DONE
