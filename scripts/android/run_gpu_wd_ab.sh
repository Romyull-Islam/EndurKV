#!/bin/bash
# GPU watchdog threshold A/B (2026-07-30).
# WHY: the paper tabulates a GPU watchdog row but describes the CPU ladder, and the
# row that exists was produced by v3 (which oscillates 1200<->1050). This measures
# the SHIPPED v5 daemon at two ladders that differ ONLY in trigger temperature:
#   v5LOW  battery 36.0/36.5/37.0/37.5/38.0  skin 39.5/40.0/40.5/41.0/41.5  (v4-era anchors)
#   v5HIGH battery 47.0/48.0/48.5/49.0/49.5  skin 50.0/51.0/51.5/52.0/52.5  (CPU-mirror, shipped)
# Same code, same tiers (1200/1050/967/902/826 MHz), same junction backstop, so any
# difference is the threshold and nothing else. Each arm gets its own same-session
# baseline; n=3; cool gate before every cell; USB-rail energy sampled throughout.
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
# 2026-07-30: the paper's GPU cells use a SPLIT binary/lib pair (see /tmp/gpu_rerun.sh):
# binary from bin_vulkan_new (2026-07-25, knows --fa-on-evict/--ignore-eos), libs from
# bin_vulkan. bin_vulkan's own binary is 2026-05-31 and rejects every current flag --
# using it silently produced 12 empty cells. PLATFORMS.md still documents the stale path.
VK=/data/local/tmp/endurkv/bin_vulkan_new
VKLIB=/data/local/tmp/endurkv/bin_vulkan
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
OUT_HOST=/tmp/gpu_wd_ab; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/gpuwdab_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
adb_safe_shell "mkdir -p $OUT" < /dev/null

cell(){ local TAG=$1 WD=$2; shift 2; local id=$TAG PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && return
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "[SKIP-HOT] $TAG"; return ;; esac
  if [ "$WD" != none ]; then
    adb_safe_shell "su -c 'rm -f /data/local/tmp/gpu_wd.stop; nohup sh $WD $OUT/${TAG}_wd.log /data/local/tmp/gpu_wd.stop >/dev/null 2>&1 &'" < /dev/null
  fi
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VKLIB timeout 2400 $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $PD/meta.json --out-gen /dev/null --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'touch /data/local/tmp/gpu_wd.stop; pkill -f sample_sensors 2>/dev/null; echo 1200 > /sys/kernel/gpu/gpu_max_clock'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  [ "$WD" != none ] && adb pull "$OUT/${TAG}_wd.log" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  local t=$(python3 -c "
import json,re
s=open('$OUT_HOST/$TAG/meta.json').read(); s=re.sub(r':\s*-?nan\b',': NaN',s)
j=json.loads(s); print('tps=%.1f wall=%.0fs'%(j.get('decode_tps') or 0, j.get('total_ms',0)/1000))" 2>/dev/null)
  echo "  [$TAG] $t"
}
for i in 1 2 3; do
  cell vanilla_$i   none                                        --policy vanilla
  cell mukv_nowd_$i none                                        $MU
  cell mukv_low_$i  /data/local/tmp/gpu_wd_v5low.sh             $MU
  cell mukv_high_$i /data/local/tmp/gpu_watchdog_v5_real.sh     $MU
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo "DONE -> $OUT_HOST"

# chain: hand the phone back to the Table C campaign (resumable, skips done cells)
echo "[$(date +%H:%M:%S)] restarting Table C campaign"
nohup bash /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/run_niah_tableC.sh >> /tmp/tableC.log 2>&1 &
