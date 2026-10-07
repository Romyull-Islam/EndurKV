#!/bin/bash
# run_gpu_8b_throttle_probe.sh: find the battery / skin temperature at which the Adreno
# GPU clock drops under sustained Bonsai-8B (1-bit) decode, to anchor gpu_watchdog_v5.
# No watchdog. Llama-1B never throttled on the GPU, so the heavier 8B model is used.
# Step 0: smoke test that Q1_0 runs on the Vulkan backend and does not fall back to CPU.
# Step 1: sustained decode with sensors at 5 Hz (enable the probe_run line after step 0).
set -u
export ANDROID_ADB_SERVER_PORT=5151            # tunnel port (NOT 5037)
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/gpu_8b_probe; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/gpu8b_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR="${SCR:-$(cd "$(dirname "$0")/../.." && pwd)/eval_corpora}"
adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
VKLIB=/data/local/tmp/endurkv/bin_vulkan
VKBIN=/data/local/tmp/endurkv/bin_vulkan_new/eviction_bench
set_charging(){ adb_safe_shell "su -c 'echo $1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }

echo "[$(date +%H:%M:%S)] === STEP 0: Vulkan Q1_0 smoke test (16 tokens, n-gpu-layers 99) ==="
adb_safe_shell "su -c 'cat /sys/kernel/gpu/gpu_busy_percentage 2>/dev/null; cat /proc/stat >/dev/null'" < /dev/null >/dev/null 2>&1
adb_safe_shell "LD_LIBRARY_PATH=$VKLIB $VKBIN --prompt $OUT/prompt.txt --prompt-id smoke \
  --eval-mode gen --max-tokens 16 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 \
  --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 --policy vanilla \
  --out-meta $OUT/smoke.json --out-gen /dev/null --out-csv /dev/null \
  > $OUT/smoke.out 2> $OUT/smoke.err" < /dev/null
adb pull "$OUT/smoke.json" "$OUT_HOST/smoke.json" < /dev/null >/dev/null 2>&1
adb pull "$OUT/smoke.err"  "$OUT_HOST/smoke.err"  < /dev/null >/dev/null 2>&1
echo "smoke.err (backend / device / any 'not supported' / fallback)"
adb_safe_shell "grep -iE 'vulkan|adreno|gpu|offload|not support|no shader|fallback|error|assert|device' $OUT/smoke.err | head -20" < /dev/null
echo "smoke decode_tps"
adb_safe_shell "grep -oE '\"decode_tps\": *[0-9.]+' $OUT/smoke.json 2>/dev/null" < /dev/null
echo
echo "  >>> INSPECT ABOVE: if it shows a Vulkan/Adreno device and layers offloaded to GPU, continue."
echo "  >>> If it fell back to CPU (no Vulkan device / q1_0 not supported), STOP -- 8B GPU not feasible."
echo "  (This script pauses here by design; re-run STEP 1 block after confirming GPU offload.)"

# STEP 1, after the smoke test confirms GPU offload: uncomment the probe_run line below.
# Vanilla is the heaviest sustained load, so it throttles first.
probe_run(){ local CELL=$1; shift
  local PD=$OUT/$CELL; adb_safe_shell "mkdir -p $PD" < /dev/null
  echo "[$(date +%H:%M:%S)] === STEP 1 probe: $CELL (natural, no watchdog) ==="
  set_charging 0
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2
  # long sustained decode to force the phone toward its throttle (ctx-limited ~4K after the 12K prompt)
  cat > /tmp/gpu8b_probe.sh <<EOF
#!/system/bin/sh
rm -f $PD/gen.DONE
LD_LIBRARY_PATH=$VKLIB $VKBIN --prompt $OUT/prompt.txt --prompt-id $CELL --eval-mode gen \
  --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 4 \
  --n-gpu-layers 99 --greedy --k-nominal 1024 $* \
  --out-meta $PD/meta.json --out-gen /dev/null --out-csv $PD/gen_steps.csv > $PD/gen.out 2> $PD/gen.err
touch $PD/gen.DONE
EOF
  adb push /tmp/gpu8b_probe.sh /data/local/tmp/gpu8b_probe.sh < /dev/null >/dev/null 2>&1
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/gpu8b_probe.sh >/dev/null 2>&1 &'" < /dev/null
  while :; do dn=$(adb_safe_shell "[ -f $PD/gen.DONE ] && echo Y || echo N" < /dev/null|tr -d '\r'); case "$dn" in *Y*) break;; esac; sleep 30; done
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL]"
}
# probe_run vanilla --policy vanilla --cache-type-k f16 --cache-type-v f16
set_charging 1
touch "$OUT_HOST/PROBE_STAGED"
echo "[$(date +%H:%M:%S)] smoke staged -> $OUT_HOST (inspect, then enable STEP 1)"
