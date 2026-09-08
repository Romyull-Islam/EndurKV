#!/bin/bash
# ============================================================================
# GPU 8B THROTTLE PROBE (2026-07-20). Goal: find at what BATTERY / SKIN temp the
# Adreno GPU clock (gpu_clk_hz) drops to its minimum under a sustained Bonsai-8B
# (1-bit) decode -- i.e. the REAL throttle trigger, to anchor gpu_watchdog_v5.
# NO watchdog here (natural). Prior GPU data is Llama-1B only; 8B is far heavier
# and may actually throttle, which 1B never did (held 1200 MHz to 94C junction).
#
# STEP 0 SMOKE: confirm Bonsai-8B (Q1_0) actually runs on the Vulkan GPU backend
#   (Q1_0 is a Prism-custom format; Vulkan may lack its shader -> would fall back
#   to CPU, making the "GPU" probe meaningless). Abort with a clear message if so.
# STEP 1 PROBE: sustained natural decode, sensors @5Hz, log gpu_clk vs bat/skin.
# ============================================================================
set -u
export ANDROID_ADB_SERVER_PORT=5151            # tunnel port (NOT 5037)
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
OUT_HOST=/tmp/gpu_8b_probe; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/gpu8b_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
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
echo "--- smoke.err (backend / device / any 'not supported' / fallback) ---"
adb_safe_shell "grep -iE 'vulkan|adreno|gpu|offload|not support|no shader|fallback|error|assert|device' $OUT/smoke.err | head -20" < /dev/null
echo "--- smoke decode_tps ---"
adb_safe_shell "grep -oE '\"decode_tps\": *[0-9.]+' $OUT/smoke.json 2>/dev/null" < /dev/null
echo
echo "  >>> INSPECT ABOVE: if it shows a Vulkan/Adreno device and layers offloaded to GPU, continue."
echo "  >>> If it fell back to CPU (no Vulkan device / q1_0 not supported), STOP -- 8B GPU not feasible."
echo "  (This script pauses here by design; re-run STEP 1 block after confirming GPU offload.)"

# ---------------------------------------------------------------------------
# STEP 1 (run after smoke confirms GPU offload): sustained natural throttle probe.
# Uncomment to run. vanilla = heaviest sustained load -> throttles soonest ->
# cleanest anchor. Sensors @5Hz capture gpu_clk_hz + battery + shell_front + gpuss.
# ---------------------------------------------------------------------------
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
