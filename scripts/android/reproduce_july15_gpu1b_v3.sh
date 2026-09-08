#!/system/bin/sh
# 2026-07-21: current-binary protocol replica of the 2026-07-15 Llama-1B
# GPU μKV-mass + watchdog-v3 row. Captures raw inputs, logs, sensor trace,
# watchdog actions, and hashes. It does NOT claim historical-binary identity.
# Args: <output-dir> <source-prompt-path>

OUT=${1:?output directory required}
PROMPT=${2:?source prompt required}
BIN=/data/local/tmp/endurkv/bin_vulkan_new/eviction_bench
LIB=/data/local/tmp/endurkv/bin_vulkan
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
WATCHDOG=/data/local/tmp/gpu_watchdog_v3.sh
SENS=/data/local/tmp/endurkv/scripts/sample_sensors.sh
STOP=$OUT/watchdog.stop

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$PROMPT" "$OUT/prompt.txt"
sha256sum "$OUT/prompt.txt" > "$OUT/prompt.sha256"
sha256sum "$BIN" > "$OUT/binary.sha256"
date -Iseconds > "$OUT/started_at.txt"

echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
echo 1200 > /sys/kernel/gpu/gpu_max_clock
rm -f "$STOP"
nohup sh "$WATCHDOG" "$OUT/watchdog.log" "$STOP" >/dev/null 2>&1 &
WDPID=$!
nohup sh "$SENS" --out "$OUT/sensors.csv" --hz 5 >/dev/null 2>&1 &
SPID=$!
sleep 2

LD_LIBRARY_PATH="$LIB" "$BIN" \
  --prompt "$OUT/prompt.txt" --prompt-id mukv_mass_v3_replica --eval-mode gen \
  --max-tokens 4096 --ignore-eos --ctx-size 16384 --model "$MODEL" --seed 42 \
  --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
  --policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 \
  --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --fa-on-evict \
  --cache-type-k f16 --cache-type-v f16 \
  --out-meta "$OUT/meta.json" --out-gen /dev/null --out-csv "$OUT/gen_steps.csv" \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
STATUS=$?

touch "$STOP"
kill "$SPID" "$WDPID" 2>/dev/null
sleep 1
echo "$STATUS" > "$OUT/exit_status.txt"
cat /sys/kernel/gpu/gpu_max_clock > "$OUT/final_gpu_max_clock_mhz.txt"
date -Iseconds > "$OUT/finished_at.txt"
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
touch "$OUT/DONE"
exit "$STATUS"
