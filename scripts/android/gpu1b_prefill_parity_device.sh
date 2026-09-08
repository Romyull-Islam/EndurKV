#!/system/bin/sh
# 2026-07-21: device half of the GPU prefill-parity protocol.
# Measures the EndurKV binary with μKV disabled (`--policy vanilla`) versus
# μKV-mass with one generated token. It is an internal regression control, not
# an upstream, unmodified llama.cpp baseline. The clock keeper applies the same
# 1200 MHz ceiling to both policies; this is not a watchdog test.
# Args: <output-dir> <vanilla|mukv>

OUT=${1:?output directory required}
MODE=${2:?mode required}
BIN=/data/local/tmp/endurkv/bin_vulkan_new/eviction_bench
LIB=/data/local/tmp/endurkv/bin_vulkan
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
PROMPT=/data/local/tmp/endurkv/logs/gpu8b_smoke/prompt.txt
case "$MODE" in
  vanilla) POL='--policy vanilla' ;;
  mukv) POL='--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --fa-on-evict --cache-type-k f16 --cache-type-v f16' ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$PROMPT" "$OUT/prompt.txt"
sha256sum "$PROMPT" > "$OUT/prompt.sha256"
sha256sum "$BIN" > "$OUT/binary.sha256"
date -Iseconds > "$OUT/started_at.txt"
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
echo 1200 > /sys/kernel/gpu/gpu_max_clock
( while [ ! -f "$OUT/DONE" ]; do echo 1200 > /sys/kernel/gpu/gpu_max_clock; sleep 2; done ) &
KPID=$!
nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out "$OUT/sensors.csv" --hz 5 >/dev/null 2>&1 &
SPID=$!
sleep 2
LD_LIBRARY_PATH="$LIB" "$BIN" --prompt "$OUT/prompt.txt" --prompt-id "prefill_parity_${MODE}" \
  --eval-mode gen --max-tokens 1 --ignore-eos --ctx-size 16384 --model "$MODEL" --seed 42 \
  --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 $POL \
  --out-meta "$OUT/meta.json" --out-gen /dev/null --out-csv "$OUT/gen_steps.csv" \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
STATUS=$?
kill "$SPID" "$KPID" 2>/dev/null
echo "$STATUS" > "$OUT/exit_status.txt"
date -Iseconds > "$OUT/finished_at.txt"
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
touch "$OUT/DONE"
exit "$STATUS"
