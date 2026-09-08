#!/system/bin/sh
# Feasibility run: retain full GPU execution but reduce only the allocated
# context to 12k, enough for the fixed 10,074-token prompt + 512-token decode.
set -u
OUT=${1:?output directory required}
WORK=/data/local/tmp/endurkv
BIN=$WORK/bin_vulkan_new/eviction_bench
LIB=$WORK/bin_vulkan
MODEL=$WORK/models/Bonsai-8B-Q1_0.gguf
PROMPT=${2:?prompt file required}
CTX=12288
D="$OUT/mukv_mass_sol2_faon_gpu99_12k"
mkdir -p "$D"; cp "$PROMPT" "$D/prompt.txt"
sha256sum "$BIN" "$LIB/libllama.so" "$LIB/libggml-vulkan.so" "$MODEL" "$D/prompt.txt" > "$D/input.sha256"

cool() {
  while true; do
    set -- $(su -c 'g=0; d=0; s=0; b=0; for z in /sys/class/thermal/thermal_zone*; do n=$(cat $z/type); t=$(( $(cat $z/temp) / 1000 )); case $n in gpuss-*) [ $t -gt $g ] && g=$t;; ddr) d=$t;; shell_back) s=$t;; battery) b=$t;; esac; done; echo $g $d $s $b')
    if [ "${1:-99}" -le 37 ] && [ "${2:-99}" -le 37 ] && [ "${3:-99}" -le 34 ] && [ "${4:-99}" -le 34 ]; then echo "GPUSS=$1 DDR=$2 shell_back=$3 battery=$4"; return; fi
    sleep 20
  done
}

date -Iseconds > "$D/started_at.txt"; cool > "$D/cold_gate.txt"
sh "$WORK/scripts/sample_sensors.sh" --out "$D/sensors.csv" --hz 5 >/dev/null 2>&1 & SPID=$!
LD_LIBRARY_PATH="$LIB" "$BIN" --prompt "$D/prompt.txt" --prompt-id mukv_mass_sol2_faon_gpu99_12k \
  --eval-mode gen --max-tokens 512 --ignore-eos --ctx-size "$CTX" --n-batch 512 --n-ubatch 64 \
  --model "$MODEL" --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
  --policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 \
  --out-meta "$D/meta.json" --out-csv "$D/gen_steps.csv" --out-prefill-csv "$D/gen_prefill.csv" --out-gen /dev/null > "$D/stdout.log" 2> "$D/stderr.log" &
PID=$!; START=$(date +%s); STATUS=running
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s); AGE=$((NOW-START))
  if [ ! -s "$D/gen_prefill.csv" ] && [ "$AGE" -ge 600 ]; then STATUS=prefill_timeout_600s; kill "$PID" 2>/dev/null; break; fi
  if [ -s "$D/gen_prefill.csv" ] && [ "$AGE" -ge 1200 ]; then STATUS=run_timeout_1200s; kill "$PID" 2>/dev/null; break; fi
  sleep 5
done
wait "$PID" 2>/dev/null; RC=$?
kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
[ "$STATUS" = running ] && STATUS="exit_${RC}"
echo "$STATUS" > "$D/status.txt"; echo "$RC" > "$D/exit_status.txt"; date -Iseconds > "$D/finished_at.txt"; touch "$D/DONE"
