#!/system/bin/sh
# 2026-07-21 — vanilla Bonsai/Prism 8B Vulkan run using the CPU WikiText-2
# workload: 9,737-token prompt + 4,096 decode, ctx=16,384.  Full GPU offload
# is deliberately bounded: if prefill has not finished within 600 s, terminate
# it and preserve logs/sensors as a prefill-timeout result.

set -u
OUT=${1:?output directory required}
WORK=/data/local/tmp/endurkv
BIN=$WORK/bin_vulkan_new/eviction_bench
LIB=$WORK/bin_vulkan
MODEL=$WORK/models/Bonsai-8B-Q1_0.gguf
PROMPT=$WORK/logs/gpu8b_smoke/prompt.txt
D="$OUT/vanilla_bonsai8b_gpu99_wt2_16k"
mkdir -p "$D"
cp "$PROMPT" "$D/prompt.txt"
sha256sum "$BIN" "$LIB/libllama.so" "$LIB/libggml-vulkan.so" "$MODEL" > "$D/input.sha256"

cool() {
  while true; do
    set -- $(su -c 'g=0;d=0;s=0;b=0; for z in /sys/class/thermal/thermal_zone*; do n=$(cat $z/type); t=$(( $(cat $z/temp)/1000 )); case $n in gpuss-*) [ $t -gt $g ] && g=$t;; ddr) d=$t;; shell_front) s=$t;; battery) b=$t;; esac; done; echo $g $d $s $b')
    if [ "${1:-99}" -le 37 ] && [ "${2:-99}" -le 37 ] && [ "${3:-99}" -le 34 ] && [ "${4:-99}" -le 34 ]; then
      echo "cold_gate: GPUSS=$1 DDR=$2 shell=$3 battery=$4"; return
    fi
    sleep 20
  done
}

cool | tee "$D/cold_gate.txt"
cat /proc/meminfo > "$D/meminfo_before.txt"
su -c 'cat /sys/kernel/gpu/gpu_max_clock /sys/kernel/gpu/gpu_min_clock /sys/class/kgsl/kgsl-3d0/gpubusy 2>/dev/null' > "$D/gpu_state_before.txt"
sh "$WORK/scripts/sample_sensors.sh" --out "$D/sensors.csv" --hz 5 >/dev/null 2>&1 & SPID=$!
date -Iseconds > "$D/started_at.txt"
LD_LIBRARY_PATH="$LIB" "$BIN" --prompt "$D/prompt.txt" --prompt-id vanilla_bonsai8b_gpu99_wt2_16k \
  --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
  --model "$MODEL" --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
  --policy vanilla --cache-type-k f16 --cache-type-v f16 \
  --out-meta "$D/meta.json" --out-csv "$D/gen_steps.csv" --out-prefill-csv "$D/gen_prefill.csv" --out-gen /dev/null \
  > "$D/stdout.log" 2> "$D/stderr.log" & PID=$!

T0=$(date +%s); PREFILL_DONE=0; TIMEOUT=0
while kill -0 "$PID" 2>/dev/null; do
  test -s "$D/gen_prefill.csv" && PREFILL_DONE=1
  if [ "$PREFILL_DONE" = 0 ] && [ $(( $(date +%s) - T0 )) -ge 600 ]; then
    echo "prefill_timeout_600s" > "$D/status.txt"; TIMEOUT=1; kill "$PID" 2>/dev/null
  fi
  sleep 5
done
wait "$PID"; RC=$?
kill "$SPID" 2>/dev/null
echo "$RC" > "$D/exit_status.txt"
[ "$TIMEOUT" = 0 ] && echo "completed_or_driver_exit" > "$D/status.txt"
cat /proc/meminfo > "$D/meminfo_after.txt"
date -Iseconds > "$D/finished_at.txt"
touch "$D/DONE"
touch "$OUT/DONE"
exit 0
