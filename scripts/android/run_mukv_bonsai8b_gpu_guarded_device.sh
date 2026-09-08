#!/system/bin/sh
# 2026-07-21 — resilient μKV Bonsai/Prism 8B hybrid-GPU runner.
#
# Full offload (99) allocates a 2,304 MiB f16 Vulkan KV cache at ctx=16k and
# has previously been killed/DeviceLost.  A partial layer split distributes KV
# buffers by layer between Vulkan and CPU RAM.  Try the fastest safe split
# first; after a Vulkan failure, record it and restart at a smaller split.
# This preserves inference availability; it cannot guarantee an Android driver
# will never reset.

set -u
OUT=${1:?output directory required}
WORK=/data/local/tmp/endurkv
BIN=$WORK/bin_vulkan_new/eviction_bench
LIB=$WORK/bin_vulkan
MODEL=$WORK/models/Bonsai-8B-Q1_0.gguf
PROMPT=$WORK/logs/gpu8b_smoke/prompt.txt
MU='--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70'
mkdir -p "$OUT"
cp "$PROMPT" "$OUT/prompt.txt"
sha256sum "$BIN" "$LIB/libllama.so" "$LIB/libggml-vulkan.so" "$MODEL" > "$OUT/input.sha256"

cool() {
  while true; do
    set -- $(su -c 'g=0;d=0;s=0;b=0; for z in /sys/class/thermal/thermal_zone*; do n=$(cat $z/type); t=$(( $(cat $z/temp)/1000 )); case $n in gpuss-*) [ $t -gt $g ] && g=$t;; ddr) d=$t;; shell_front) s=$t;; battery) b=$t;; esac; done; echo $g $d $s $b')
    if [ "${1:-99}" -le 37 ] && [ "${2:-99}" -le 37 ] && [ "${3:-99}" -le 34 ] && [ "${4:-99}" -le 34 ]; then
      echo "cold_gate: GPUSS=$1 DDR=$2 shell=$3 battery=$4"; return
    fi
    sleep 20
  done
}

for LAYERS in 24 18 12 0; do
  D="$OUT/layers_${LAYERS}"; mkdir -p "$D"
  cool | tee "$D/cold_gate.txt"
  cat /proc/meminfo > "$D/meminfo_before.txt"
  sh "$WORK/scripts/sample_sensors.sh" --out "$D/sensors.csv" --hz 5 >/dev/null 2>&1 & SPID=$!
  LD_LIBRARY_PATH="$LIB" "$BIN" --prompt "$OUT/prompt.txt" --prompt-id "mukv_bonsai8b_l${LAYERS}" \
    --eval-mode gen --max-tokens 512 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model "$MODEL" --seed 42 --threads 4 --n-gpu-layers "$LAYERS" --greedy --k-nominal 1024 \
    $MU --out-meta "$D/meta.json" --out-csv "$D/gen_steps.csv" --out-prefill-csv "$D/gen_prefill.csv" --out-gen /dev/null \
    > "$D/stdout.log" 2> "$D/stderr.log"
  RC=$?
  kill "$SPID" 2>/dev/null
  echo "$RC" > "$D/exit_status.txt"
  cat /proc/meminfo > "$D/meminfo_after.txt"
  if [ "$RC" = 0 ] && test -s "$D/meta.json"; then
    echo "$LAYERS" > "$OUT/successful_n_gpu_layers.txt"
    touch "$OUT/DONE"
    exit 0
  fi
  # A failed Vulkan attempt is retained; allow thermal state to recover before
  # trying the next lower split.
  sleep 30
done
touch "$OUT/DONE"
exit 1
