#!/system/bin/sh
# 2026-07-21 — matched Llama-1B GPU comparison.
#
# vanilla: clean llama.cpp snapshot e03fdcf / llama-completion, with its normal
#          Vulkan defaults. No μKV code or policy flags are present.
# mukv:    current EndurKV eviction_bench with μKV mass + Solution 2 FA-on.
#
# The phone's GPU governor and charging state are deliberately NOT changed.
# Both cells wait for the same cool gate, use the same model/prompt/context,
# batch sizes, threads, GPU offload, seed, and 4096-token decode length.
# GPU-side defrag is unavailable by design; record that rather than implying it.

set -u
OUT=${1:?output directory required}
WORK=/data/local/tmp/endurkv
STOCK=$WORK/bin_vulkan_stock_e03
MUKV_BIN=$WORK/bin_vulkan_new/eviction_bench
MUKV_LIB=$WORK/bin_vulkan
MODEL=$WORK/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
PROMPT=$WORK/logs/gpu8b_smoke/prompt.txt

mkdir -p "$OUT"
cp "$PROMPT" "$OUT/prompt.txt"
sha256sum "$OUT/prompt.txt" > "$OUT/prompt.sha256"
date -Iseconds > "$OUT/started_at.txt"

cool() {
  while true; do
    set -- $(su -c 'g=0;d=0;s=0;b=0; for z in /sys/class/thermal/thermal_zone*; do n=$(cat $z/type); t=$(( $(cat $z/temp)/1000 )); case $n in gpuss-*) [ $t -gt $g ] && g=$t;; ddr) d=$t;; shell_front) s=$t;; battery) b=$t;; esac; done; echo $g $d $s $b')
    if [ "${1:-99}" -le 37 ] && [ "${2:-99}" -le 37 ] && [ "${3:-99}" -le 34 ] && [ "${4:-99}" -le 34 ]; then
      echo "cold_gate: GPUSS=$1 DDR=$2 shell=$3 battery=$4"; return
    fi
    sleep 20
  done
}

run_cell() {
  NAME=$1; shift
  D="$OUT/$NAME"; mkdir -p "$D"
  cool | tee "$D/cold_gate.txt"
  sh "$WORK/scripts/sample_sensors.sh" --out "$D/sensors.csv" --hz 5 >/dev/null 2>&1 & SPID=$!
  sleep 1
  # Android mksh arithmetic is 32-bit on this phone; nanosecond epoch values
  # overflow. Whole seconds are sufficient for the auxiliary wall clock.
  T0=$(date +%s)
  "$@" > "$D/stdout.log" 2> "$D/stderr.log"
  RC=$?
  T1=$(date +%s)
  kill "$SPID" 2>/dev/null
  echo "$RC" > "$D/exit_status.txt"
  echo $(( (T1-T0)*1000 )) > "$D/wall_ms.txt"
  date -Iseconds > "$D/finished_at.txt"
}

# No clock cap, watchdog, charging toggle, or Flash-Attention override: normal
# llama.cpp CLI settings, except matched workload and resource parameters.
mkdir -p "$OUT/vanilla_stock"
sha256sum "$STOCK/llama-completion" "$STOCK/libllama.so" "$STOCK/libggml-vulkan.so" > "$OUT/vanilla_stock/binary_and_libs.sha256"
run_cell vanilla_stock env LD_LIBRARY_PATH="$STOCK" "$STOCK/llama-completion" \
  -m "$MODEL" -f "$OUT/prompt.txt" -n 4096 -c 16384 -b 512 -ub 64 \
  -t 4 -tb 4 -ngl 99 --temp 0 --seed 42 --no-warmup --no-display-prompt -no-cnv

# μKV mass + Solution 2. GPU defrag is intentionally unavailable (the runner's
# n_gpu_layers=99 causes the CPU-only defrag guard to skip it).
mkdir -p "$OUT/mukv_mass_sol2_gpu_no_defrag"
sha256sum "$MUKV_BIN" "$MUKV_LIB/libllama.so" "$MUKV_LIB/libggml-vulkan.so" > "$OUT/mukv_mass_sol2_gpu_no_defrag/binary_and_libs.sha256"
run_cell mukv_mass_sol2_gpu_no_defrag env LD_LIBRARY_PATH="$MUKV_LIB" "$MUKV_BIN" \
  --prompt "$OUT/prompt.txt" --prompt-id mukv_mass_sol2_gpu_no_defrag --eval-mode gen \
  --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
  --model "$MODEL" --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
  --policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 \
  --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 \
  --out-meta "$OUT/mukv_mass_sol2_gpu_no_defrag/meta.json" \
  --out-csv "$OUT/mukv_mass_sol2_gpu_no_defrag/gen_steps.csv" \
  --out-prefill-csv "$OUT/mukv_mass_sol2_gpu_no_defrag/gen_prefill.csv" --out-gen /dev/null

date -Iseconds > "$OUT/finished_at.txt"
touch "$OUT/DONE"
