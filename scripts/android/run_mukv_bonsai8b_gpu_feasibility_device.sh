#!/system/bin/sh
# 2026-07-21 — μKV-only Bonsai/Prism 8B Vulkan feasibility run.
#
# A 16,384-token f16 GPU KV allocation is known to be killed during context
# reservation (2,304 MiB).  This uses the same 9,737-token workload, 128-token
# decode, and 10,000-token context (1,406 MiB f16 KV) to test the real μKV
# GPU path without claiming full-16k feasibility.  No watchdog, GPU cap,
# charging toggle, or vanilla baseline is used.

set -u
OUT=${1:?output directory required}
WORK=/data/local/tmp/endurkv
BIN=$WORK/bin_vulkan_new/eviction_bench
LIB=$WORK/bin_vulkan
MODEL=$WORK/models/Bonsai-8B-Q1_0.gguf
PROMPT=$WORK/logs/gpu8b_smoke/prompt.txt
D="$OUT/mukv_bonsai8b_gpu_ctx10k_d128"
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

date -Iseconds > "$D/started_at.txt"
cool | tee "$D/cold_gate.txt"
cat /proc/meminfo > "$D/meminfo_before.txt"
su -c 'cat /sys/class/kgsl/kgsl-3d0/gpubusy /sys/kernel/gpu/gpu_max_clock 2>/dev/null' > "$D/gpu_state_before.txt"
sh "$WORK/scripts/sample_sensors.sh" --out "$D/sensors.csv" --hz 5 >/dev/null 2>&1 & SPID=$!
sleep 1
LD_LIBRARY_PATH="$LIB" "$BIN" --prompt "$D/prompt.txt" --prompt-id mukv_bonsai8b_gpu_ctx10k_d128 \
  --eval-mode gen --max-tokens 128 --ignore-eos --ctx-size 10000 --n-batch 512 --n-ubatch 64 \
  --model "$MODEL" --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
  --policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 \
  --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 \
  --out-meta "$D/meta.json" --out-csv "$D/gen_steps.csv" --out-prefill-csv "$D/gen_prefill.csv" --out-gen /dev/null \
  > "$D/stdout.log" 2> "$D/stderr.log"
RC=$?
kill "$SPID" 2>/dev/null
echo "$RC" > "$D/exit_status.txt"
cat /proc/meminfo > "$D/meminfo_after.txt"
su -c 'cat /sys/class/kgsl/kgsl-3d0/gpubusy /sys/kernel/gpu/gpu_max_clock 2>/dev/null' > "$D/gpu_state_after.txt"
date -Iseconds > "$D/finished_at.txt"
touch "$D/DONE"
exit "$RC"
