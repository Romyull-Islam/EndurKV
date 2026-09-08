#!/system/bin/sh
# 2026-07-22 — reliable μKV Bonsai/Prism 8B CPU run.
# Same WikiText workload as the CPU study. μKV uses mass + Solution-2 FA-on
# and, because n_gpu_layers=0, the CPU defrag/state-transfer path is active.
# Decode is guarded: abort and retain evidence after a 60-s stall or sustained
# rate below 1 token/s across a 120-s window. This prevents infinite/too-slow
# decode from being mistaken for a successful run.

set -u
OUT=${1:?output directory required}
WORK=/data/local/tmp/endurkv
BIN=$WORK/bin_cpu_sol2/eviction_bench
LIB=$WORK/bin_cpu_sol2
MODEL=$WORK/models/Bonsai-8B-Q1_0.gguf
PROMPT=$WORK/logs/gpu8b_smoke/prompt.txt
D="$OUT/mukv_mass_sol2_faon_cpu_defrag"
mkdir -p "$D"
cp "$PROMPT" "$D/prompt.txt"
sha256sum "$BIN" "$MODEL" > "$D/input.sha256"

cool() {
  while true; do
    set -- $(su -c 'g=0;d=0;s=0;b=0;c=0; for z in /sys/class/thermal/thermal_zone*; do n=$(cat $z/type); t=$(( $(cat $z/temp)/1000 )); case $n in gpuss-*) [ $t -gt $g ] && g=$t;; ddr) d=$t;; shell_front) s=$t;; battery) b=$t;; cpu-hw-trip-*) ;; cpu-*|cpullc-*) [ $t -gt $c ] && c=$t;; esac; done; echo $g $d $s $b $c')
    if [ "${2:-99}" -le 37 ] && [ "${3:-99}" -le 34 ] && [ "${4:-99}" -le 34 ] && [ "${5:-99}" -le 45 ]; then
      echo "cold_gate: GPUSS=$1 DDR=$2 shell=$3 battery=$4 CPU=$5"; return
    fi
    sleep 20
  done
}

cool | tee "$D/cold_gate.txt"
cat /proc/meminfo > "$D/meminfo_before.txt"
sh "$WORK/scripts/sample_sensors.sh" --out "$D/sensors.csv" --hz 5 >/dev/null 2>&1 & SPID=$!
date -Iseconds > "$D/started_at.txt"
LD_LIBRARY_PATH="$LIB" "$BIN" --prompt "$D/prompt.txt" --prompt-id mukv_mass_sol2_faon_cpu_defrag \
  --eval-mode gen --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
  --model "$MODEL" --seed 42 --threads 6 --n-gpu-layers 0 --greedy --k-nominal 1024 \
  --policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 \
  --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 \
  --out-meta "$D/meta.json" --out-csv "$D/gen_steps.csv" --out-prefill-csv "$D/gen_prefill.csv" --out-gen /dev/null \
  > "$D/stdout.log" 2> "$D/stderr.log" & PID=$!

PREFILL_T0=$(date +%s); DECODE=0; LAST_N=0; LAST_CHANGE=$(date +%s); WN=0; WT=0; STOP=0
while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s)
  N=$(wc -l < "$D/gen_steps.csv" 2>/dev/null); N=${N:-0}
  if [ "$N" -gt 1 ]; then
    [ "$DECODE" = 0 ] && { DECODE=1; LAST_CHANGE=$NOW; WN=$N; WT=$NOW; echo "decode_started" > "$D/decode_guard.log"; }
    [ "$N" -gt "$LAST_N" ] && LAST_CHANGE=$NOW
    if [ $(( NOW - LAST_CHANGE )) -ge 60 ]; then echo "decode_stall_60s" > "$D/status.txt"; STOP=1; kill "$PID" 2>/dev/null; fi
    if [ $(( NOW - WT )) -ge 120 ]; then
      DN=$((N-WN)); DT=$((NOW-WT));
      if [ "$DN" -lt "$DT" ]; then echo "decode_below_1_tps_${DN}_tokens_${DT}s" > "$D/status.txt"; STOP=1; kill "$PID" 2>/dev/null; fi
      WN=$N; WT=$NOW
    fi
  elif [ $(( NOW - PREFILL_T0 )) -ge 1800 ]; then
    echo "prefill_timeout_1800s" > "$D/status.txt"; STOP=1; kill "$PID" 2>/dev/null
  fi
  LAST_N=$N
  sleep 5
done
wait "$PID"; RC=$?
kill "$SPID" 2>/dev/null
echo "$RC" > "$D/exit_status.txt"
[ "$STOP" = 0 ] && echo "completed" > "$D/status.txt"
cat /proc/meminfo > "$D/meminfo_after.txt"
date -Iseconds > "$D/finished_at.txt"
touch "$D/DONE" "$OUT/DONE"
exit 0
