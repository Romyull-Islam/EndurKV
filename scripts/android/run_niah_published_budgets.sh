#!/bin/bash
# ============================================================================
# run_niah_published_budgets.sh -- Ada-KV, TOVA and H2O on the needle grid at
# the budgets their own papers specify. (2026-09-20)
#
# The grid ran every policy at k_nominal=1024. For muKV that is its design
# budget and for SnapKV it is the published per-head budget, so those rows are
# right. For three baselines it is not:
#
#     Ada-KV  published 2048        ran at 1024
#     TOVA    published 2048        ran at 1024
#     H2O     published 20% of N    ran at 1024
#
# (StreamingLLM, published 4+2000, is corrected separately by
# run_niah_sllm2004.sh.) Reporting a baseline below its own budget and then
# scoring it on retrieval is precisely the error this paper criticises, and the
# CPU table already gives all three their published budgets, so the needle
# table contradicted it.
#
# H2O's budget is a fraction of the prompt, so it differs per model and per
# stimulus: /tmp/h2o_budgets.txt carries "<model> <stim> <n_prompt> <k>" rows
# computed from each stimulus's own measured prompt length (624 to 1447).
#
# Same stimuli, ctx, protocol and build (bin_cpu_v87) as the other rows. New tags
# (adakv2048, tova2048, h2opub) so the 1024 cells are kept. Resumable: a cell
# whose meta.json exists is skipped. The cool gate is deliberately dropped; see
# the note in cell(). No timing may be read from these cells.
# ============================================================================
set -u
for _p in ${ADB_PORTS:-5161 5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] using server port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh

CTX=${CTX:-16384}
OUT_HOST=/tmp/niah_tableC_k1024_ctx${CTX}; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/niahpub_$(date +%Y%m%d_%H%M%S)
NIAH_SRC=/home/mislam22/EndurKV_workspace/EndurKV/benchmarks/niah
CB=/data/local/tmp/endurkv/bin_cpu_v87
BUD=/tmp/h2o_budgets.txt
adb_safe_shell "mkdir -p $OUT" < /dev/null
for f in "$NIAH_SRC"/niah_L*_n0.txt; do adb push "$f" "$OUT/$(basename "$f")" < /dev/null >/dev/null 2>&1; done
STIMS=$(cd "$NIAH_SRC" && ls niah_L*_n0.txt)

declare -A MODELS=(
  [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
  [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
  [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
  [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
)

# cell <model> <tag> <stim> <k> <policy flags...>
cell(){ local MT=$1 TAG=$2 STIM=$3 KB=$4; shift 4; local id="${MT}__${TAG}__${STIM%.txt}"; local PD=$OUT/$id
  [ -f "$OUT_HOST/$id/meta.json" ] && { echo "  [$id] cached"; return; }
  adb_safe_shell "mkdir -p $PD" < /dev/null
  # NO COOL GATE HERE, deliberately. This table reports two things, retrieval hits
  # and live cells, and both are invariant to temperature: decoding is greedy with a
  # fixed seed on a fixed build, so the token sequence is identical whatever clock
  # the chip runs at, and the surviving cell count is set by the policy, not the
  # thermal state. The paper says so itself ("A quality axis only ... We read nothing
  # about speed or energy from this table"). The gate cost 160 to 714 s per cell in
  # the StreamingLLM re-run, which over 168 cells is about 8 hours of waiting for a
  # number that cannot move. The sensors are still logged, but no timing from these
  # cells may be reported; use the gated campaigns for that.
  # A guard rail replaces it: if DDR is genuinely hot, wait, so a 7-hour unattended
  # CPU soak cannot cook the phone.
  local hot=0
  while [ $hot -lt 40 ]; do
    T=$(adb_safe_shell "su -c 'cat /sys/class/thermal/thermal_zone47/temp'" < /dev/null 2>/dev/null | tr -dc '0-9')
    [ -n "$T" ] || break
    [ "$((T/1000))" -le 52 ] && break
    echo "  [$id] DDR $((T/1000))C > 52C, waiting"; sleep 60; hot=$((hot+1))
  done
  adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null >/dev/null 2>&1
  # no watchdog: muKV-only by design, as for every other baseline row
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$CB timeout ${TMO:-3600} $CB/eviction_bench --prompt $OUT/$STIM --prompt-id $id \
    --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX --model ${MODELS[$MT]} --seed 42 \
    --threads 6 --n-gpu-layers 0 --greedy --k-nominal $KB $* \
    --out-meta $PD/meta.json --out-csv /dev/null --out-gen $PD/gen.txt > $PD/c.out 2> $PD/c.err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "build=bin_cpu_v87 k_nominal=$KB published budget" > "$OUT_HOST/$id/BUILD.txt" 2>/dev/null
  local h=miss; grep -qi 'mango' "$OUT_HOST/$id/gen.txt" 2>/dev/null && h=HIT
  echo "  [$id] k=$KB $h"
}

for MT in llama1b gemma2b phi3 bonsai8b; do
  i=0
  for STIM in $STIMS; do
    i=$((i+1)); echo "[$(date +%H:%M:%S)] $MT ($i/14) $STIM"
    HK=$(awk -v m=$MT -v s=${STIM%.txt} '$1==m && $2==s {print $4}' $BUD)
    [ -n "$HK" ] || { echo "  no H2O budget for $MT $STIM; skipping H2O"; }
    cell "$MT" adakv2048 "$STIM" 2048 --policy adakv --n-sink 0 --obs-window 32
    cell "$MT" tova2048  "$STIM" 2048 --policy tova
    [ -n "$HK" ] && cell "$MT" h2opub "$STIM" "$HK" --policy h2o --n-sink 0 --obs-window 64
  done
  echo "[$(date +%H:%M:%S)] === $MT published-budget block done ==="
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
touch /tmp/niah_pubbud_DONE
echo "[$(date +%H:%M:%S)] NIAH PUBLISHED BUDGETS DONE -> $OUT_HOST"
