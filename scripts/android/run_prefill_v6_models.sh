#!/bin/bash
# ============================================================================
# run_prefill_v6_models.sh -- close the prefill-vs-vanilla gap for the two models
# that have no pair at all: Bonsai-8B and gemma-2. (2026-08-29)
#
# Llama-3.2-1B and Phi-3-mini-128k already have 35 archival pairs (medians +1.9%
# to +2.7%) plus v5's pinned-clock +0.7%. Bonsai-8B and gemma-2 have none, so the
# per-model prefill claim cannot currently be made for them.
#
# Method is v5's, which is the only one that worked: pin the CPU clock with
# pin_dvfs.sh (performance governor at a fixed scaling_max_freq) instead of trying
# to control temperature. v1-v4 all chased temperature and failed, because the CPU
# sheds shallow heat within ~1 s of load stopping; temperature was only ever a
# proxy for the clock. At a pinned clock v5 got vanilla 227.0 / muKV 228.5 /
# sllm 227.4 s with SD 1-3 s.
#
# IMPORTANT -- gemma-2's trained window is 8192 tokens (gemma2.context_length),
# so it gets a truncated ~5.8K prompt. The 9737-token WikiText prompt used
# everywhere else is 1.2x outside that window; running it there is what made all
# five gemma cells of the compaction campaign unusable.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
LOG(){ echo "[$(date +%H:%M:%S)] $*"; }

exec 9>/tmp/.endurkv_queue.lock; flock 9
DEV=/data/local/tmp/pf6; HOST=/tmp/prefill_v6; PIN_KHZ=1632000
SCRATCH=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
CB=/data/local/tmp/endurkv/bin_cpu_kd
MODELS=/data/local/tmp/endurkv/models
mkdir -p $HOST
adb_safe_shell "mkdir -p $DEV/wt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$SCRATCH/wikitext_16k_p12k_d4k.txt" "$DEV/wt/full.txt"  < /dev/null >/dev/null 2>&1
timeout 180 adb push "$SCRATCH/wikitext_gemma_6k.txt"     "$DEV/wt/gemma.txt" < /dev/null >/dev/null 2>&1

restore_dvfs(){ LOG "restoring DVFS"; adb_safe_shell "su -c 'sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh restore'" < /dev/null >/dev/null 2>&1; }
trap restore_dvfs EXIT INT TERM

MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
curfreq(){ adb_safe_shell "su -c 'cat /sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq 2>/dev/null'" < /dev/null 2>/dev/null | tr -d ' \r'; }

# cell <tag> <model-file> <prompt-file> <ctx> <flags...>
cell(){
  local TAG=$1 MDL=$2 PR=$3 CTX=$4; shift 4
  [ -s "$HOST/$TAG/meta.json" ] && { LOG "  $TAG cached"; return; }
  mkdir -p "$HOST/$TAG"
  CG=$(adb_safe_shell "su -c 'COOL_CPU_MAX=40 . /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null 2>/dev/null | tail -1)
  case "$CG" in *"cool ddr="*) : ;; *) LOG "  [SKIP-HOT] $TAG ($CG)"; return;; esac
  adb_safe_shell "su -c 'TARGET_MHZ=$PIN_KHZ LITTLE_TARGET_MHZ=$PIN_KHZ sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin'" < /dev/null >/dev/null 2>&1
  local F0; F0=$(curfreq)
  adb_safe_shell "rm -f $DEV/$TAG.done; setsid nohup sh -c \"timeout 5400 env LD_LIBRARY_PATH=$CB $CB/eviction_bench \
    --prompt $DEV/wt/$PR --prompt-id $TAG --eval-mode gen --max-tokens 64 --ignore-eos --ctx-size $CTX \
    --n-batch 512 --ubatch-size 64 --model $MODELS/$MDL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null > /dev/null 2> $DEV/$TAG.err ; \
    echo DONE > $DEV/$TAG.done\" >/dev/null 2>&1 &" < /dev/null
  local w=0 FMID=""
  while [ $w -lt 5500 ]; do
    adb_safe_shell "[ -f $DEV/$TAG.done ] && echo yes" < /dev/null 2>/dev/null | grep -q yes && break
    [ $w -eq 60 ] && FMID=$(curfreq)
    sleep 20; w=$((w+20)); done
  local F1; F1=$(curfreq)
  echo "pin=$PIN_KHZ f_start=$F0 f_mid=$FMID f_end=$F1" > "$HOST/$TAG/clock.txt"
  adb_safe_pull "$DEV/$TAG.json" "$HOST/$TAG/meta.json" >/dev/null 2>&1
  local PF NP
  read -r PF NP < <(python3 -c "
import json,re
try:
  j=json.loads(re.sub(r':\s*-?nan\b',': NaN',open('$HOST/$TAG/meta.json').read()))
  print('%.0f %s'%(j['prefill_ms']/1000, j.get('n_prompt_tokens','?')))
except Exception: print('-- ?')" 2>/dev/null)
  LOG "  [$TAG] prefill=${PF}s n_prompt=${NP} clock ${F0:-?}/${FMID:-?}/${F1:-?}"
}

LOG "=== prefill v6: Bonsai-8B + gemma-2, CPU pinned to ${PIN_KHZ} kHz ==="
adb_safe_shell "su -c 'TARGET_MHZ=$PIN_KHZ LITTLE_TARGET_MHZ=$PIN_KHZ sh /data/local/tmp/endurkv/scripts/pin_dvfs.sh pin'" < /dev/null 2>&1 | tail -2

for r in 1 2 3; do
  # rotate so neither arm always follows the other
  case $r in 1) O="vanilla mukv";; 2) O="mukv vanilla";; 3) O="vanilla mukv";; esac
  LOG "repeat $r order: $O"
  for a in $O; do
    case $a in
      vanilla) BF="--policy vanilla --k-nominal 1024";;
      mukv)    BF="$MU --k-nominal 1024";;
    esac
    cell bonsai_${a}_r$r Bonsai-8B-Q1_0.gguf       full.txt  16384 $BF
    cell gemma_${a}_r$r  gemma-2-2b-it-Q4_K_M.gguf gemma.txt  8192 $BF
  done
done
LOG "PREFILL_V6_DONE"
touch /tmp/prefill_v6_DONE
