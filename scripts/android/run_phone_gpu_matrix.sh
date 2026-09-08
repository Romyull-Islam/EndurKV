#!/bin/bash
# Phone GPU (Adreno 840) matrix: 4 models x 3 policies on the SAME workload the RTX
# runs (12K wikitext prompt, 256-token decode, ctx 16384), so the two devices are
# directly comparable. Compaction is forced ON for muKV: measured 2.28x on Adreno,
# which contradicts the old "round-trip is prohibitive" assumption the default was
# built on. meta.json now records compaction_applied, so a silent fallback is visible.
# Records prefill / decode / wall separately -- the speedup is a DECODE effect and
# wall includes prefill, which eviction does not shrink.
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
VK=/data/local/tmp/endurkv/bin_vk_v2
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
OUT_HOST=/tmp/phone_gpu_matrix; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/pgm_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
adb_safe_shell "mkdir -p $OUT" < /dev/null
declare -A M=( [llama1b]=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
               [phi3]=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
               [gemma2b]=/data/local/tmp/endurkv/models/gemma-2-2b-it-Q4_K_M.gguf
               [bonsai8b]=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf )
cell(){ local TAG=$1 MP=$2; shift 2; local PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && return
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  case "$CG" in *"cool ddr="*) : ;; *) echo "  [SKIP-HOT] $TAG"; return ;; esac
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout 5400 $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 256 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model $MP --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
    --cache-type-k q8_0 --cache-type-v q8_0 $* \
    --out-meta $PD/meta.json --out-gen /dev/null --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
d,t=sys.argv[1],sys.argv[2]; f=os.path.join(d,'meta.json')
if not os.path.exists(f):
    e=os.path.join(d,'err'); m=open(e).read().strip().split('\n')[-1][:52] if os.path.exists(e) else '?'
    print("  [%-18s] FAILED %s"%(t,m))
else:
    s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); j=json.loads(s)
    print("  [%-18s] prefill=%7.1fs decode=%7.1fs wall=%7.1fs tps=%6.2f compact=%s"%(
        t,j.get('prefill_ms',0)/1000,j.get('decode_ms',0)/1000,j.get('total_ms',0)/1000,
        j.get('decode_tps') or 0,j.get('compaction_applied')))
PY
}
for k in llama1b gemma2b phi3 bonsai8b; do
  echo "[$(date +%H:%M:%S)] === $k ==="
  cell ${k}_vanilla "${M[$k]}" --policy vanilla
  cell ${k}_mukv    "${M[$k]}" $MU --force-defrag
  cell ${k}_snapkv  "${M[$k]}" --policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo PHONE_MATRIX_DONE
