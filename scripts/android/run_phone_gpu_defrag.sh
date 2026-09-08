#!/bin/bash
# Phone-GPU compaction A/B (2026-08-01).
# On CUDA (RTX 4500) compaction took muKV from 1.03x to 2.42x vs vanilla. The Adreno
# path has compaction OFF because an early state-swap measurement called the round-trip
# "prohibitive" -- the SAME generalisation that silently cost us the 2.42x on CUDA.
# This measures it on Adreno instead of trusting it. Phi-3-mini is the fat-KV model
# (no GQA, 32 layers), so it is where a bandwidth win can show.
set -u
for _p in ${ADB_PORTS:-5152 5037 5151}; do
  (exec 3<>/dev/tcp/127.0.0.1/$_p) 2>/dev/null || continue
  exec 3<&- 2>/dev/null
  if ANDROID_ADB_SERVER_PORT=$_p adb devices 2>/dev/null | grep -qw device; then export ANDROID_ADB_SERVER_PORT=$_p; break; fi
done
echo "[adb] port ${ANDROID_ADB_SERVER_PORT:-unset}"
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
VK=/data/local/tmp/endurkv/bin_vk_v2                 # binary AND libs, both new
M=/data/local/tmp/endurkv/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
OUT_HOST=/tmp/phone_gpu_dfg; mkdir -p "$OUT_HOST"
OUT=/data/local/tmp/endurkv/logs/pgdfg_$(date +%Y%m%d_%H%M%S)
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
adb_safe_shell "mkdir -p $OUT" < /dev/null
cell(){ local TAG=$1; shift; local PD=$OUT/$TAG
  [ -f "$OUT_HOST/$TAG/meta.json" ] && return
  adb_safe_shell "mkdir -p $PD" < /dev/null
  CG=$(adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null)
  echo "$CG" | tail -1
  case "$CG" in *"cool ddr="*) : ;; *) echo "[SKIP-HOT] $TAG"; return ;; esac
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  adb_safe_shell "LD_LIBRARY_PATH=$VK timeout 3600 $VK/eviction_bench --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 128 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 \
    --model $M --seed 42 --threads 4 --n-gpu-layers 99 --greedy --k-nominal 1024 \
    --cache-type-k q8_0 --cache-type-v q8_0 $* \
    --out-meta $PD/meta.json --out-gen /dev/null --out-csv /dev/null > $PD/out 2> $PD/err" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  python3 - "$OUT_HOST/$TAG" "$TAG" <<'PY'
import json,re,sys,os
d,t=sys.argv[1],sys.argv[2]
f=os.path.join(d,'meta.json')
if not os.path.exists(f):
    e=os.path.join(d,'err'); m=open(e).read().strip().split('\n')[-1][:60] if os.path.exists(e) else '?'
    print("  [%-16s] FAILED %s"%(t,m))
else:
    s=open(f).read(); s=re.sub(r':\s*-?nan\b',': NaN',s); j=json.loads(s)
    print("  [%-16s] tps=%6.2f  prefill=%7.1fs  wall=%7.1fs  retained=%7.1f MiB"%(
        t,j.get('decode_tps') or 0,j.get('prefill_ms',0)/1000,j.get('total_ms',0)/1000,
        (j.get('retained_kv_bytes') or 0)/1048576))
PY
}
for i in 1 2; do
  cell vanilla_$i    --policy vanilla
  cell mukv_nodfg_$i $MU --no-defrag
  cell mukv_dfg_$i   $MU --force-defrag
done
adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; charging_restore'" < /dev/null
echo PHONE_GPU_DFG_DONE
