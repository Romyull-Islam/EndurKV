#!/bin/bash
# ============================================================================
# run_streamingllm_faithful.sh -- StreamingLLM at its REAL strength, head to head.
# (2026-08-14)
#
# WHY THIS RUN EXISTS. The paper claims "muKV is the ONLY policy in this set that is
# FASTER than not evicting at all on a phone", and fig_retention_vs_speedup is built on
# "StreamingLLM keeps 8.0% of cells and runs at 0.23x while muKV keeps 7.4% and runs at
# 1.23x -- same retention, 5.3x apart". Both rest on a StreamingLLM arm we handicapped.
#
# THE HANDICAP, VERIFIED AT SOURCE. StreamingLLM's official implementation
# (mit-han-lab/streaming-llm, streaming_llm/kv_cache.py) evicts by PHYSICALLY
# CONCATENATING the survivors:
#     torch.cat([self.k_slice(k, 0, self.start_size),
#                self.k_slice(k, seq_len - self.recent_size, seq_len)],
#               dim=self.k_seq_dim)
# Its cache is dense and bounded by construction -- sinks plus recent window, no holes,
# ever. Compaction is not something we generously grant StreamingLLM; it is INTRINSIC to
# StreamingLLM and our harness removed it. The published phone cell ran FA-off AND
# uncompacted (777 live cells, 5.55 tok/s, 0.19x), which is StreamingLLM crippled by a
# sequence-level cell array it never uses in its own implementation.
#
# AND THE REALIZABILITY ARGUMENT DOES NOT EXCUSE IT. That argument is real for PER-HEAD
# evictors (SnapKV, Ada-KV, H2O, TOVA), whose union across heads and layers genuinely
# cannot free a cell on a shared array. StreamingLLM is SEQUENCE-LEVEL, exactly like
# muKV: 4 sinks + one contiguous recent run, trivially compactable. Grouping it with the
# per-head policies was our error, not a property of the device.
#
# ARMS. Llama-3.2-1B, phone GPU, ctx 16384, same 9737-token prompt + 4096 generated as
# the published table, so the numbers drop straight into it.
#   vanilla        full cache, the do-not-evict reference the claim is about
#   mukv           frozen config + in-place compaction, K=1024
#   sfown          StreamingLLM at ITS OWN DOCUMENTED BUDGET: FA-on + compacted,
#                  start_size=4 + recent_size=2000 = K 2004
#   sh             StreamingLLM as we published it: FA-off, uncompacted -- kept so the
#                  size of our own handicap is measured rather than asserted
#
# WHERE K=2004 COMES FROM. examples/run_streaming_llama.py in mit-han-lab/streaming-llm:
#     "--start_size",  type=int, default=4
#     "--recent_size", type=int, default=2000
# so the documented cache is 4 sinks + a 2000-token recent window = 2004 cells. An earlier
# version of this script ran StreamingLLM at K=1024 -- roughly HALF its published budget --
# which is the same class of error as the FA-off/uncompacted handicap it was written to
# correct. Both arms are now run: its own budget answers "is StreamingLLM as published
# faster than muKV", and the matched-K arm answers "at equal retention, which is faster".
# A matched-K=1024 arm was written and then REMOVED (2026-08-14). Forcing our budget onto
# a baseline is the same defect as the FA-off handicap -- it deletes the policy being
# compared. The only setting StreamingLLM is entitled to is its own. The consequence is
# that fig_retention_vs_speedup loses its "same retention, 5.3x apart" annotation: at
# K=2004 StreamingLLM retains ~15% against muKV's 7.4%, so the two are no longer
# coincidentally equal. That annotation should go. The figure's real claim -- that
# retention does not predict speed across the scatter -- needs no matched pair to stand.
#
# PINNED, AND THAT IS NOT OPTIONAL. Unpinned, this phone's GPU decode is bimodal --
# 28.20 to 39.21 tok/s from an unchanged command line, a 39% spread that makes any ratio
# below 1.39x unclaimable. taskset f0 + nice -20 collapses that to 3% (SD 4.79 -> 0.48),
# at the cost of measuring in the slower mode. That is the conservative direction for
# muKV, which is the right place to argue from.
#
# COMPACTION MODE FOR StreamingLLM: in-place. Verified equivalent to the round-trip on
# StreamingLLM at THIS context (audit R5: 0.010-0.096%). Do NOT reuse this at 64K, where
# in-place is currently broken for StreamingLLM (PPL 13.474 vs 10.432 round-trip) -- an
# open bug in our compaction, not in StreamingLLM.
#
# sllm_handicap's OUTPUT IS NUMERICALLY INVALID by construction: FA-off attaches the
# attention capture, and on this Adreno the resulting graph split corrupts the
# computation (root-caused 2026-08-14). Its TIMING is still the honest cost of the path
# we published, which is the only thing it is here to provide. No PPL cell is run for it.
# ============================================================================
set -u
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
E=/data/local/tmp/endurkv/eval_data/dis.txt      # asserted 0/119 and 0/399 vs P
DEV=/data/local/tmp/endurkv/logs/sllm_$(date +%Y%m%d_%H%M%S)
HOST=/tmp/sllm_faithful; mkdir -p $HOST
PIN="taskset f0 nice -n -20"
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024"
SF_OWN="--policy streamingllm --n-sink 4 --k-nominal 2004 --compact-inplace"
SH="--policy streamingllm --n-sink 4 --k-nominal 1024 --no-fa-positional --no-defrag"
VA="--policy vanilla --k-nominal 1024"
adb_safe_shell "mkdir -p $DEV" < /dev/null
adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/sample_sensors.sh /data/local/tmp/sample_sensors.sh < /dev/null >/dev/null 2>&1
cleanup(){ adb_safe_shell "su -c 'pkill -f sample_sensors; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null; }
trap cleanup EXIT INT TERM
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null

settle(){
  for a in 1 2 3; do
    adb_safe_shell "su -c '. /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36'" < /dev/null | tail -1
    adb_safe_shell "su -c 'prev=999; same=0; for i in \$(seq 1 90); do d=\$((\$(cat /sys/class/thermal/thermal_zone47/temp)/100)); diff=\$((d-prev)); [ \$diff -lt 0 ] && diff=\$((-diff)); if [ \$diff -le 3 ]; then same=\$((same+1)); else same=0; fi; [ \$same -ge 3 ] && break; prev=\$d; sleep 10; done'" < /dev/null >/dev/null
    R=$(adb_safe_shell "su -c 'echo \$(cat /sys/class/thermal/thermal_zone47/temp) \$(cat /sys/class/thermal/thermal_zone93/temp)'" < /dev/null | tr -d '\r')
    local _sd=$(echo $R|awk '{print int($1/1000)}'); local _sb=$(echo $R|awk '{print int($2/1000)}')
    echo "    post-settle ddr=${_sd}C batt=${_sb}C"
    [ "${_sd:-99}" -le 35 ] && [ "${_sb:-99}" -le 33 ] && return 0
  done; return 1; }

cell(){ # tag mode flags...
  local TAG=$1 MODE=$2; shift 2
  local D=$HOST/$TAG; [ -f "$D/meta.json" ] && { echo "  [$TAG] cached"; return; }
  mkdir -p "$D"
  local EX="--eval-mode gen --max-tokens 4096 --ignore-eos"
  [ "$MODE" = ppl ] && EX="--eval-mode ppl --eval-text $E"
  echo "[$(date +%H:%M:%S)] cooling for $TAG ..."; settle || { echo "  [SKIP-HOT] $TAG"; return; }
  adb_safe_shell "su -c 'rm -f /data/local/tmp/sl_$TAG.csv; nohup sh /data/local/tmp/sample_sensors.sh --out /data/local/tmp/sl_$TAG.csv --hz 2 >/dev/null 2>&1 &'" < /dev/null
  echo "[$(date +%H:%M:%S)] running $TAG ..."
  adb_safe_shell "su -c 'cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $M --prompt $P \
    --prompt-id $TAG $EX --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy \
    --cache-type-k f16 --cache-type-v f16 $* --n-batch 512 --n-ubatch 64 \
    --out-meta $DEV/$TAG.json --out-gen $DEV/$TAG.gen --out-csv /dev/null \
    > /dev/null 2> $DEV/$TAG.err'" < /dev/null
  adb_safe_shell "su -c 'pkill -f sample_sensors'" < /dev/null
  adb_safe_pull "$DEV/$TAG.json" "$D/meta.json" >/dev/null 2>&1
  adb_safe_pull "$DEV/$TAG.gen"  "$D/gen.txt"   >/dev/null 2>&1
  adb_safe_pull "/data/local/tmp/sl_$TAG.csv" "$D/sensors.csv" >/dev/null 2>&1
  T=$(grep -oE '"decode_tps": *[0-9.]+' $D/meta.json 2>/dev/null | grep -oE '[0-9.]+')
  echo "  [$TAG] tok/s=$T"
}

# interleaved across reps so session drift is shared, not loaded onto one arm
for r in 1 2 3; do
  cell v_r$r        gen $VA
  cell mukv_r$r     gen $MU
  cell sfown_r$r    gen $SF_OWN
  cell sh_r$r       gen $SH
done
# quality: one PPL cell per valid arm (sllm_handicap's output is corrupt by construction)
cell v_ppl       ppl $VA
cell mukv_ppl    ppl $MU
cell sfown_ppl   ppl $SF_OWN
echo SLLM_FAITHFUL_DONE
