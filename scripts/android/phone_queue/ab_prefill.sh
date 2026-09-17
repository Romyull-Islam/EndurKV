#!/system/bin/sh
# Prefill-cost diagnostic. Waits for the queue to finish its running cell and enter the
# cooling gate of the next one, pauses the queue there (nothing is mid-run), runs four
# cooled prefill-only cells that isolate the attention-capture readback cost, then
# restarts the queue, which skips every cell whose json exists.
RES=/data/local/tmp/endurkv/qres; LOG=$RES/ab.log
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
PIN="taskset f0 nice -n -20"
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
log "ab start: waiting for snapkv_own.json"
while [ ! -s $RES/snapkv_own.json ]; do sleep 20; done
log "snapkv_own done; waiting for the queue to enter a cooling gate"
n=0
while [ $n -lt 400 ]; do
  L=$(tail -1 $RES/queue.log)
  case "$L" in *running*) sleep 3; n=$((n+1)); continue;; *cooling*|*settle*) break;; esac
  sleep 3; n=$((n+1))
done
for p in $(ps -A -o PID,ARGS | grep 'queue_gpu.sh' | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 2
log "queue paused at: $(tail -1 $RES/queue.log)"
settle(){
  a=0
  while [ $a -lt 3 ]; do
    a=$((a+1))
    . /data/local/tmp/endurkv/scripts/cool_gate.sh; cool_ddr36 >/dev/null 2>&1
    prev=999; same=0; i=0
    while [ $i -lt 90 ]; do
      d=$(( $(cat /sys/class/thermal/thermal_zone47/temp)/100 )); diff=$((d-prev)); [ $diff -lt 0 ] && diff=$((-diff))
      if [ $diff -le 3 ]; then same=$((same+1)); else same=0; fi
      [ $same -ge 3 ] && break; prev=$d; sleep 10; i=$((i+1))
    done
    sd=$(( $(cat /sys/class/thermal/thermal_zone47/temp)/1000 )); sb=$(( $(cat /sys/class/thermal/thermal_zone93/temp)/1000 ))
    log "  settle ddr=${sd}C batt=${sb}C"
    [ $sd -le 35 ] && [ $sb -le 33 ] && return 0
  done; return 1; }
cell(){
  TAG=$1; GL=$2; shift 2
  [ -s $RES/$TAG.json ] && { log "[$TAG] cached"; return; }
  log "[$TAG] cooling"; settle || log "[$TAG] hot after 3 settles, running anyway"
  rm -f $RES/$TAG.sensors.csv
  nohup sh /data/local/tmp/sample_sensors.sh --out $RES/$TAG.sensors.csv --hz 2 >/dev/null 2>&1 &
  log "[$TAG] running"
  cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $M --prompt $P --prompt-id $TAG \
    --eval-mode gen --max-tokens 8 --ignore-eos --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers $GL \
    --greedy --cache-type-k f16 --cache-type-v f16 "$@" --n-batch 512 --n-ubatch 64 \
    --out-meta $RES/$TAG.json --out-gen $RES/$TAG.gen --out-csv /dev/null > /dev/null 2> $RES/$TAG.err
  pkill -f sample_sensors 2>/dev/null
  log "[$TAG] done $(grep -oE 'prefill=[0-9.]+ms' $RES/$TAG.err | tail -1)"
}
cell ab_gpu_snapkv_w16        99 --policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024
cell ab_gpu_snapkv_w64_noread 99 --policy snapkv --obs-window 64 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024 --cb-eval-noread
cell ab_cpu_vanilla            0 --policy vanilla --k-nominal 1024
cell ab_cpu_snapkv_w64         0 --policy snapkv --obs-window 64 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024
log "ab done; restarting queue"
nohup sh /data/local/tmp/endurkv/queue_gpu.sh > /data/local/tmp/endurkv/queue_nohup.log 2>&1 &
