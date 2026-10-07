#!/system/bin/sh
# Phone-resident GPU campaign runner. Runs as root, detached, needs no host.
# Protocol is run_streamingllm_faithful.sh's cell(): cool gate, 2 Hz sensors,
# pinned eviction_bench, ctx 16384, f16 KV, seed 42. Cells whose result json
# exists in RES are skipped, so the queue can be restarted at any time.
Q=/data/local/tmp/endurkv/queue_gpu.txt
RES=/data/local/tmp/endurkv/qres; mkdir -p $RES
LOG=$RES/queue.log
BIN=/data/local/tmp/ukv_n3
M=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
P=/data/local/tmp/endurkv/corpora/prompt_12k.txt
E=/data/local/tmp/endurkv/eval_data/dis.txt
PIN="taskset f0 nice -n -20"
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
pkill -f eviction_bench 2>/dev/null; pkill -f sample_sensors 2>/dev/null; sleep 2
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "queue start"
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
grep -vE '^[[:space:]]*(#|$)' $Q | while read -r TAG MODE STEPS FLAGS; do
  [ -s $RES/$TAG.json ] && { log "[$TAG] cached"; continue; }
  if [ "$MODE" = ppl ]; then EX="--eval-mode ppl --eval-text $E"; else EX="--eval-mode gen --max-tokens $STEPS --ignore-eos"; fi
  # cells-only runs (<= 64-token decode) measure what survives prefill, which does not
  # depend on temperature, so they skip the gate
  if [ "$STEPS" -le 64 ]; then log "[$TAG] no gate, cells-only cell"; else log "[$TAG] cooling"; settle || log "[$TAG] hot after 3 settles, running anyway"; fi
  rm -f $RES/$TAG.sensors.csv
  nohup sh /data/local/tmp/sample_sensors.sh --out $RES/$TAG.sensors.csv --hz 2 >/dev/null 2>&1 &
  log "[$TAG] running"
  cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $M --prompt $P --prompt-id $TAG $EX \
    --ctx-size 16384 --seed 42 --threads 4 --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 \
    $FLAGS --n-batch 512 --n-ubatch 64 --out-meta $RES/$TAG.json --out-gen $RES/$TAG.gen --out-csv /dev/null \
    > /dev/null 2> $RES/$TAG.err
  pkill -f sample_sensors 2>/dev/null
  T=$(grep -oE '"decode_tps": *[0-9.]+' $RES/$TAG.json 2>/dev/null | grep -oE '[0-9.]+')
  log "[$TAG] done tok/s=${T:-none}"
done
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "QUEUE_DONE"
