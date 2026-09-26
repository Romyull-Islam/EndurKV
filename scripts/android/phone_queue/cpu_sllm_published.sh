#!/system/bin/sh
# E2c/E2d, phone-resident: StreamingLLM at its published budget (4 sinks + 2000) on the phone CPU with
# the fused kernel and in-place compaction (the reference implementation concatenates survivors).
# The July binary cannot run it that way (it forces FA off for StreamingLLM and has no compaction flag),
# so both StreamingLLM and a same-build full-cache run use bin_cpu_cur; ratios are taken within the build.
# Same prompt, threads and generation length as the July CPU table; cooled per cell, charging off, 5 Hz sensors.
RES=/data/local/tmp/endurkv/qres_cpu; mkdir -p $RES; LOG=$RES/cpu.log
CB=/data/local/tmp/endurkv/bin_cpu_cur
P=/data/local/tmp/endurkv/logs/natcpu_20260718_085317/prompt.txt
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "cpu start"
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
  TAG=$1; M=$2; shift 2
  [ -s $RES/$TAG.json ] && { log "[$TAG] cached"; return; }
  log "[$TAG] cooling"; settle || log "[$TAG] hot after 3 settles, running anyway"
  rm -f $RES/$TAG.sensors.csv
  nohup sh /data/local/tmp/sample_sensors.sh --out $RES/$TAG.sensors.csv --hz 5 >/dev/null 2>&1 &
  log "[$TAG] running"
  cd $CB && LD_LIBRARY_PATH=$CB ./eviction_bench --prompt $P --prompt-id $TAG --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $M --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    "$@" \
    --out-meta $RES/$TAG.json --out-csv /dev/null --out-gen $RES/$TAG.gen > /dev/null 2> $RES/$TAG.err
  pkill -f sample_sensors 2>/dev/null
  log "[$TAG] done $(grep -oE '"decode_tps": *[0-9.]+' $RES/$TAG.json 2>/dev/null)"
}
ML=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
MB=/data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf
SL="--policy streamingllm --n-sink 4 --k-nominal 2004 --compact-inplace"
cell llama_vanilla_cur $ML --policy vanilla --k-nominal 1024
cell llama_sllm_pub $ML $SL
cell bonsai_vanilla_cur $MB --policy vanilla --k-nominal 1024
cell bonsai_sllm_pub $MB $SL
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "CPU_DONE"
