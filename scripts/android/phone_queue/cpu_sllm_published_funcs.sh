# shared definitions for the CPU runners (sourced)
RES=/data/local/tmp/endurkv/qres_cpu; mkdir -p $RES; LOG=$RES/cpu.log
CB=/data/local/tmp/endurkv/bin_cpu_cur
P=/data/local/tmp/endurkv/logs/natcpu_20260718_085317/prompt.txt
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
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
